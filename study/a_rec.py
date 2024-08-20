import math
from collections import defaultdict

from silence_tensorflow import silence_tensorflow

silence_tensorflow("WARNING")  # clean notebooks
import tensorflow as tf
from keras import Input, Sequential
from keras.src.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Dense
from keras.src.metrics import SparseTopKCategoricalAccuracy
from keras.src.optimizers import Adam
from sklearn.model_selection import train_test_split

from data_layer import activity_data as act
from data_layer import sensor_data as sen
from data_layer import session_data as ses
from database.tools import my_conn


def _split_dataset(test_size=0.1, val_size=0.1, seed=42):
    session_ids, labels, subject_ids = ses.get_session_label_ids()

    subject_to_sessions = defaultdict(list)
    subject_to_labels = defaultdict(list)

    for session_id, label, subject_id in zip(session_ids, labels, subject_ids):
        subject_to_sessions[subject_id].append(session_id)
        subject_to_labels[subject_id].append(label)

    unique_subjects = list(subject_to_sessions.keys())
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=seed
    )

    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=val_size / (1 - test_size), random_state=seed
    )

    train_ids = [
        session_id
        for subject in train_subjects
        for session_id in subject_to_sessions[subject]
    ]
    val_ids = [
        session_id
        for subject in val_subjects
        for session_id in subject_to_sessions[subject]
    ]
    test_ids = [
        session_id
        for subject in test_subjects
        for session_id in subject_to_sessions[subject]
    ]

    label_dict = dict(zip(session_ids, labels))

    return train_ids, val_ids, test_ids, label_dict


class ActivityRecognitionModel(Sequential):
    def __init__(
        self,
        segment_size,
        batch_size,
        n_shifts=1,
        l_rate=0.001,
        epochs=30,
        optimizer_name="adam",
        use_lr_scheduler=False,
        scheduler_factor=0.5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.use_lr_scheduler = use_lr_scheduler
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.n_shifts = n_shifts
        self.n_features = len(sen.get_sensor_columns())
        self.scheduler_factor = scheduler_factor

        self.train_ids, self.val_ids, self.test_ids, self.label_dict = _split_dataset()
        with my_conn.get_db_session() as db:
            self.n_classes = act.count_activity(db)

        self.epochs = epochs

        if optimizer_name == "adam":
            self.optimizer = Adam(learning_rate=l_rate)
        elif optimizer_name == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=l_rate)
        elif optimizer_name == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=l_rate)
        elif optimizer_name == "adamax":
            self.optimizer = tf.keras.optimizers.Adamax(learning_rate=l_rate)
        elif optimizer_name == "nadam":
            self.optimizer = tf.keras.optimizers.Nadam(learning_rate=l_rate)

    def train_model(self, callbacks: list[Callback] = None, max_epochs=None, verbose=1):
        train, train_steps = self._create_dataset(self.train_ids)
        val, val_steps = self._create_dataset(self.val_ids)

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            min_delta=0.01,
        )

        if callbacks is not None:
            callbacks = [early_stopping] + callbacks
        else:
            callbacks = [early_stopping]

        if self.use_lr_scheduler:
            lr_scheduler = ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.scheduler_factor,
                patience=1,
                min_lr=1e-6,
            )

            callbacks.append(lr_scheduler)

        if max_epochs is None:
            max_epochs = self.epochs

        history = self.fit(
            train,
            steps_per_epoch=train_steps,
            validation_data=val,
            validation_steps=val_steps,
            callbacks=callbacks,
            epochs=max_epochs,
            verbose=verbose,
        )

        return history

    def evaluate_model(self, verbose=1):
        test_dataset, test_steps = self._create_dataset(self.test_ids)
        results = self.evaluate(test_dataset, steps=test_steps, verbose=verbose)

        metrics_names = ["loss", "sparse_categorical_accuracy", "sparse_top_3_accuracy"]

        return dict(zip(metrics_names, results))

    def compile_model(self):
        self.compile(
            optimizer=self.optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[
                "sparse_categorical_accuracy",
                SparseTopKCategoricalAccuracy(k=3, name="sparse_top_3_accuracy"),
            ],
        )

    def add_input_layer(self):
        self.add(Input(shape=(self.segment_size, self.n_features)))

    def add_output_layer(self):
        self.add(Dense(self.n_classes, activation="softmax"))

    def _calc_steps_per_epoch(self, session_ids):
        session_lengths = ses.fetch_all_session_lengths(session_ids)
        shift_steps = self._get_shifts_arr()

        total_segments = 0
        for session_id, length in session_lengths.items():
            for shift in shift_steps:
                fin_len = length - shift
                if fin_len >= self.segment_size:
                    total_segments += (fin_len - self.segment_size) // self.segment_size

        steps_per_epoch = math.ceil(total_segments / self.batch_size)
        return steps_per_epoch

    def _get_shifts_arr(self):
        return [i * self.segment_size // self.n_shifts for i in range(self.n_shifts)]

    def _create_dataset(self, ids):
        shifts = self._get_shifts_arr()

        def data_generator(session_id):
            label = self.label_dict[session_id]
            with my_conn.get_db_session() as db:
                session_data = sen.fetch_sensor_data(db, session_id)
            for shift in shifts:
                for offset in range(shift, len(session_data), self.segment_size):
                    end = offset + self.segment_size
                    if end <= len(session_data):
                        segment = session_data[offset:end]
                        yield segment, label

        dataset = tf.data.Dataset.from_tensor_slices(ids)
        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.from_generator(
                data_generator,
                args=(x,),
                output_signature=(
                    tf.TensorSpec(
                        shape=(self.segment_size, self.n_features), dtype=tf.float32
                    ),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                ),
            ),
            cycle_length=2,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        steps_per_epoch = self._calc_steps_per_epoch(ids)
        dataset = dataset.shuffle(
            buffer_size=self.batch_size * 4, reshuffle_each_iteration=True
        )
        dataset = dataset.repeat()
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, steps_per_epoch
