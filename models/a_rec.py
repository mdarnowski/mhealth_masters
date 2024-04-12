import math
import random

import tensorflow as tf
from keras import Input, Sequential
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split

from data_layer import activity_data as act
from data_layer import sensor_data as sen
from data_layer import session_data as ses
from database.tools import my_conn


def _split_dataset(test_size=0.1, val_size=0.1, seed=42):
    ids, labels = ses.get_session_label_ids()

    train_val_ids, test_ids, train_val_labels, _ = train_test_split(
        ids, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids,
        train_val_labels,
        test_size=val_size / (1 - test_size),
        stratify=train_val_labels,
        random_state=seed,
    )
    return train_ids, val_ids, test_ids


class ActivityRecognitionModel(Sequential):
    def __init__(
        self,
        segment_size,
        batch_size,
        n_shifts=1,
        incl_acc=True,
        incl_gyro=True,
        incl_mag=True,
        incl_ecg=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.segment_size = segment_size
        self.batch_size = batch_size
        self.n_shifts = n_shifts
        self.incl_acc = incl_acc
        self.incl_gyro = incl_gyro
        self.incl_mag = incl_mag
        self.incl_ecg = incl_ecg
        self.n_features = self._count_features()
        self.train_ids, self.val_ids, self.test_ids = _split_dataset()
        with my_conn.get_db_session() as db:
            self.n_classes = act.count_activity(db)

    def train_model(self):
        train, train_steps = self._create_dataset(self.train_ids)
        val, val_steps = self._create_dataset(self.val_ids)

        early_stopping = EarlyStopping(
            monitor="val_accuracy",
            patience=2,
            restore_best_weights=True,
            min_delta=0.01,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.3, patience=1, min_lr=0.0001, cooldown=0
        )

        history = self.fit(
            train,
            steps_per_epoch=train_steps,
            validation_data=val,
            validation_steps=val_steps,
            callbacks=[early_stopping, reduce_lr],
            epochs=20,
        )

        return history

    def evaluate_model(self, verbose=1):
        test_dataset, test_steps = self._create_dataset(self.test_ids)
        results = self.evaluate(test_dataset, steps=test_steps, verbose=verbose)
        test_loss, test_accuracy = results[0], results[1]
        tf.print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    def add_input_layer(self):
        self.add(Input(shape=(self.segment_size, self.n_features)))

    def add_output_layer(self, activation="softmax"):
        self.add(Dense(self.n_classes, activation=activation))

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

    def _count_features(self):
        columns = sen.get_sensor_columns(
            incl_acc=self.incl_acc,
            incl_gyro=self.incl_gyro,
            incl_mag=self.incl_mag,
            incl_ecg=self.incl_ecg,
        )
        return len(columns)

    def _get_shifts_arr(self):
        return [i * self.segment_size // self.n_shifts for i in range(self.n_shifts)]

    def _create_dataset(self, ids):
        shifts = self._get_shifts_arr()

        def data_generator():
            random.shuffle(ids)
            with my_conn.get_db_session() as db:
                for shift in shifts:
                    for session_id in ids:
                        session_data = sen.fetch_sensor_data(
                            db,
                            session_id,
                            self.incl_acc,
                            self.incl_gyro,
                            self.incl_mag,
                            self.incl_ecg,
                        )
                        label = act.fetch_activity_label(db, session_id)
                        for offset in range(
                            shift, len(session_data), self.segment_size
                        ):
                            end = offset + self.segment_size
                            if end <= len(session_data):
                                segment = session_data[offset:end]
                                yield segment, label

        dataset = (
            tf.data.Dataset.from_generator(
                data_generator,
                output_signature=(
                    tf.TensorSpec(
                        shape=(self.segment_size, self.n_features), dtype=tf.float32
                    ),
                    tf.TensorSpec(shape=(), dtype=tf.int32),
                ),
            )
            .batch(self.batch_size)
            .prefetch(tf.data.experimental.AUTOTUNE)
            .repeat()
        )
        steps_per_epoch = self._calc_steps_per_epoch(ids)

        return dataset, steps_per_epoch
