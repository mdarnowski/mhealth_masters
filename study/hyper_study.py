import warnings

from optuna.exceptions import ExperimentalWarning
from silence_tensorflow import silence_tensorflow

# clean notebooks
warnings.filterwarnings("ignore", category=ExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
silence_tensorflow("WARNING")

import optuna
from keras import layers, optimizers

# from optuna_integration import KerasPruningCallback
from study.a_rec import ActivityRecognitionModel

default_params = {
    "segment_size": 75,
    "batch_size": 25,
    "n_shifts": 5,
    "optimizer": "adam",
    "l_rate": 1e-3,
    "recurrent_type": "GRU",
    "n_recurrent_layers": 1,
    "recurrent_units": [128],
    "n_dense_layers": 0,
    "dense_units": [128],
    "dropout_rate": 0.3,
    "n_conv_layers": 0,
    "conv_filters": [32],
    "use_batch_norm": False,
    "pooling_type": "max",
    "pool_after_each": True,
    "kernel_size": 3,
    "pool_size": 2,
}

# Hyperparameter set 1
hyperparameter_set_1 = {
    "segment_size": 80,
    "batch_size": 50,
    "optimizer": lambda trial: trial.suggest_categorical(
        "optimizer", ["adam", "sgd", "rmsprop", "adamax", "nadam"]
    ),
    "n_shifts": 5,
    "l_rate": lambda trial: trial.suggest_float("l_rate", 1e-4, 1e-2, log=True),
    "n_conv_layers": lambda trial: trial.suggest_int("n_conv_layers", 1, 3),
    "conv_filters": lambda trial, i: trial.suggest_int(
        f"conv_filters_{i}", 32, 128, step=32
    ),
    # 'kernel_size': 3,
    "recurrent_type": lambda trial: trial.suggest_categorical(
        "recurrent_type", ["GRU", "LSTM"]
    ),
    "n_recurrent_layers": lambda trial: trial.suggest_int("n_recurrent_layers", 1, 3),
    "recurrent_units": lambda trial, i: trial.suggest_int(
        f"recurrent_units_{i}", 32, 128, step=32
    ),
    "n_dense_layers": lambda trial: trial.suggest_int("n_dense_layers", 0, 1),
    "dense_units": lambda trial, i: trial.suggest_int(
        f"dense_units_{i}", 64, 256, step=64
    ),
    "dropout_rate": lambda trial: trial.suggest_float("dropout_rate", 0.2, 0.5),
    "use_batch_norm": lambda trial: trial.suggest_categorical(
        "use_batch_norm", [True, False]
    ),
    "pooling_type": lambda trial: trial.suggest_categorical(
        "pooling_type", ["max", "average"]
    ),
    "pool_after_each": lambda trial: trial.suggest_categorical(
        "pool_after_each", [True, False]
    ),
}

# Example hyperparameter set 2
hyperparameter_set_2 = {
    "segment_size": 100,  # Fixed value
    "batch_size": lambda trial: trial.suggest_int("batch_size", 32, 256, step=32),
    "n_shifts": 7,  # Fixed value
    "optimizer": lambda trial: trial.suggest_categorical(
        "optimizer", ["adam", "rmsprop"]
    ),
    "l_rate": 1e-3,  # Fixed value
    "n_conv_layers": 2,  # Fixed value
    "conv_filters": lambda trial, i: trial.suggest_int(
        f"conv_filters_{i}", 64, 256, step=64
    ),
    "kernel_size": 3,  # Fixed value for kernel size
    "n_recurrent_layers": 2,  # Updated to n_recurrent_layers
    "recurrent_units": [
        128,
        64,
    ],  # Fixed value (changed from lstm_units to recurrent_units)
    "n_dense_layers": lambda trial: trial.suggest_int("n_dense_layers", 1, 3),
    "dense_units": lambda trial, i: trial.suggest_int(
        f"dense_units_{i}", 128, 512, step=128
    ),
    "dropout_rate": lambda trial: trial.suggest_float(
        "dropout_rate", 0.3, 0.7, step=0.1
    ),
    "use_batch_norm": True,  # Fixed value
    "pooling_type": lambda trial: trial.suggest_categorical(
        "pooling_type", ["max", "average"]
    ),
    "pool_after_each": False,  # Fixed value
}


class ActivityRecognitionOptunaStudy:
    def __init__(
        self, epochs=10, hyperparameter_suggestions=None, n_trials=50, sampler=None
    ):
        self.epochs = epochs
        self.n_trials = n_trials
        self.sampler = sampler if sampler is not None else optuna.samplers.TPESampler()
        self.hyperparameter_suggestions = hyperparameter_suggestions or {}

    def _get_param(self, trial, param_name, default_value):
        """Helper function to get a parameter from suggestions or defaults."""
        suggestion = self.hyperparameter_suggestions.get(param_name)
        if callable(suggestion):
            if param_name in ["conv_filters", "recurrent_units", "dense_units"]:
                num_layers_param = "n_" + param_name.split("_")[0] + "_layers"
                num_layers = self._get_param(trial, num_layers_param, 0)
                return [suggestion(trial, i) for i in range(num_layers)]
            return suggestion(trial)
        elif suggestion is not None:
            return suggestion
        return default_value

    def _suggest_hyperparameters(self, trial):
        params = {
            "segment_size": self._get_param(
                trial, "segment_size", default_params["segment_size"]
            ),
            "batch_size": self._get_param(
                trial, "batch_size", default_params["batch_size"]
            ),
            "n_shifts": self._get_param(trial, "n_shifts", default_params["n_shifts"]),
            "optimizer": self._get_param(
                trial, "optimizer", default_params["optimizer"]
            ),
            "l_rate": self._get_param(trial, "l_rate", default_params["l_rate"]),
            "recurrent_type": self._get_param(
                trial, "recurrent_type", default_params["recurrent_type"]
            ),
            "n_recurrent_layers": self._get_param(
                trial, "n_recurrent_layers", default_params["n_recurrent_layers"]
            ),
            "recurrent_units": self._get_param(
                trial, "recurrent_units", default_params["recurrent_units"]
            ),
            "n_dense_layers": self._get_param(
                trial, "n_dense_layers", default_params["n_dense_layers"]
            ),
            "dense_units": self._get_param(
                trial, "dense_units", default_params["dense_units"]
            ),
            "dropout_rate": self._get_param(
                trial, "dropout_rate", default_params["dropout_rate"]
            ),
            "n_conv_layers": self._get_param(
                trial, "n_conv_layers", default_params["n_conv_layers"]
            ),
            "conv_filters": self._get_param(
                trial, "conv_filters", default_params["conv_filters"]
            ),
            "use_batch_norm": self._get_param(
                trial, "use_batch_norm", default_params["use_batch_norm"]
            ),
            "pooling_type": self._get_param(
                trial, "pooling_type", default_params["pooling_type"]
            ),
            "pool_after_each": self._get_param(
                trial, "pool_after_each", default_params["pool_after_each"]
            ),
            "kernel_size": self._get_param(
                trial, "kernel_size", default_params["kernel_size"]
            ),
            "pool_size": default_params["pool_size"],
        }
        return params

    def _build_model(self, params):
        model = ActivityRecognitionModel(
            segment_size=params["segment_size"],
            batch_size=params["batch_size"],
            n_shifts=params["n_shifts"],
            epochs=self.epochs,
        )
        model.add_input_layer()

        for i in range(params["n_conv_layers"]):
            model.add(
                layers.Conv1D(
                    filters=params["conv_filters"][i],
                    kernel_size=params["kernel_size"],
                    padding="same",
                )
            )
            if params["use_batch_norm"]:
                model.add(layers.BatchNormalization())
            model.add(layers.ReLU())
            if params["pool_after_each"]:
                if params["pooling_type"] == "max":
                    model.add(
                        layers.MaxPool1D(pool_size=params["pool_size"])
                    )  # Use fixed pool size from params
                else:
                    model.add(
                        layers.AveragePooling1D(pool_size=params["pool_size"])
                    )  # Use fixed pool size from params

        if not params["pool_after_each"] and params["n_conv_layers"] > 0:
            if params["pooling_type"] == "max":
                model.add(
                    layers.MaxPool1D(pool_size=params["pool_size"])
                )  # Use fixed pool size from params
            else:
                model.add(
                    layers.AveragePooling1D(pool_size=params["pool_size"])
                )  # Use fixed pool size from params

        for i in range(params["n_recurrent_layers"]):  # Updated to n_recurrent_layers
            return_sequences = i < params["n_recurrent_layers"] - 1
            if params["recurrent_type"] == "LSTM":
                model.add(
                    layers.LSTM(
                        units=params["recurrent_units"][i],
                        return_sequences=return_sequences,
                    )
                )  # Use units=params['recurrent_units'][i]
            else:
                model.add(
                    layers.GRU(
                        units=params["recurrent_units"][i],
                        return_sequences=return_sequences,
                    )
                )  # Use units=params['recurrent_units'][i]
            if params["use_batch_norm"]:
                model.add(layers.BatchNormalization())

        for i in range(params["n_dense_layers"]):
            model.add(
                layers.Dense(units=params["dense_units"][i], activation="relu")
            )  # Use units=params['dense_units'][i]
            if params["use_batch_norm"]:
                model.add(layers.BatchNormalization())
            model.add(layers.Dropout(params["dropout_rate"]))

        model.add_output_layer()

        optimizer = None
        if params["optimizer"] == "adam":
            optimizer = optimizers.Adam(learning_rate=params["l_rate"])
        elif params["optimizer"] == "sgd":
            optimizer = optimizers.SGD(learning_rate=params["l_rate"])
        elif params["optimizer"] == "rmsprop":
            optimizer = optimizers.RMSprop(learning_rate=params["l_rate"])
        elif params["optimizer"] == "adamax":
            optimizer = optimizers.Adamax(learning_rate=params["l_rate"])
        elif params["optimizer"] == "nadam":
            optimizer = optimizers.Nadam(learning_rate=params["l_rate"])

        model.compile_model(optimizer=optimizer)

        return model

    def objective(self, trial):
        params = self._suggest_hyperparameters(trial)
        try:
            model = self._build_model(params)
        except ValueError as e:
            print(f"Trial {trial.number} failed due to: {e}")
            return 0

        # pruning_callback = KerasPruningCallback(trial, 'val_sparse_categorical_accuracy')
        history = model.train_model(verbose=0)
        val_accuracy = max(history.history["val_sparse_categorical_accuracy"])
        return val_accuracy

    def perform_trials(self):
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        study.optimize(self.objective, n_trials=self.n_trials)
        return study

    def recreate_model_from_study(self, study):
        best_params = study.best_params
        filled_params = {**default_params, **best_params}
        for param in ["recurrent_units", "conv_filters", "dense_units"]:
            num_layers_param = "n_" + param.split("_")[0] + "_layers"
            if num_layers_param in filled_params:
                num_layers = filled_params[num_layers_param]
                filled_params[param] = [
                    filled_params.get(f"{param}_{i}", default_params[param][i])
                    for i in range(num_layers)
                ]
            else:
                filled_params[param] = default_params[param]

        return self._build_model(filled_params)
