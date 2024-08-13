import optuna
from keras import layers, optimizers

from study.a_rec import ActivityRecognitionModel
from util import vis

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
    "kernel_size": [3],
    "pool_size": 2,
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
            if param_name in ["conv_filters", "kernel_size"]:
                num_conv_layers = self._get_param(trial, "n_conv_layers", 0)
                # Ensure the list is correctly sized based on the number of convolutional layers
                return [suggestion(trial, i) for i in range(num_conv_layers)]
            elif param_name in ["recurrent_units", "dense_units"]:
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
            "kernel_size": self._get_param(
                trial, "kernel_size", default_params["kernel_size"]
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
            "pool_size": self._get_param(
                trial, "pool_size", default_params["pool_size"]
            ),
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
                    kernel_size=params["kernel_size"][
                        i
                    ],  # Different kernel sizes for each layer
                    padding="same",
                )
            )
            if params["use_batch_norm"]:
                model.add(layers.BatchNormalization())
            model.add(layers.ReLU())
            model.add(layers.Dropout(params["dropout_rate"]))
            if params["pool_after_each"]:
                if params["pooling_type"] == "max":
                    model.add(layers.MaxPool1D(pool_size=params["pool_size"]))
                else:
                    model.add(layers.AveragePooling1D(pool_size=params["pool_size"]))

        if not params["pool_after_each"] and params["n_conv_layers"] > 0:
            if params["pooling_type"] == "max":
                model.add(layers.MaxPool1D(pool_size=params["pool_size"]))
            else:
                model.add(layers.AveragePooling1D(pool_size=params["pool_size"]))

        for i in range(params["n_recurrent_layers"]):
            return_sequences = i < params["n_recurrent_layers"] - 1
            if params["recurrent_type"] == "LSTM":
                model.add(
                    layers.LSTM(
                        units=params["recurrent_units"][i],
                        return_sequences=return_sequences,
                    )
                )
            else:
                model.add(
                    layers.GRU(
                        units=params["recurrent_units"][i],
                        return_sequences=return_sequences,
                    )
                )

        for i in range(params["n_dense_layers"]):
            model.add(layers.Dense(units=params["dense_units"][i], activation="relu"))
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

        history = model.train_model(verbose=0)
        val_accuracy = max(history.history["val_sparse_categorical_accuracy"])
        return val_accuracy

    def perform_trials(self):
        study = optuna.create_study(direction="maximize", sampler=self.sampler)
        study.optimize(self.objective, n_trials=self.n_trials)
        return study

    def recreate_model_from_study(self, study):
        best_params = study.best_params

        # Initialize filled_params with default parameters
        filled_params = {**default_params}

        # First, ensure fixed hyperparameters from hyperparameter_suggestions are included
        for param_name, suggestion in self.hyperparameter_suggestions.items():
            if param_name not in filled_params or not isinstance(
                filled_params[param_name], list
            ):
                if callable(suggestion):
                    # Skip callable as they are handled by Optuna
                    continue
                else:
                    filled_params[param_name] = suggestion

        # Process and fill layer-specific parameters like recurrent_units, conv_filters, kernel_size, and dense_units
        for param_prefix in [
            "recurrent_units",
            "conv_filters",
            "kernel_size",
            "dense_units",
        ]:
            param_list = []
            layer_index = 0
            while f"{param_prefix}_{layer_index}" in best_params:
                param_list.append(best_params[f"{param_prefix}_{layer_index}"])
                layer_index += 1

            # If we found any layer-specific params, update filled_params
            if param_list:
                filled_params[param_prefix] = param_list
                filled_params[f"n_{param_prefix.split('_')[0]}_layers"] = len(
                    param_list
                )

        filled_params = {**filled_params, **best_params}

        return self._build_model(filled_params)
