import optuna
from keras import layers

from study.a_rec import ActivityRecognitionModel

default_params = {
    "segment_size": 50,
    "batch_size": 35,
    "n_shifts": 4,
    "optimizer": "adam",
    "l_rate": 1e-3,
    "recurrent_type": "GRU",
    "n_recurrent_layers": 1,
    "recurrent_units": [128],
    "n_dense_layers": 0,
    "dense_units": [128],
    "dropout_rates": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "n_conv_layers": 0,
    "conv_filters": [32],
    "use_batch_norm": False,
    "pooling_type": "max",
    "pool_after_each": True,
    "kernel_size": [3, 3, 3, 3],
    "pool_size": 2,
}


class ActivityRecognitionOptunaStudy:
    def __init__(
        self,
        epochs=10,
        hyperparameter_suggestions=None,
        n_trials=50,
        sampler=None,
        optimize_for_loss=False,
    ):
        self.best_history = None
        self.study = None
        self.epochs = epochs
        self.n_trials = n_trials
        self.sampler = sampler if sampler is not None else optuna.samplers.TPESampler()
        self.hyperparameter_suggestions = hyperparameter_suggestions or {}
        self.best_model = None
        self.optimize_for_loss = optimize_for_loss
        self.best_val_metric = float("inf") if optimize_for_loss else -float("inf")

    def _get_param(self, trial, param_name, default_value):
        """Helper function to get a parameter from suggestions or defaults."""
        suggestion = self.hyperparameter_suggestions.get(param_name)
        if callable(suggestion):
            if param_name == "dropout_rates":
                num_layers = sum(
                    [
                        self._get_param(trial, "n_conv_layers", 0),
                        self._get_param(trial, "n_recurrent_layers", 0),
                        self._get_param(trial, "n_dense_layers", 0),
                    ]
                )
                return [suggestion(trial, i) for i in range(num_layers)]
            elif param_name in ["conv_filters", "kernel_size"]:
                num_conv_layers = self._get_param(trial, "n_conv_layers", 0)
                return [suggestion(trial, i) for i in range(num_conv_layers)]
            elif param_name in ["recurrent_units"]:
                num_conv_layers = self._get_param(trial, "n_recurrent_layers", 0)
                return [suggestion(trial, i) for i in range(num_conv_layers)]
            elif param_name in ["dense_units"]:
                num_conv_layers = self._get_param(trial, "n_dense_layers", 0)
                return [suggestion(trial, i) for i in range(num_conv_layers)]
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
            "dropout_rates": self._get_param(
                trial, "dropout_rates", default_params["dropout_rates"]
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

    @staticmethod
    def _add_conditional_dropout(model, dropout_rate):
        """Adds a Dropout layer to the model if the dropout rate is greater than 0.0."""
        if dropout_rate > 0.0:
            model.add(layers.Dropout(dropout_rate))

    def _build_model(self, params):
        model = ActivityRecognitionModel(
            segment_size=params["segment_size"],
            batch_size=params["batch_size"],
            n_shifts=params["n_shifts"],
            epochs=self.epochs,
            optimizer_name=params["optimizer"],
            l_rate=params["l_rate"],
        )
        model.add_input_layer()

        dropout_rates = params["dropout_rates"].copy()

        for i in range(params["n_conv_layers"]):
            model.add(
                layers.Conv1D(
                    filters=params["conv_filters"][i],
                    kernel_size=params["kernel_size"][i],
                    padding="same",
                )
            )
            if params["use_batch_norm"]:
                model.add(layers.BatchNormalization())
            model.add(layers.ReLU())

            self._add_conditional_dropout(model, dropout_rates.pop(0))

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

            self._add_conditional_dropout(model, dropout_rates.pop(0))

        for i in range(params["n_dense_layers"]):
            model.add(layers.Dense(units=params["dense_units"][i], activation="relu"))
            self._add_conditional_dropout(model, dropout_rates.pop(0))

        model.add_output_layer()

        model.compile_model()

        return model

    def objective(self, trial):
        params = self._suggest_hyperparameters(trial)
        try:
            model = self._build_model(params)
        except ValueError as e:
            print(f"Trial {trial.number} failed due to: {e}")
            return float("inf") if self.optimize_for_loss else 0

        history = model.train_model(verbose=0)
        if self.optimize_for_loss:
            val_metric = min(history.history["val_loss"])
            best_improvement_condition = val_metric < self.best_val_metric
        else:
            val_metric = max(history.history["val_sparse_categorical_accuracy"])
            best_improvement_condition = val_metric > self.best_val_metric

        if best_improvement_condition:
            self.best_val_metric = val_metric
            self.best_model = model
            self.best_history = history

        return val_metric

    def perform_trials(self):
        direction = "minimize" if self.optimize_for_loss else "maximize"
        self.study = optuna.create_study(direction=direction, sampler=self.sampler)
        self.study.optimize(self.objective, n_trials=self.n_trials)
        return self.best_model

    def get_optuna_study(self):
        return self.study

    def get_best_model_history(self):
        return self.best_history
