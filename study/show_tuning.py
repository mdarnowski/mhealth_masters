import io
import warnings

import numpy as np
from IPython.display import Image
from optuna.exceptions import ExperimentalWarning
from silence_tensorflow import silence_tensorflow

# clean notebooks
warnings.filterwarnings("ignore", category=ExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
silence_tensorflow("WARNING")


import matplotlib.pyplot as plt
from IPython.display import display
from plot_keras_history import plot_history

from study.hyper_study import ActivityRecognitionOptunaStudy
from util import vis
from util.vis import plot_evaluation_results


def perform_study_show_graphs(
    hyperparameter_set, n_trials=50, epochs=10, sampler=None, max_epochs=30
):
    study = ActivityRecognitionOptunaStudy(
        epochs=epochs,
        hyperparameter_suggestions=hyperparameter_set,
        n_trials=n_trials,
        sampler=sampler,
    )

    optuna_study = study.perform_trials()

    vis.show_optuna_study_images(optuna_study)

    best_model = study.recreate_model_from_study(optuna_study)

    history = best_model.train_model(max_epochs=max_epochs, verbose=0)
    plot_history(history, title="Optimized Model Retraining")
    plt.show()

    # display(vis.get_model_vis(best_model))
    vis.show_model_from_console(best_model)

    plot_evaluation_results(best_model, verbose=0)

    return best_model


from optuna.samplers import CmaEsSampler

if __name__ == "__main__":
    hyperparameter_set = {
        "batch_size": lambda trial: trial.suggest_int("batch_size", 16, 64),
        "optimizer": "adamax",
        "l_rate": lambda trial: trial.suggest_float("l_rate", 1e-5, 1e-3, log=True),
        "recurrent_type": "GRU",
        "n_recurrent_layers": 3,
        "recurrent_units": lambda trial, i: trial.suggest_int(
            f"recurrent_units_{i}", 32, 256
        ),
        "n_dense_layers": 0,
        "dropout_rate": lambda trial: trial.suggest_float("dropout_rate", 0.1, 0.5),
        "n_conv_layers": 3,
        "conv_filters": lambda trial, i: trial.suggest_int(
            f"conv_filters_{i}", 16, 128
        ),
        "kernel_size": lambda trial, i: trial.suggest_int(
            f"kernel_size_{i}", 3, 7, step=2
        ),
        "use_batch_norm": False,
        "pooling_type": "average",
        "pool_after_each": False,
        "pool_size": lambda trial: trial.suggest_int("pool_size", 2, 5),
    }

    perform_study_show_graphs(
        hyperparameter_set=hyperparameter_set,
        n_trials=1,
        epochs=1,
        sampler=CmaEsSampler(),
        max_epochs=1,
    )
