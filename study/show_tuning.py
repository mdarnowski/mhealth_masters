import warnings

from optuna.exceptions import ExperimentalWarning
from silence_tensorflow import silence_tensorflow

warnings.filterwarnings("ignore", category=ExperimentalWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
silence_tensorflow("WARNING")

from IPython.display import display

from study.hyper_study import ActivityRecognitionOptunaStudy
from util import vis
from util.vis import plot_evaluation_results


def perform_study_show_graphs(
    hyperparameter_set,
    n_trials=50,
    epochs=10,
    sampler=None,
    optimize_for_loss=True,
    use_lr_scheduler=False,
    use_pruner=False,
):
    study = ActivityRecognitionOptunaStudy(
        epochs=epochs,
        hyperparameter_suggestions=hyperparameter_set,
        n_trials=n_trials,
        sampler=sampler,
        optimize_for_loss=optimize_for_loss,
        use_lr_scheduler=use_lr_scheduler,
        use_pruner=use_pruner,
    )

    best_model = study.perform_trials()

    vis.show_optuna_study_images(study.get_optuna_study())

    vis.plot_val_history(study.get_best_model_history())

    display(vis.get_model_vis(best_model))

    plot_evaluation_results(best_model, verbose=0)

    return best_model
