import matplotlib.pyplot as plt
from IPython.display import display
from plot_keras_history import plot_history

from study.hyper_study import ActivityRecognitionOptunaStudy
from util import vis
from util.vis import plot_evaluation_results


def perform_study_show_graphs(hyperparameter_set, n_trials=50, epochs=10, sampler=None):
    study = ActivityRecognitionOptunaStudy(
        epochs=epochs,
        hyperparameter_suggestions=hyperparameter_set,
        n_trials=n_trials,
        sampler=sampler,
    )

    optuna_study = study.perform_trials()

    vis.show_optuna_study_images(optuna_study)

    best_model = study.recreate_model_from_study(optuna_study)

    history = best_model.train_model(max_epochs=30, verbose=0)
    plot_history(history, title="Optimized Model Retraining")
    plt.show()

    display(vis.get_model_vis(best_model))
    plot_evaluation_results(best_model, verbose=0)

    return best_model
