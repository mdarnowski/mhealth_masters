import numpy as np
import optuna.visualization as opt_vis
import plotly.io as pio
from IPython.core.display_functions import display
from IPython.display import Image
from jupyter_server.services.config import ConfigManager
from keras.src.utils import model_to_dot
from matplotlib import pyplot as plt

cm = ConfigManager()
cm.update(
    "notebook",
    {"ServerApp": {"iopub_data_rate_limit": 10000000, "rate_limit_window": 10.0}},
)


def get_model_vis(
    model,
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    expand_nested=True,
    dpi=100,
    show_layer_activations=True,
    show_trainable=False,
):
    dot = model_to_dot(
        model,
        show_shapes=show_shapes,
        show_dtype=show_dtype,
        show_layer_names=show_layer_names,
        expand_nested=expand_nested,
        dpi=dpi,
        show_layer_activations=show_layer_activations,
        show_trainable=show_trainable,
    )

    png_data = dot.create_png()
    image = Image(data=png_data)

    return image


def plotly_fig_to_png(fig):
    png_bytes = pio.to_image(fig, format="png")
    image = Image(data=png_bytes)
    return image


def show_optuna_study_images(study, suppress_warning=True):
    visualization_functions = {
        "Optimization History": opt_vis.plot_optimization_history,
        "Parallel Coordinate": opt_vis.plot_parallel_coordinate,
        "Parameter Importances": opt_vis.plot_param_importances,
        "Slice Plot": opt_vis.plot_slice,
    }

    for vis_name, vis_function in visualization_functions.items():
        try:
            fig = vis_function(study)

            if vis_name == "Parallel Coordinate":
                fig.update_layout(
                    width=1600,
                    height=800,
                    margin=dict(l=40, r=40, t=40, b=150),
                    xaxis_tickangle=-45,
                    font=dict(size=12),
                )

            if vis_name == "Slice Plot":
                fig.update_layout(
                    width=1800,
                    height=700,
                    margin=dict(l=40, r=40, t=40, b=200),
                    xaxis_tickangle=-45,
                    font=dict(size=12),
                )

            image = plotly_fig_to_png(fig)
            display(image)
        except Exception as e:
            if not suppress_warning:
                print(f"Could not generate {vis_name} due to: {e}")


def plot_evaluation_results(
    model, verbose=0, title="Final Model Evaluation on Test Set"
):
    evaluation_results = model.evaluate_model(verbose=verbose)

    metrics = ["loss", "sparse_categorical_accuracy", "sparse_top_3_accuracy"]
    values = [evaluation_results[metric] for metric in metrics]

    metrics.reverse()
    values.reverse()

    scaled_values = [
        np.log10(value + 1) if metric == "loss" else value
        for metric, value in zip(metrics, values)
    ]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    colors.reverse()

    fig, ax = plt.subplots(figsize=(6, 2))
    y_positions = np.arange(len(metrics))
    bars = ax.barh(
        y_positions, scaled_values, color=colors[: len(metrics)], align="center"
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(metrics)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([])

    for bar, original_value in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{original_value:.4f}",
            va="center",
            fontsize=10,
        )

    ax.set_title(title, pad=20, fontsize=14)

    plt.tight_layout()
    plt.show()
