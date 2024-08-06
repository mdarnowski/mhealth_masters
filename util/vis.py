from IPython.display import Image
from keras.src.utils import model_to_dot


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
