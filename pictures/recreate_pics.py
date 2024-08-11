import base64
import os
from io import BytesIO

from nbformat import read
from PIL import Image

root_dir = "../"
output_dir = "../pictures"


def extract_images_from_ipynb(ipynb_path):
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = read(f, as_version=4)

    name_of_the_file = os.path.splitext(os.path.basename(ipynb_path))[0]
    image_nr = 1

    for cell in nb.cells:
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if output.output_type == "display_data" and "image/png" in output.data:
                    img_data = output.data["image/png"]
                    img = Image.open(BytesIO(base64.b64decode(img_data)))
                    img_filename = f"{name_of_the_file}-{image_nr}.png"
                    img.save(os.path.join(output_dir, img_filename))
                    image_nr += 1


def process_directory():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ipynb"):
                ipynb_path = os.path.join(subdir, file)
                extract_images_from_ipynb(ipynb_path)


if __name__ == "__main__":
    process_directory()
    print("Image extraction completed.")
