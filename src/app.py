import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from PIL import Image
import pillow_heif

detector = dlib.get_frontal_face_detector()


def convert_to_jpeg(input_file_path):
    """Convert any image format to JPEG, using pillow_heif for HEIF files, and return the path of the JPEG image."""
    output_file_path = os.path.splitext(input_file_path)[0] + ".jpg"

    if input_file_path.lower().endswith((".heif", ".heic")):
        try:
            heif_image = pillow_heif.open_heif(input_file_path)
            image = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
            image = image.convert("RGB")
            image.save(output_file_path, "JPEG")
            return output_file_path
        except Exception as e:
            raise ValueError(f"Error converting HEIF to JPEG: {e}")
    else:
        try:
            with Image.open(input_file_path) as img:
                img = img.convert("RGB")
                img.save(output_file_path, "JPEG")
            return output_file_path
        except IOError as e:
            raise ValueError(
                f"Error converting image to JPEG: Unsupported format or cannot read the file. {e}"
            )
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while converting the image: {e}"
            )


def detect_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Error: Image not found or could not be loaded.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = detector(gray_image)
    if len(faces) == 0:
        raise ValueError("Error: No faces detected in the image.")
    elif len(faces) > 1:
        raise ValueError(
            f"Error: Multiple faces detected ({len(faces)} found). Only one face is expected."
        )

    x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
    face = image_rgb[y : y + h, x : x + w]
    return face


def get_random_stick_figure_image_path(stick_figures_folder):
    all_images = [
        f
        for f in os.listdir(stick_figures_folder)
        if f.endswith((".jpeg", ".jpg", ".png"))
    ]

    if not all_images:
        raise ValueError("Error: No stick figure images found in the folder.")

    random_image_name = random.choice(all_images)
    random_image_path = os.path.join(stick_figures_folder, random_image_name)

    return random_image_path


def combine_face_with_stick_figure(face_image, stick_figure_image):
    stick_figure_image_resized = cv2.resize(
        stick_figure_image, (face_image.shape[1], stick_figure_image.shape[0])
    )

    combined_image = np.vstack((face_image, stick_figure_image_resized))

    canvas_height = combined_image.shape[0] + 50
    canvas_width = combined_image.shape[1] + 50
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    y_offset = 25
    x_offset = (canvas.shape[1] - combined_image.shape[1]) // 2
    canvas[
        y_offset : y_offset + combined_image.shape[0],
        x_offset : x_offset + combined_image.shape[1],
    ] = combined_image

    return canvas


if __name__ == "__main__":
    stick_figures_folder = "images/stick_figures"

    user_image_path = input("Please enter the path to your image: ")

    try:
        jpeg_image_path = convert_to_jpeg(user_image_path)

        face_image = detect_face(jpeg_image_path)

        stick_figure_image_path = get_random_stick_figure_image_path(
            stick_figures_folder
        )
        stick_figure_image = cv2.imread(stick_figure_image_path)

        final_image = combine_face_with_stick_figure(face_image, stick_figure_image)

        plt.figure(figsize=(3, 5))
        plt.imshow(final_image)
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(e)
