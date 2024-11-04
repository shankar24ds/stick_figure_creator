import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from PIL import Image
import pillow_heif
import io

detector = dlib.get_frontal_face_detector()

TARGET_FACE_WIDTH = 800
TARGET_FACE_HEIGHT = 800


def convert_to_jpeg(input_file_path):
    """Convert any image format to JPEG in memory, using pillow_heif for HEIF files, and return the image as a numpy array."""
    if input_file_path.lower().endswith((".heif", ".heic")):
        try:
            heif_image = pillow_heif.open_heif(input_file_path)
            image = Image.frombytes(heif_image.mode, heif_image.size, heif_image.data)
            image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Error converting HEIF to JPEG: {e}")
    else:
        try:
            with Image.open(input_file_path) as img:
                image = img.convert("RGB")
        except IOError as e:
            raise ValueError(
                f"Error converting image to JPEG: Unsupported format or cannot read the file. {e}"
            )
        except Exception as e:
            raise ValueError(
                f"An unexpected error occurred while converting the image: {e}"
            )

    image_np = np.array(image)
    return image_np


def detect_face(image_np):
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = detector(gray_image)
    if len(faces) == 0:
        raise ValueError("Error: No faces detected in the image.")
    elif len(faces) > 1:
        raise ValueError(
            f"Error: Multiple faces detected ({len(faces)} found). Only one face is expected."
        )

    x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
    face = image_np[y : y + h, x : x + w]
    face_resized = cv2.resize(face, (TARGET_FACE_WIDTH, TARGET_FACE_HEIGHT))
    return face_resized


def get_stick_figure_image_path(stick_figures_folder, manual_image_path=None):
    if manual_image_path:
        if not os.path.isfile(manual_image_path):
            raise FileNotFoundError(
                f"Error: Specified image '{manual_image_path}' not found."
            )
        return manual_image_path

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


def combine_face_with_stick_figure(face_image, stick_figure_image, face_scale=0.3):
    # Resize face image based on the provided scale
    face_height = int(face_image.shape[0] * face_scale)
    face_width = int(face_image.shape[1] * face_scale)
    face_image_resized = cv2.resize(face_image, (face_width, face_height))

    # Set the canvas size to just fit the face and stick figure images
    canvas_height = face_image_resized.shape[0] + stick_figure_image.shape[0]
    canvas_width = max(face_image_resized.shape[1], stick_figure_image.shape[1])
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Center-align the face and stick figure on the canvas, with no extra space
    x_offset_face = (canvas_width - face_image_resized.shape[1]) // 2
    x_offset_stick = (canvas_width - stick_figure_image.shape[1]) // 2

    # Position the face image at the top and the stick figure directly below it
    canvas[
        0 : face_image_resized.shape[0],
        x_offset_face : x_offset_face + face_image_resized.shape[1],
    ] = face_image_resized
    canvas[
        face_image_resized.shape[0] :,
        x_offset_stick : x_offset_stick + stick_figure_image.shape[1],
    ] = stick_figure_image

    return canvas


if __name__ == "__main__":
    stick_figures_folder = "images/stick_figures"
    user_image_path = input("Please enter the path to your image: ")

    try:
        image_np = convert_to_jpeg(user_image_path)

        face_image = detect_face(image_np)

        stick_figure_image_path = get_stick_figure_image_path(
            stick_figures_folder,
            "/Users/shankar.selvaraj/Documents/personal/projects/stick_figure_creator/images/stick_figures/weirdize_1.png",
        )
        stick_figure_image = cv2.cvtColor(
            cv2.imread(stick_figure_image_path), cv2.COLOR_BGR2RGB
        )

        final_image = combine_face_with_stick_figure(face_image, stick_figure_image)

        plt.figure(figsize=(5, 8))
        plt.imshow(final_image)
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(e)
