import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
from PIL import Image
import pillow_heif

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


def load_image(image_path):
    """Load an image from a given path and convert it to RGB format."""
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None


def attach_hair_and_ears(
    face_image, hair_image_path, left_ear_image_path, right_ear_image_path
):
    """Attach hair and ears to the face image without modifying the face image size."""
    hair_image = load_image(hair_image_path)
    left_ear_image = load_image(left_ear_image_path)
    right_ear_image = load_image(right_ear_image_path)

    combined_height = face_image.shape[0] + hair_image.shape[0]
    combined_width = (
        face_image.shape[1] + left_ear_image.shape[1] + right_ear_image.shape[1]
    )

    combined_image = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255

    hair_x_offset = left_ear_image.shape[1]

    combined_image[
        : hair_image.shape[0], hair_x_offset : hair_x_offset + hair_image.shape[1]
    ] = hair_image

    combined_image[
        hair_image.shape[0] :, hair_x_offset : hair_x_offset + face_image.shape[1]
    ] = face_image

    if left_ear_image is not None:
        ear_height = left_ear_image.shape[0]
        ear_width = left_ear_image.shape[1]
        combined_image[
            hair_image.shape[0] : ear_height + hair_image.shape[0], :ear_width
        ] = left_ear_image

    if right_ear_image is not None:
        ear_height = right_ear_image.shape[0]
        ear_width = right_ear_image.shape[1]
        combined_image[
            hair_image.shape[0] : ear_height + hair_image.shape[0], -ear_width:
        ] = right_ear_image

    return combined_image


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
    face_height = int(face_image.shape[0] * face_scale)
    face_width = int(face_image.shape[1] * face_scale)
    face_image_resized = cv2.resize(face_image, (face_width, face_height))

    canvas_height = face_image_resized.shape[0] + stick_figure_image.shape[0]
    canvas_width = max(face_image_resized.shape[1], stick_figure_image.shape[1])
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    x_offset_face = (canvas_width - face_image_resized.shape[1]) // 2
    x_offset_stick = (canvas_width - stick_figure_image.shape[1]) // 2

    canvas[
        0 : face_image_resized.shape[0],
        x_offset_face : x_offset_face + face_image_resized.shape[1],
    ] = face_image_resized
    canvas[
        face_image_resized.shape[0] :,
        x_offset_stick : x_offset_stick + stick_figure_image.shape[1],
    ] = stick_figure_image

    return canvas


def shape_face_with_double_triangle_cut(face_image):
    """Trim the lower corners of the face image to create two inverted triangle shapes as specified."""
    height, width = face_image.shape[:2]

    trimmed_face_image = np.ones((height, width, 3), dtype=np.uint8) * 255

    left_triangle = np.array(
        [
            [0, height],
            [0, height - 200],
            [200, height],
        ]
    )

    right_triangle = np.array(
        [
            [width, height],
            [width, height - 200],
            [width - 200, height],
        ]
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [left_triangle], 255)
    cv2.fillPoly(mask, [right_triangle], 255)

    trimmed_face_image[mask == 0] = face_image[mask == 0]

    return trimmed_face_image


if __name__ == "__main__":
    stick_figures_folder = "images/stick_figures"
    user_image_path = input("Please enter the path to your image: ")

    hair_image_path = "images/stick_figures/hair.png"
    left_ear_image_path = "images/stick_figures/left_ear.png"
    right_ear_image_path = "images/stick_figures/right_ear.png"

    try:
        image_np = convert_to_jpeg(user_image_path)

        face_image = detect_face(image_np)

        trimmed_face_image = shape_face_with_double_triangle_cut(face_image)

        face_with_hair_and_ears = attach_hair_and_ears(
            trimmed_face_image,
            hair_image_path,
            left_ear_image_path,
            right_ear_image_path,
        )

        stick_figure_image_path = get_stick_figure_image_path(
            stick_figures_folder,
            "/Users/shankar.selvaraj/Documents/personal/projects/stick_figure_creator/images/stick_figures/traumatize_1.png",
        )
        stick_figure_image = cv2.cvtColor(
            cv2.imread(stick_figure_image_path), cv2.COLOR_BGR2RGB
        )

        final_image = combine_face_with_stick_figure(
            face_with_hair_and_ears, stick_figure_image
        )

        plt.figure(figsize=(5, 8))
        plt.imshow(final_image)
        plt.axis("off")
        plt.show()

    except Exception as e:
        print(e)
