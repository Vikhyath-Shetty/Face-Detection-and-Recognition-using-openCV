from argparse import ArgumentTypeError
import os
import cv2 as cv
from cv2.typing import MatLike
import shutil


def cameraType(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        pass
    if value.startswith(("http://", "https://", "rtsp://")):
        return value
    raise ArgumentTypeError("--camera must be an integer or valid URL")


def create_dir(dir_name: str) -> str:
    try:
        path = os.path.join(os.getcwd(), "dataset", dir_name)
    except FileNotFoundError:
        print("Failed to create directory...")
    os.makedirs(path, exist_ok=True)
    return path


def crop_and_save(image: MatLike, points: tuple, dir_name: str, image_id: int) -> None:
    x, y, w, h = points
    face = image[y:y+h, x:x+w]
    face_resized = cv.resize(face, (200, 200))
    directory = create_dir(dir_name)
    cv.imwrite(os.path.join(directory, f"{image_id}.jpg"), face_resized)


def clear_data(dir_name: str) -> None:
    path = os.path.join(os.getcwd(), "dataset", dir_name)
    if os.path.exists(path):
        shutil.rmtree(path)
    else:
        print("Provided directory doesn't exist!")
