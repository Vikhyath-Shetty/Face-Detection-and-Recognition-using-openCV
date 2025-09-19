from argparse import ArgumentTypeError
import json
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
    dataset_path = os.path.join(os.getcwd(), "dataset", dir_name)
    model_path = os.path.join(os.path.join(os.getcwd(),"model"))
    os.path.exists(dataset_path) and shutil.rmtree(dataset_path)
    os.path.exists(model_path) and shutil.rmtree(model_path)



def save_as_json(location: str, data: dict):
    with open(location, "w") as f:
        json.dump(data, f)
