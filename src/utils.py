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

 # DEV FUNCTION: Cleans up the 'model/' and 'dataset/' directories.


def clear_data() -> None:
    dataset_path = os.path.join(os.getcwd(), "dataset")
    model_path = os.path.join(os.getcwd(), "model")
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)


def save_as_json(dir_name: str, file_name: str, data: dict) -> None:
    dir_path = os.path.join(os.getcwd(), dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, file_name)
    with open(file_path, "w") as f:
        json.dump(data, f)


def parse_json(location: str) -> dict:
    with open(location, "r") as f:
        data = json.load(f)
    return data
