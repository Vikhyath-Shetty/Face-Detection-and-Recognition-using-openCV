import os
import cv2 as cv
from .utils import save_as_json
import numpy as np
import logging


def label_encoding() -> tuple:
    current_id = 0
    images, labels = [], []
    label_map = {}
    dataset_dir = os.path.join(os.getcwd(), "dataset")
    if not os.path.exists(dataset_dir):
        raise RuntimeError(
            "Dataset directory doesn't exist. Run capture first to get images!")
    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)
        label_map[current_id] = person
        logging.info(f"Training the recognizer for {person}...")
        for file in os.listdir(person_path):
            if file.endswith('jpg'):
                img_path = os.path.join(person_path, file)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                images.append(img)
                labels.append(current_id)
        current_id += 1
        logging.info(f"Completed training for {person}!")

    save_as_json("model", "labels.json", label_map)
    return images, labels


def train() -> None:
    images, labels = label_encoding()
    recognizer = cv.face.LBPHFaceRecognizer_create()  # type:ignore
    recognizer.train(images, np.array(labels))
    recognizer.save(os.path.join(os.getcwd(), "model", "model.yml"))
    logging.info("Completed training. Proceed with detection!")


if __name__ == "__main__":
    train()
