import os
import cv2 as cv
import json

label_map = {}
images = []
labels = []


def label_encoding() -> None:
    current_id = 0
    dataset_dir = os.path.join(os.getcwd(), "dataset")
    if not os.listdir(dataset_dir):
        raise RuntimeError(
            "Dataset directory is empty! Run capture mode first.")
    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)
        label_map[current_id] = person
        for file in os.listdir(person_path):
            if file.endswith('jpg'):
                img_path = os.path.join(person_path, file)
                img = cv.imread(img_path)
                images.append(img)
                labels.append(current_id)
        current_id += 1
        
    


if __name__ == "__main__":
    label_encoding()
