import cv2 as cv
import logging

def capture(camera: str | int) -> None:
    cap = cv.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {camera}")
    while True:
        ret,frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame...")
            continue
        


        
