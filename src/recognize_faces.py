import cv2 as cv
import logging
import os
from .utils import parse_json
from cv2.typing import MatLike


def draw_prediction(frame: MatLike, coordinates: tuple, name: str) -> None:
    x, y, w, h = coordinates
    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2, cv.LINE_AA)
    (text_w, text_h), _ = cv.getTextSize(name.upper(), cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv.rectangle(frame, (x, y - text_h - 10), (x + text_w+2, y), (0, 255, 255), -1)
    cv.putText(frame, name.upper(), (x+2, y - 5), cv.FONT_HERSHEY_SIMPLEX, 
               0.6, (128, 0, 128), 2, cv.LINE_AA)
    logging.info(f"{name} is being recognized...")


def recognize(camera: str | int) -> None:
    model_dir = os.path.join(os.getcwd(), "model")
    if not os.path.exists(model_dir):
        raise RuntimeError(
            "Model directory doesn't exist. Run 'train' to create recognition model!")
    model_path = os.path.join(model_dir, "model.yml")
    label_path = os.path.join(model_dir, "labels.json")

    cap = cv.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open the camera source: {camera}")

    # intialization
    haar = cv.CascadeClassifier(
        cv.data.haarcascades+"haarcascade_frontalface_default.xml")  # type:ignore
    recognizer = cv.face.LBPHFaceRecognizer_create()  # type:ignore
    recognizer.read(model_path)
    label_map = parse_json(label_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame...")
            continue
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(100, 100))
        for x, y, w, h in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv.resize(face, (200, 200))
            label, confidence = recognizer.predict(face_resized)
            if confidence < 80:
                draw_prediction(frame, (x, y, w, h), label_map[str(label)])
            else:
                draw_prediction(frame, (x, y, w, h), "Unknown")
        cv.imshow("Face Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
