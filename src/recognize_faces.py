import cv2 as cv
import logging
import os
from .utils import parse_json
from cv2.typing import MatLike


def draw_prediction(frame: MatLike, coordinates: tuple, name: str) -> None:
    x, y, w, h = coordinates
    cv.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
    cv.putText(frame, name, (x-10, y-10),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    logging.info(f"{name} is being recognized...")


def recognize(camera: str | int, detector_type: str) -> None:
    cap = cv.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open the camera source: {camera}")

    # intialization
    haar = cv.CascadeClassifier(
        cv.data.haarcascades+"haarcascade_frontalface_default.xml")  # type:ignore
    recognizer = cv.face.LBPHFaceRecognizer_create()  # type:ignore
    recognizer.read(os.path.join(os.getcwd(), "model", "model.yml"))
    label_map = parse_json(os.path.join(os.getcwd(), "model", "labels.json"))

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


if __name__ == "__main__":
    camera1 = "http://10.115.233.175:4747/video"
    camera2 = 0
    detector_type = "haar"
    recognize(camera1, detector_type)
