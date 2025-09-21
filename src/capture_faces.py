import cv2 as cv
import logging
from .utils import crop_and_save


def capture(camera: str | int, detector_type: str) -> None:
    cap = cv.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {camera}")

    missed_frame, image_count = 0, 0
    person_name = input("Enter the name of the person to be captured: ")
    if detector_type == 'haar':
        haar = cv.CascadeClassifier(
            cv.data.haarcascades+"haarcascade_frontalface_default.xml")  # type:ignore
    else:
        hog = cv.HOGDescriptor()
        hog.setSVMDetector(
            cv.HOGDescriptor_getDefaultPeopleDetector())  # type:ignore

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Failed to grab frame...")
            missed_frame += 1
            if missed_frame > 10:
                break
            continue

        missed_frame = 0
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        logging.info("Capturing images for training!")
        if detector_type == 'haar':
            faces = haar.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5, minSize=(10, 10))
        else:
            faces, _ = hog.detectMultiScale(gray, winStride=(
                8, 8), padding=(8, 8), scale=1.05, groupThreshold=2)
        for x, y, w, h in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            image_count += 1
            if (image_count <= 20):
                crop_and_save(gray, (x, y, w, h), person_name, image_count)

        cv.imshow("Capture Faces - Press 'q' to Quit", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    camera = 0
    detector_type = "haar"
    capture(camera, detector_type)
