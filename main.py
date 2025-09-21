import argparse
import logging
from src.utils import cameraType
from src.capture_faces import capture
from src.train_faces import train
from src.recognize_faces import recognize


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def main() -> None:
    parser = argparse.ArgumentParser("Face detection and recognition")
    parser.add_argument("mode", choices=["capture", "train", "detect"],
                        help="Select operation: 'Capture' to collect face images, 'train' to build the recoginition modelm, 'detect' to run real time face recognition")
    parser.add_argument("-c", "--camera", default=0, type=cameraType,
                        help="Camera source: pass an integer(0,1,2) for local webcam, or an HTTP/RTSP URL for IP camera/stream,(default=0)")
    parser.add_argument("-d", "--detector", default="haar", type=str, choices=["haar"],
                        help="Face Detector: 'haar' to use Haar detector, 'hog' to use HOG+SVM detector,(default=haar)")
    args = parser.parse_args()

    try:
        if args.mode == 'capture':
            capture(args.camera, args.detector)
        elif args.mode == 'train':
            train()
        else:
            recognize(args.camera, args.detector)
    except RuntimeError as e:
        logging.error(f"Encountered runtime error:{e}")


if __name__ == '__main__':
    main()
