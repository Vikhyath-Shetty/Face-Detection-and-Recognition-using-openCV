import argparse
import logging
from src.utils import cameraType, create_dir
from src.capture_faces import capture


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
    parser.add_argument("-d", "--detector", default="haar", type=str,
                        help="Face Detector: 'haar' to use Haar detector, 'hog' to use HOG+SVM detector,(default=haar)")
    args = parser.parse_args()
    if args.mode == 'capture':
        capture(args.camera)
        


if __name__ == '__main__':
    main()
