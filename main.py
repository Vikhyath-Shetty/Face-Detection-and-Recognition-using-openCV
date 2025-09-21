import argparse
import logging
from src import cameraType,capture,train,recognize


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
    args = parser.parse_args()

    try:
        if args.mode == 'capture':
            capture(args.camera)
        elif args.mode == 'train':
            train()
        else:
            recognize(args.camera)
    except RuntimeError as e:
        logging.error(f"Encountered runtime error:{e}")


if __name__ == '__main__':
    main()
