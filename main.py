import argparse
from src.utils import cameraType

def main() -> None:
    parser = argparse.ArgumentParser("Face detection and recognition")
    parser.add_argument("mode", choices=["capture", "train", "detect"],
                        help="Select operation: 'Capture' to collect face images, 'train' to build the recoginition modelm, 'detect' to run real time face recognition")
    parser.add_argument("-c", "--camera", default=0, type=cameraType,
                        help="Camera source: pass an integer(0,1,2) for local webcam, or an HTTP/RTSP URL for IP camera/stream")
    args = parser.parse_args()
    print(f"Mode is: {args.mode}, Camera is : {args.camera}, type: {type(args.camera)}")


if __name__ == '__main__':
    main()
