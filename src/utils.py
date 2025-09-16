from argparse import ArgumentTypeError


def cameraType(value: str) -> str | int:
    try:
        return int(value)
    except ValueError:
        pass
    if value.startswith(("http://", "https://", "rtsp://")):
        return value
    raise ArgumentTypeError("--camera must be an integer or valid URL")
