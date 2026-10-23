"""Opening camera/video sources and saving user uploads."""
from pathlib import Path
import cv2


def open_video(source):
    """Return an OpenCV capture; caller validates isOpened and releases it."""
    return cv2.VideoCapture(str(source))


def save_uploaded_video(uploaded, directory, camera_name):
    """Save uploaded Streamlit video in the temporary directory."""
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi"}:
        raise ValueError("Unsupported video format")
    source = Path(directory) / f"{camera_name}{suffix}"
    source.write_bytes(uploaded.getvalue())
    return source