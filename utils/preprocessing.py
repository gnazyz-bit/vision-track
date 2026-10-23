"""Frame resizing and RGB conversion for consistent Streamlit display."""
import cv2
import numpy as np

DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 540


def display_frame(frame):
    """Letterbox without cropping or stretching; constant output resolution."""

    height, width = frame.shape[:2]

    canvas = np.zeros(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
        dtype=np.uint8
    )

    scale = min(
        DISPLAY_WIDTH / width,
        DISPLAY_HEIGHT / height
    )

    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        frame,
        (new_w, new_h),
        interpolation=interpolation
    )

    x = (DISPLAY_WIDTH - new_w) // 2
    y = (DISPLAY_HEIGHT - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )