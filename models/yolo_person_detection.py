from pathlib import Path

import cv2
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "data" / "raw_videos" / "test.mp4"


# --------------------------------------------------
# Settings
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.5

PERSON_CLASS_ID = 0


# --------------------------------------------------
# Load YOLO
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# Open video
# --------------------------------------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))


if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}"
    )


# --------------------------------------------------
# Process video
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break

    # Run YOLO only for the person class
    results = model(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    # Get detections from current frame
    result = results[0]

    # Number of people detected in current frame
    person_count = len(result.boxes)

    # Draw every detected person
    for box in result.boxes:

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )

        confidence = float(box.conf[0])

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"Person {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # Show number of people in current frame
    cv2.putText(
        frame,
        f"People detected: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # Show frame
    cv2.imshow(
        "VisionTrack - Person Detection",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

cap.release()

cv2.destroyAllWindows()