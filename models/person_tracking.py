from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATH = PROJECT_ROOT / "data" / "raw_videos" / "mall.mp4"


# --------------------------------------------------
# Settings
# --------------------------------------------------

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5


# --------------------------------------------------
# Load YOLO
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# Create tracker
# --------------------------------------------------

tracker = sv.ByteTrack()


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

    # --------------------------------------------------
    # Person detection with YOLO
    # --------------------------------------------------

    results = model(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------
    # Convert YOLO detections to Supervision format
    # --------------------------------------------------

    detections = sv.Detections.from_ultralytics(result)


    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # --------------------------------------------------
    # Draw tracked people
    # --------------------------------------------------

    for xyxy, confidence, tracker_id in zip(
        detections.xyxy,
        detections.confidence,
        detections.tracker_id
    ):

        x1, y1, x2, y2 = map(int, xyxy)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Create label
        label = (
            f"ID {tracker_id} "
            f"{confidence:.2f}"
        )

        # Draw label
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # --------------------------------------------------
    # Show video
    # --------------------------------------------------

    cv2.imshow(
        "VisionTrack - Person Tracking",
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