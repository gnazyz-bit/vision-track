from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VIDEO_PATHS = {
    "Crosswalk": PROJECT_ROOT / "data" / "raw_videos" / "crosswalk.mp4",
    "Mall": PROJECT_ROOT / "data" / "raw_videos" / "mall.mp4",
}


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
# Create video captures and trackers
# --------------------------------------------------

streams = {}

for name, path in VIDEO_PATHS.items():

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {path}"
        )

    streams[name] = {
        "cap": cap,
        "tracker": sv.ByteTrack(),
    }


# --------------------------------------------------
# Process videos
# --------------------------------------------------

while True:

    all_finished = True

    for name, stream in streams.items():

        cap = stream["cap"]
        tracker = stream["tracker"]

        success, frame = cap.read()

        if not success:
            continue

        all_finished = False

        # ------------------------------------------
        # Detection
        # ------------------------------------------

        results = model(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        result = results[0]

        # ------------------------------------------
        # Convert detections
        # ------------------------------------------

        detections = sv.Detections.from_ultralytics(
            result
        )

        # ------------------------------------------
        # Tracking
        # ------------------------------------------

        detections = tracker.update_with_detections(
            detections
        )

        # ------------------------------------------
        # Draw tracked people
        # ------------------------------------------

        for xyxy, confidence, tracker_id in zip(
            detections.xyxy,
            detections.confidence,
            detections.tracker_id
        ):

            x1, y1, x2, y2 = map(int, xyxy)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = f"ID {tracker_id} {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ------------------------------------------
        # Show video
        # ------------------------------------------

        cv2.imshow(
            f"VisionTrack - {name}",
            frame
        )

    # Stop when both videos finish
    if all_finished:
        break

    # Q = stop all streams
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

for stream in streams.values():
    stream["cap"].release()

cv2.destroyAllWindows()