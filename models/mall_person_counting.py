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
# Get video size
# --------------------------------------------------

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print(
    f"Video size: "
    f"{frame_width} x {frame_height}"
)


# --------------------------------------------------
# Create counting line
# --------------------------------------------------

# The line represents the main mall entrance.
# People crossing the line in one direction are counted
# as entering, and in the opposite direction as exiting.

line_y = int(frame_height * 0.93)

LINE_START = sv.Point(
    x=int(frame_width * 0.03),
    y=line_y
)

LINE_END = sv.Point(
    x=int(frame_width * 0.91),
    y=line_y
)

line_zone = sv.LineZone(
    start=LINE_START,
    end=LINE_END,
    triggering_anchors=[
        sv.Position.BOTTOM_CENTER
    ]
)


# --------------------------------------------------
# Process video
# --------------------------------------------------

while True:

    success, frame = cap.read()

    if not success:
        break


    # --------------------------------------------------
    # Person detection
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

    detections = sv.Detections.from_ultralytics(
        result
    )


    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------

    detections = tracker.update_with_detections(
        detections
    )


    # --------------------------------------------------
    # Counting
    # --------------------------------------------------

    line_zone.trigger(
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

        # Label with tracker ID
        label = (
            f"ID {tracker_id} "
            f"{confidence:.2f}"
        )

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
    # Draw counting line
    # --------------------------------------------------

    cv2.line(
        frame,
        (LINE_START.x, LINE_START.y),
        (LINE_END.x, LINE_END.y),
        (0, 0, 255),
        3
    )


    # --------------------------------------------------
    # Show IN / OUT counts
    # --------------------------------------------------

    cv2.putText(
        frame,
        f"IN: {line_zone.in_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"OUT: {line_zone.out_count}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )


    # --------------------------------------------------
    # Show video
    # --------------------------------------------------

    cv2.imshow(
        "VisionTrack - Mall Counting",
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


# --------------------------------------------------
# Final result
# --------------------------------------------------

print("\nMall video processing finished.")
print(f"IN: {line_zone.in_count}")
print(f"OUT: {line_zone.out_count}")