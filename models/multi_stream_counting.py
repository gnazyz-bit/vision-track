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
# Load YOLO model once
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# Stream configuration
# --------------------------------------------------
# Each stream has its own:
# - video capture
# - tracker
# - counting line
# - line zone
# --------------------------------------------------

streams = {}


for name, path in VIDEO_PATHS.items():

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {path}"
        )

    frame_width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    print(
        f"{name} video size: "
        f"{frame_width} x {frame_height}"
    )

    # --------------------------------------------------
    # Crosswalk configuration
    # --------------------------------------------------

    if name == "Crosswalk":

        line_y = int(frame_height * 0.51)

        line_start = sv.Point(
            x=0,
            y=line_y
        )

        line_end = sv.Point(
            x=int(frame_width * 0.58),
            y=line_y
        )

    # --------------------------------------------------
    # Mall configuration
    # --------------------------------------------------

    elif name == "Mall":

        line_y = int(frame_height * 0.93)

        line_start = sv.Point(
            x=int(frame_width * 0.03),
            y=line_y
        )

        line_end = sv.Point(
            x=int(frame_width * 0.91),
            y=line_y
        )

    else:
        raise ValueError(
            f"No counting configuration for stream: {name}"
        )

    # --------------------------------------------------
    # Create separate tracker and line zone
    # --------------------------------------------------

    tracker = sv.ByteTrack()

    line_zone = sv.LineZone(
        start=line_start,
        end=line_end,
        triggering_anchors=[
            sv.Position.BOTTOM_CENTER
        ]
    )

    streams[name] = {
        "cap": cap,
        "tracker": tracker,
        "line_zone": line_zone,
        "line_start": line_start,
        "line_end": line_end,
    }


# --------------------------------------------------
# Process all streams
# --------------------------------------------------

while True:

    all_finished = True

    for name, stream in streams.items():

        cap = stream["cap"]
        tracker = stream["tracker"]
        line_zone = stream["line_zone"]
        line_start = stream["line_start"]
        line_end = stream["line_end"]

        # --------------------------------------------------
        # Read frame
        # --------------------------------------------------

        success, frame = cap.read()

        if not success:
            continue

        all_finished = False


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
        # Convert YOLO results to supervision format
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

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Tracking label
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
            (line_start.x, line_start.y),
            (line_end.x, line_end.y),
            (0, 0, 255),
            3
        )


        # --------------------------------------------------
        # Show stream name
        # --------------------------------------------------

        cv2.putText(
            frame,
            name,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


        # --------------------------------------------------
        # Show counts
        # --------------------------------------------------

        cv2.putText(
            frame,
            f"IN: {line_zone.in_count}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"OUT: {line_zone.out_count}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )


        # --------------------------------------------------
        # Resize only for display
        # --------------------------------------------------

        display_frame = cv2.resize(
            frame,
            (700, 400)
        )


        # --------------------------------------------------
        # Show stream
        # --------------------------------------------------

        cv2.imshow(
            f"VisionTrack - {name}",
            display_frame
        )


    # --------------------------------------------------
    # Stop when all videos finish
    # --------------------------------------------------

    if all_finished:
        break


    # --------------------------------------------------
    # Q = stop all streams
    # --------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# --------------------------------------------------
# Cleanup
# --------------------------------------------------

for stream in streams.values():
    stream["cap"].release()

cv2.destroyAllWindows()


# --------------------------------------------------
# Final results
# --------------------------------------------------

print("\nMulti-stream processing finished.")

for name, stream in streams.items():

    line_zone = stream["line_zone"]

    print(
        f"{name}: "
        f"IN = {line_zone.in_count}, "
        f"OUT = {line_zone.out_count}"
    )