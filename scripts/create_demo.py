"""
VisionTrack - Professional Demo Video Generator

Creates:
    reports/demo_results/multi_stream_demo.mp4
    reports/demo_results/roi_counting_example.png

Features:
    - YOLOv8n person detection
    - ByteTrack multi-object tracking
    - LineZone IN / OUT counting
    - Two synchronized video panels
    - Clean dashboard visualization
    - Tracking IDs
    - Counting statistics
    - H.264 MP4 export

Run:
    python scripts/create_demo.py
"""

from pathlib import Path
import subprocess

import cv2
import numpy as np
import supervision as sv
import imageio_ffmpeg

from ultralytics import YOLO


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "yolov8n.pt"

VIDEO_PATHS = {
    "Crosswalk": (
        ROOT / "data" / "raw_videos" / "crosswalk.mp4"
    ),
    "Mall": (
        ROOT / "data" / "raw_videos" / "mall.mp4"
    ),
}

OUTPUT_DIR = ROOT / "reports" / "demo_results"

VIDEO_OUTPUT = OUTPUT_DIR / "multi_stream_demo.mp4"

IMAGE_OUTPUT = OUTPUT_DIR / "roi_counting_example.png"


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

CONFIDENCE = 0.35

IMAGE_SIZE = 960

PERSON_CLASS_ID = 0

MAX_DURATION_SECONDS = 30

OUTPUT_FPS = 25


# ============================================================
# 3. VIDEO LAYOUT
# ============================================================

VIDEO_WIDTH = 960
VIDEO_HEIGHT = 540

HEADER_HEIGHT = 72
FOOTER_HEIGHT = 132

PANEL_WIDTH = VIDEO_WIDTH

PANEL_HEIGHT = (
    HEADER_HEIGHT
    + VIDEO_HEIGHT
    + FOOTER_HEIGHT
)

OUTPUT_WIDTH = PANEL_WIDTH * 2
OUTPUT_HEIGHT = PANEL_HEIGHT


# ============================================================
# 4. COLORS (BGR)
# ============================================================

BACKGROUND = (30, 24, 17)

PANEL_BACKGROUND = (42, 34, 26)

CARD_BACKGROUND = (55, 45, 34)

WHITE = (255, 255, 255)

TEXT_SECONDARY = (175, 165, 155)

GREEN = (105, 210, 80)

BLUE = (245, 175, 70)

RED = (75, 80, 240)

YELLOW = (80, 215, 245)

BORDER = (80, 70, 60)


# ============================================================
# 5. DRAWING HELPERS
# ============================================================

def draw_text(
    image,
    text,
    x,
    y,
    scale=0.65,
    color=WHITE,
    thickness=2,
):

    cv2.putText(
        image,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_rectangle(
    image,
    x1,
    y1,
    x2,
    y2,
    color,
    thickness=-1,
):

    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# 6. LETTERBOX RESIZING
# ============================================================

def resize_frame(frame):

    height, width = frame.shape[:2]

    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions")

    scale = min(
        VIDEO_WIDTH / width,
        VIDEO_HEIGHT / height,
    )

    new_width = max(
        1,
        round(width * scale),
    )

    new_height = max(
        1,
        round(height * scale),
    )

    interpolation = (
        cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR
    )

    resized = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=interpolation,
    )

    canvas = np.full(
        (
            VIDEO_HEIGHT,
            VIDEO_WIDTH,
            3,
        ),
        BACKGROUND,
        dtype=np.uint8,
    )

    offset_x = (
        VIDEO_WIDTH - new_width
    ) // 2

    offset_y = (
        VIDEO_HEIGHT - new_height
    ) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized

    return (
        canvas,
        scale,
        offset_x,
        offset_y,
    )


# ============================================================
# 7. CREATE VIDEO STREAM
# ============================================================

def create_stream(name, video_path):

    if not video_path.is_file():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        cap.release()

        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if width <= 0 or height <= 0:

        cap.release()

        raise RuntimeError(
            f"Invalid video dimensions: {name}"
        )

    if not 1 <= fps <= 120:
        fps = 25.0

    # ----------------------------------------
    # Counting lines
    # ----------------------------------------

    if name == "Crosswalk":

        line_y = int(
            height * 0.51
        )

        line_start = sv.Point(
            0,
            line_y,
        )

        line_end = sv.Point(
            int(width * 0.58),
            line_y,
        )

    elif name == "Mall":

        line_y = int(
            height * 0.93
        )

        line_start = sv.Point(
            int(width * 0.03),
            line_y,
        )

        line_end = sv.Point(
            int(width * 0.91),
            line_y,
        )

    else:

        cap.release()

        raise ValueError(
            f"Unknown camera: {name}"
        )

    # ----------------------------------------
    # ByteTrack
    # ----------------------------------------

    tracker = sv.ByteTrack(
        frame_rate=int(round(fps)),
    )

    # ----------------------------------------
    # LineZone
    # ----------------------------------------

    zone = sv.LineZone(
        start=line_start,
        end=line_end,
        triggering_anchors=[
            sv.Position.BOTTOM_CENTER
        ],
    )

    return {
        "name": name,
        "cap": cap,
        "fps": fps,
        "width": width,
        "height": height,
        "tracker": tracker,
        "zone": zone,
        "line_start": line_start,
        "line_end": line_end,
        "frame_index": -1,
        "last_panel": None,
        "finished": False,
        "people": 0,
        "detections": 0,
        "frame": None,
    }


# ============================================================
# 8. DRAW TRACKED PERSONS
# ============================================================

def draw_detections(
    frame,
    detections,
    scale,
    offset_x,
    offset_y,
):

    if detections is None:
        return frame

    for index, box in enumerate(
        detections.xyxy
    ):

        x1, y1, x2, y2 = map(
            float,
            box,
        )

        # Convert original coordinates
        # to the displayed video coordinates.

        x1 = int(x1 * scale + offset_x)
        y1 = int(y1 * scale + offset_y)

        x2 = int(x2 * scale + offset_x)
        y2 = int(y2 * scale + offset_y)

        x1 = max(
            0,
            min(x1, VIDEO_WIDTH - 1),
        )

        x2 = max(
            0,
            min(x2, VIDEO_WIDTH - 1),
        )

        y1 = max(
            0,
            min(y1, VIDEO_HEIGHT - 1),
        )

        y2 = max(
            0,
            min(y2, VIDEO_HEIGHT - 1),
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # ------------------------------------
        # Bounding box
        # ------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            GREEN,
            2,
            cv2.LINE_AA,
        )

        # ------------------------------------
        # Tracking ID
        # ------------------------------------

        tracker_id = None

        if detections.tracker_id is not None:

            tracker_id = int(
                detections.tracker_id[index]
            )

        if tracker_id is not None:

            label = f"ID {tracker_id}"

        else:

            label = "Person"

        # ------------------------------------
        # Label background
        # ------------------------------------

        font_scale = 0.48
        font_thickness = 1

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )

        text_width = text_size[0]
        text_height = text_size[1]

        label_width = text_width + 12
        label_height = text_height + 10

        label_x = min(
            x1,
            VIDEO_WIDTH - label_width,
        )

        # Put label above the box when possible,
        # otherwise inside its upper edge.

        if y1 >= label_height:

            label_top = y1 - label_height

        else:

            label_top = y1

        label_bottom = min(
            VIDEO_HEIGHT,
            label_top + label_height,
        )

        draw_rectangle(
            frame,
            label_x,
            label_top,
            label_x + label_width,
            label_bottom,
            GREEN,
        )

        draw_text(
            frame,
            label,
            label_x + 6,
            label_top + text_height + 4,
            scale=font_scale,
            color=BACKGROUND,
            thickness=1,
        )

    return frame


# ============================================================
# 9. DRAW COUNTING LINE
# ============================================================

def draw_counting_line(
    frame,
    stream,
    scale,
    offset_x,
    offset_y,
):

    start = stream["line_start"]
    end = stream["line_end"]

    x1 = int(
        start.x * scale + offset_x
    )

    y1 = int(
        start.y * scale + offset_y
    )

    x2 = int(
        end.x * scale + offset_x
    )

    y2 = int(
        end.y * scale + offset_y
    )

    # Dark outline for readability.

    cv2.line(
        frame,
        (x1, y1),
        (x2, y2),
        BACKGROUND,
        6,
        cv2.LINE_AA,
    )

    # Counting line.

    cv2.line(
        frame,
        (x1, y1),
        (x2, y2),
        RED,
        3,
        cv2.LINE_AA,
    )

    # Endpoints.

    cv2.circle(
        frame,
        (x1, y1),
        5,
        RED,
        -1,
        cv2.LINE_AA,
    )

    cv2.circle(
        frame,
        (x2, y2),
        5,
        RED,
        -1,
        cv2.LINE_AA,
    )

    return frame


# ============================================================
# 10. PROCESS FRAME WITH YOLO AND BYTETRACK
# ============================================================

def process_frame(
    frame,
    stream,
    model,
):

    # ----------------------------------------
    # YOLO detection
    # ----------------------------------------

    results = model.predict(
        frame,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    detections = (
        sv.Detections.from_ultralytics(
            results[0]
        )
    )

    # ----------------------------------------
    # ByteTrack
    # ----------------------------------------

    tracked = (
        stream["tracker"].update_with_detections(
            detections
        )
    )

    # ----------------------------------------
    # LineZone counting
    # ----------------------------------------

    stream["zone"].trigger(
        tracked
    )

    stream["people"] = len(tracked)

    stream["detections"] = len(detections)

    # ----------------------------------------
    # Resize original frame
    # ----------------------------------------

    (
        display,
        scale,
        offset_x,
        offset_y,
    ) = resize_frame(frame)

    # ----------------------------------------
    # Draw boxes AFTER resizing
    # for better visual quality
    # ----------------------------------------

    display = draw_detections(
        display,
        tracked,
        scale,
        offset_x,
        offset_y,
    )

    # ----------------------------------------
    # Counting line
    # ----------------------------------------

    display = draw_counting_line(
        display,
        stream,
        scale,
        offset_x,
        offset_y,
    )

    return display


# ============================================================
# 11. METRIC CARD
# ============================================================

def draw_metric_card(
    panel,
    x,
    y,
    width,
    height,
    title,
    value,
    value_color,
):

    draw_rectangle(
        panel,
        x,
        y,
        x + width,
        y + height,
        CARD_BACKGROUND,
    )

    draw_text(
        panel,
        title,
        x + 15,
        y + 25,
        scale=0.48,
        color=TEXT_SECONDARY,
        thickness=1,
    )

    draw_text(
        panel,
        str(value),
        x + 15,
        y + 64,
        scale=1.0,
        color=value_color,
        thickness=2,
    )


# ============================================================
# 12. CREATE DASHBOARD PANEL
# ============================================================

def draw_panel(
    video_frame,
    stream,
    timestamp,
):

    panel = np.full(
        (
            PANEL_HEIGHT,
            PANEL_WIDTH,
            3,
        ),
        PANEL_BACKGROUND,
        dtype=np.uint8,
    )

    name = stream["name"]

    # ----------------------------------------
    # Header
    # ----------------------------------------

    draw_rectangle(
        panel,
        0,
        0,
        PANEL_WIDTH,
        HEADER_HEIGHT,
        BACKGROUND,
    )

    draw_text(
        panel,
        name,
        24,
        43,
        scale=0.95,
        color=WHITE,
        thickness=2,
    )

    # Camera indicator

    cv2.circle(
        panel,
        (PANEL_WIDTH - 162, 35),
        6,
        GREEN if not stream["finished"] else TEXT_SECONDARY,
        -1,
        cv2.LINE_AA,
    )

    status = (
        "TRACKING"
        if not stream["finished"]
        else "FINISHED"
    )

    draw_text(
        panel,
        status,
        PANEL_WIDTH - 146,
        41,
        scale=0.48,
        color=TEXT_SECONDARY,
        thickness=1,
    )

    # ----------------------------------------
    # Video
    # ----------------------------------------

    panel[
        HEADER_HEIGHT:HEADER_HEIGHT + VIDEO_HEIGHT,
        0:VIDEO_WIDTH,
    ] = video_frame

    # ----------------------------------------
    # Footer
    # ----------------------------------------

    footer_y = HEADER_HEIGHT + VIDEO_HEIGHT

    draw_rectangle(
        panel,
        0,
        footer_y,
        PANEL_WIDTH,
        PANEL_HEIGHT,
        BACKGROUND,
    )

    in_count = int(
        stream["zone"].in_count
    )

    out_count = int(
        stream["zone"].out_count
    )

    people = int(
        stream["people"]
    )

    # ----------------------------------------
    # Metric cards
    # ----------------------------------------

    margin = 18
    gap = 12

    card_width = (
        PANEL_WIDTH - 2 * margin - 2 * gap
    ) // 3

    card_height = 82

    card_y = footer_y + 12

    draw_metric_card(
        panel,
        margin,
        card_y,
        card_width,
        card_height,
        "IN",
        in_count,
        GREEN,
    )

    draw_metric_card(
        panel,
        margin + card_width + gap,
        card_y,
        card_width,
        card_height,
        "OUT",
        out_count,
        BLUE,
    )

    draw_metric_card(
        panel,
        margin + (card_width + gap) * 2,
        card_y,
        card_width,
        card_height,
        "TRACKED PEOPLE",
        people,
        WHITE,
    )

    # ----------------------------------------
    # Time label
    # ----------------------------------------

    minutes = int(timestamp // 60)
    seconds = int(timestamp % 60)

    time_label = (
        f"{minutes:02d}:{seconds:02d}"
    )

    draw_text(
        panel,
        f"TIME {time_label}",
        20,
        PANEL_HEIGHT - 15,
        scale=0.45,
        color=TEXT_SECONDARY,
        thickness=1,
    )

    draw_text(
        panel,
        "YOLOv8n  |  ByteTrack  |  LineZone",
        550,
        PANEL_HEIGHT - 15,
        scale=0.42,
        color=TEXT_SECONDARY,
        thickness=1,
    )

    return panel


# ============================================================
# 13. PROCESS SOURCE VIDEO UP TO TIMESTAMP
# ============================================================

def process_until(
    stream,
    timestamp,
    model,
):

    if stream["finished"]:

        return stream["last_panel"]

    target_index = int(
        timestamp * stream["fps"] + 1e-6
    )

    # Process every source frame up to this point.
    # Important for ByteTrack continuity and LineZone counting.

    while (
        stream["frame_index"] < target_index
    ):

        success, frame = (
            stream["cap"].read()
        )

        if not success:

            stream["finished"] = True

            stream["cap"].release()

            break

        stream["frame_index"] += 1

        processed = process_frame(
            frame,
            stream,
            model,
        )

        stream["frame"] = processed

    if stream["frame"] is None:

        return stream["last_panel"]

    # Keep the same frame if source FPS and output FPS
    # do not match, while refreshing the time display.

    stream["last_panel"] = draw_panel(
        stream["frame"],
        stream,
        timestamp,
    )

    return stream["last_panel"]


# ============================================================
# 14. START FFMPEG ENCODER
# ============================================================

def start_encoder():

    ffmpeg_path = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",

        "-f",
        "rawvideo",

        "-vcodec",
        "rawvideo",

        "-pix_fmt",
        "bgr24",

        "-s",
        f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",

        "-r",
        str(OUTPUT_FPS),

        "-i",
        "-",

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        "medium",

        "-crf",
        "19",

        "-pix_fmt",
        "yuv420p",

        "-movflags",
        "+faststart",

        str(VIDEO_OUTPUT),
    ]

    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


# ============================================================
# 15. COMBINE TWO VIDEO PANELS
# ============================================================

def combine_panels(panels):

    combined = np.hstack(
        panels
    )

    expected_shape = (
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
        3,
    )

    if combined.shape != expected_shape:

        raise RuntimeError(
            f"Invalid output shape: "
            f"{combined.shape}. "
            f"Expected: {expected_shape}"
        )

    if combined.dtype != np.uint8:

        raise RuntimeError(
            "Output frames must use uint8"
        )

    return np.ascontiguousarray(
        combined
    )


# ============================================================
# 16. MAIN DEMO GENERATOR
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("VISIONTRACK - DEMO GENERATOR")
    print("=" * 60)

    # ----------------------------------------
    # Load YOLO
    # ----------------------------------------

    print("\nLoading YOLOv8n...")

    model = YOLO(
        str(MODEL_PATH)
        if MODEL_PATH.is_file()
        else "yolov8n.pt"
    )

    # ----------------------------------------
    # Create streams
    # ----------------------------------------

    streams = {}

    encoder = None
    encoder_closed = False

    screenshot_candidate = None

    best_screenshot_score = -1

    frames_written = 0

    try:

        for name, path in VIDEO_PATHS.items():

            stream = create_stream(
                name,
                path,
            )

            streams[name] = stream

            print(
                f"{name}: "
                f"{stream['width']}x"
                f"{stream['height']} "
                f"@ {stream['fps']:.2f} FPS"
            )

        # ------------------------------------
        # Start encoder
        # ------------------------------------

        encoder = start_encoder()

        frame_limit = int(
            MAX_DURATION_SECONDS * OUTPUT_FPS
        )

        print(
            f"\nOutput resolution: "
            f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
        )

        print(
            f"Output FPS: {OUTPUT_FPS}"
        )

        print(
            f"Maximum duration: "
            f"{MAX_DURATION_SECONDS} seconds"
        )

        print("\nGenerating demo...\n")

        # ------------------------------------
        # Main output timeline
        # ------------------------------------

        for output_index in range(
            frame_limit
        ):

            timestamp = (
                output_index / OUTPUT_FPS
            )

            panels = []

            for name in VIDEO_PATHS:

                stream = streams[name]

                panel = process_until(
                    stream,
                    timestamp,
                    model,
                )

                if panel is None:

                    panel = np.full(
                        (
                            PANEL_HEIGHT,
                            PANEL_WIDTH,
                            3,
                        ),
                        BACKGROUND,
                        dtype=np.uint8,
                    )

                panels.append(panel)

                # --------------------------------
                # Choose representative screenshot
                # from Crosswalk
                # --------------------------------

                if name == "Crosswalk":

                    crossings = (
                        int(
                            stream["zone"].in_count
                        )
                        +
                        int(
                            stream["zone"].out_count
                        )
                    )

                    people = stream["people"]

                    # Prefer frames with:
                    # 1. Actual crossings
                    # 2. Visible tracked people

                    if (
                        crossings > 0
                        and people >= 3
                    ):

                        score = (
                            min(crossings, 30) * 2
                            + min(people, 15)
                        )

                        if (
                            score
                            > best_screenshot_score
                        ):

                            best_screenshot_score = score

                            screenshot_candidate = (
                                panel.copy()
                            )

            # ------------------------------------
            # Combine both cameras
            # ------------------------------------

            combined = combine_panels(
                panels
            )

            # ------------------------------------
            # Check FFmpeg status
            # ------------------------------------

            if encoder.poll() is not None:

                raise RuntimeError(
                    "FFmpeg stopped unexpectedly"
                )

            # ------------------------------------
            # Write output frame
            # ------------------------------------

            encoder.stdin.write(
                combined.tobytes()
            )

            frames_written += 1

            # ------------------------------------
            # Progress display
            # ------------------------------------

            if (
                output_index % OUTPUT_FPS
                == 0
            ):

                print(
                    f"Processed timeline: "
                    f"{timestamp:.0f}s / "
                    f"{MAX_DURATION_SECONDS}s"
                )

            # ------------------------------------
            # Stop when both videos have ended
            # ------------------------------------

            if all(
                stream["finished"]
                for stream in streams.values()
            ):

                print(
                    "\nBoth source videos finished."
                )

                break

        # ----------------------------------------
        # Validate output
        # ----------------------------------------

        if frames_written == 0:

            raise RuntimeError(
                "No frames were generated"
            )

        # ----------------------------------------
        # Finalize FFmpeg
        # ----------------------------------------

        encoder.stdin.close()

        error_output = (
            encoder.stderr.read().decode(
                "utf-8",
                errors="replace",
            )
        )

        return_code = encoder.wait()

        encoder_closed = True

        if return_code != 0:

            raise RuntimeError(
                f"FFmpeg encoding failed:\n"
                f"{error_output}"
            )

        # ----------------------------------------
        # Save screenshot
        # ----------------------------------------

        if screenshot_candidate is None:

            # Fallback if no actual crossings
            # were observed in the selected interval.

            crosswalk = streams["Crosswalk"]

            screenshot_candidate = (
                crosswalk["last_panel"]
            )

        if screenshot_candidate is not None:

            success = cv2.imwrite(
                str(IMAGE_OUTPUT),
                screenshot_candidate,
            )

            if not success:

                raise RuntimeError(
                    "Could not save demo screenshot"
                )

        else:

            print(
                "Warning: screenshot unavailable."
            )

        # ----------------------------------------
        # Verify MP4
        # ----------------------------------------

        if not VIDEO_OUTPUT.is_file():

            raise RuntimeError(
                "Output MP4 not found"
            )

        if (
            VIDEO_OUTPUT.stat().st_size
            == 0
        ):

            raise RuntimeError(
                "Output MP4 is empty"
            )

        # ----------------------------------------
        # Final results
        # ----------------------------------------

        print("\n" + "=" * 60)

        print("DEMO GENERATION COMPLETE")

        print("=" * 60)

        print(
            f"Frames written: {frames_written}"
        )

        print(
            f"Duration: "
            f"{frames_written / OUTPUT_FPS:.2f} seconds"
        )

        print(
            f"Resolution: "
            f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}"
        )

        print(
            f"\nVideo: {VIDEO_OUTPUT}"
        )

        if IMAGE_OUTPUT.is_file():

            print(
                f"Screenshot: {IMAGE_OUTPUT}"
            )

        print("\nCounting results:")

        for name, stream in streams.items():

            print(
                f"{name}: "
                f"IN={int(stream['zone'].in_count)}, "
                f"OUT={int(stream['zone'].out_count)}"
            )

        print("=" * 60)

    finally:

        # ----------------------------------------
        # Release video resources
        # ----------------------------------------

        for stream in streams.values():

            stream["cap"].release()

        # ----------------------------------------
        # Close encoder if still active
        # ----------------------------------------

        if (
            encoder is not None
            and not encoder_closed
        ):

            if encoder.poll() is None:

                encoder.kill()

            encoder.wait()


# ============================================================
# 17. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()