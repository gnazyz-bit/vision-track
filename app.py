"""VisionTrack: simultaneous two-video detection, tracking and counting.
Run: python -m streamlit run app.py
"""
from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import threading
import time

import cv2
import numpy as np
import streamlit as st
import supervision as sv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

VIDEO_PATHS = {
    "Crosswalk": ROOT / "data" / "raw_videos" / "crosswalk.mp4",
    "Mall": ROOT / "data" / "raw_videos" / "mall.mp4",
}

MODEL_PATH = ROOT / "yolov8n.pt"

DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 540

LOG_FILE = ROOT / "logs" / "app_errors.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.ERROR
)

st.set_page_config(
    page_title="VisionTrack",
    page_icon="🎥",
    layout="wide"
)

st.title("VisionTrack")
st.caption("Two-camera person detection, tracking and line counting")


@st.cache_resource(show_spinner="Loading YOLO...")
def load_model(path: str):
    return YOLO(path), threading.Lock()


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


class VideoWorker:

    def __init__(
        self,
        name,
        source,
        model,
        model_lock,
        config,
        confidence,
        threshold
    ):

        self.name = name
        self.source = Path(source)

        self.model = model
        self.model_lock = model_lock

        self.config = config
        self.confidence = confidence
        self.threshold = threshold

        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.cap = cv2.VideoCapture(str(self.source))

        if not self.cap.isOpened():
            self.cap.release()
            raise OSError(
                f"Cannot open {name}: {self.source}"
            )

        self.width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        self.height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if self.width <= 0 or self.height <= 0:
            self.cap.release()
            raise ValueError(
                f"Invalid video dimensions: {name}"
            )

        self.source_fps = float(
            self.cap.get(cv2.CAP_PROP_FPS)
        )

        if not 1 <= self.source_fps <= 120:
            self.source_fps = 25.0

        if name == "Crosswalk":

            y = int(self.height * 0.51)

            p1 = sv.Point(0, y)

            p2 = sv.Point(
                int(self.width * 0.58),
                y
            )

        else:

            y = int(self.height * 0.93)

            p1 = sv.Point(
                int(self.width * 0.03),
                y
            )

            p2 = sv.Point(
                int(self.width * 0.91),
                y
            )

        self.p1 = p1
        self.p2 = p2

        self.tracker = sv.ByteTrack(
            frame_rate=int(round(self.source_fps))
        )

        self.zone = sv.LineZone(
            start=p1,
            end=p2,
            triggering_anchors=[
                sv.Position.BOTTOM_CENTER
            ]
        )

        self.frame_rgb = None

        self.people = 0
        self.in_count = 0
        self.out_count = 0

        self.processing_fps = 0.0
        self.latency_ms = 0.0

        self.done = False
        self.error = None

        # Show the first frame before inference
        ok, first = self.cap.read()

        if ok:

            self.frame_rgb = display_frame(first)

            self.cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

        else:

            self.done = True
            self.cap.release()

    def start(self):

        if not self.done:

            self.thread = threading.Thread(
                target=self.run,
                name=f"VisionTrack-{self.name}",
                daemon=True
            )

            self.thread.start()

    def stop(self):

        self.stop_event.set()

        if self.thread and self.thread.is_alive():

            self.thread.join(timeout=3)

        if not self.thread or not self.thread.is_alive():

            self.cap.release()

    def snapshot(self):

        with self.lock:

            return {
                "image": self.frame_rgb,
                "people": self.people,
                "in": self.in_count,
                "out": self.out_count,
                "fps": self.processing_fps,
                "latency": self.latency_ms,
                "done": self.done,
                "error": self.error,
            }

    def process(self, frame):

        output = frame.copy()

        people = 0

        if self.config["detection"]:

            # Shared YOLO model
            with self.model_lock:

                prediction = self.model(
                    frame,
                    classes=[0],
                    conf=self.confidence,
                    verbose=False
                )[0]

            detections = (
                sv.Detections.from_ultralytics(
                    prediction
                )
            )

            if self.config["tracking"]:

                detections = (
                    self.tracker.update_with_detections(
                        detections
                    )
                )

            people = len(detections)

            ids = detections.tracker_id

            for i, (box, score) in enumerate(
                zip(
                    detections.xyxy,
                    detections.confidence
                )
            ):

                x1, y1, x2, y2 = (
                    int(v) for v in box
                )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 220, 0),
                    2
                )

                label = (
                    f"Person {float(score):.2f}"
                )

                if (
                    self.config["tracking"]
                    and ids is not None
                ):

                    label = (
                        f"ID {int(ids[i])} "
                        f"{float(score):.2f}"
                    )

                cv2.putText(
                    output,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 220, 0),
                    2
                )

            if self.config["counting"]:

                self.zone.trigger(detections)

                cv2.line(
                    output,
                    (
                        int(self.p1.x),
                        int(self.p1.y)
                    ),
                    (
                        int(self.p2.x),
                        int(self.p2.y)
                    ),
                    (0, 0, 255),
                    3
                )

        return output, people

    def run(self):

        frame_number = 0
        started_at = time.perf_counter()

        try:

            while not self.stop_event.is_set():

                # Independent playback clock
                target_time = (
                    started_at
                    + frame_number / self.source_fps
                )

                wait = (
                    target_time
                    - time.perf_counter()
                )

                if wait > 0:

                    if self.stop_event.wait(wait):
                        break

                success, frame = self.cap.read()

                if not success:
                    break

                t0 = time.perf_counter()

                result, people = self.process(frame)

                latency = (
                    time.perf_counter() - t0
                ) * 1000

                rgb = display_frame(result)

                with self.lock:

                    self.frame_rgb = rgb

                    self.people = people

                    self.in_count = int(
                        self.zone.in_count
                    )

                    self.out_count = int(
                        self.zone.out_count
                    )

                    self.latency_ms = latency

                    self.processing_fps = (
                        1000 / latency
                        if latency > 0
                        else 0.0
                    )

                frame_number += 1

        except Exception as exc:

            logging.exception(
                "Error processing %s",
                self.name
            )

            with self.lock:
                self.error = str(exc)

        finally:

            self.cap.release()

            with self.lock:
                self.done = True


def stop_workers():

    workers = st.session_state.get(
        "workers",
        {}
    )

    for worker in workers.values():
        worker.stop()

    st.session_state.workers = {}
    st.session_state.running = False

    temp = st.session_state.get("upload_dir")

    if temp is not None:

        if all(
            not w.thread
            or not w.thread.is_alive()
            for w in workers.values()
        ):

            temp.cleanup()

            st.session_state.upload_dir = None


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "workers" not in st.session_state:
    st.session_state.workers = {}

if "running" not in st.session_state:
    st.session_state.running = False

if "upload_dir" not in st.session_state:
    st.session_state.upload_dir = None


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.header("Controls")

confidence = st.sidebar.slider(
    "Detection confidence",
    0.10,
    0.95,
    0.35,
    0.05
)

threshold = st.sidebar.number_input(
    "Crowd threshold (people)",
    min_value=1,
    value=20
)

st.sidebar.caption(
    "Model: pre-trained YOLOv8n"
)

camera_config = {}

for name in VIDEO_PATHS:

    with st.sidebar.expander(
        name,
        expanded=True
    ):

        enabled = st.checkbox(
            "Enable camera",
            value=True,
            key=f"enabled_{name}"
        )

        uploaded = st.file_uploader(
            "Upload video (optional)",
            type=["mp4", "mov", "avi"],
            key=f"upload_{name}"
        )

        detection = st.checkbox(
            "Detection",
            value=True,
            key=f"detection_{name}"
        )

        tracking = st.checkbox(
            "Tracking",
            value=True,
            disabled=not detection,
            key=f"tracking_{name}"
        )

        counting = st.checkbox(
            "Counting",
            value=True,
            disabled=not (
                detection and tracking
            ),
            key=f"counting_{name}"
        )

        camera_config[name] = {
            "enabled": enabled,
            "uploaded": uploaded,
            "detection": detection,
            "tracking": detection and tracking,
            "counting": (
                detection
                and tracking
                and counting
            ),
        }


# --------------------------------------------------
# Start / Stop buttons
# --------------------------------------------------

left, right = st.sidebar.columns(2)

start_clicked = left.button(
    "Start / Restart",
    type="primary",
    use_container_width=True
)

stop_clicked = right.button(
    "Stop",
    use_container_width=True
)

if stop_clicked:
    stop_workers()

if start_clicked:

    stop_workers()

    try:

        model, model_lock = load_model(
            str(MODEL_PATH)
            if MODEL_PATH.is_file()
            else "yolov8n.pt"
        )

        directory = tempfile.TemporaryDirectory(
            prefix="visiontrack_"
        )

        st.session_state.upload_dir = directory

        new_workers = {}

        for name, default_source in VIDEO_PATHS.items():

            cfg = camera_config[name]

            if not cfg["enabled"]:
                continue

            source = default_source

            if cfg["uploaded"] is not None:

                suffix = Path(
                    cfg["uploaded"].name
                ).suffix.lower()

                source = (
                    Path(directory.name)
                    / f"{name}{suffix}"
                )

                source.write_bytes(
                    cfg["uploaded"].getvalue()
                )

            try:

                worker = VideoWorker(
                    name,
                    source,
                    model,
                    model_lock,
                    cfg,
                    confidence,
                    threshold
                )

                new_workers[name] = worker

            except Exception as exc:

                logging.exception(
                    "Unable to open %s",
                    name
                )

                st.error(
                    f"{name}: {exc}"
                )

        st.session_state.workers = new_workers

        st.session_state.running = bool(
            new_workers
        )

        for worker in new_workers.values():
            worker.start()

        if not new_workers:

            stop_workers()

            st.warning(
                "No video source available."
            )

    except Exception as exc:

        logging.exception(
            "Start failure"
        )

        stop_workers()

        st.error(
            f"Start failed: {exc}"
        )


if st.session_state.running:

    st.sidebar.caption(
        "To apply changed settings, "
        "click Start / Restart."
    )


# --------------------------------------------------
# Live dashboard
# --------------------------------------------------

@st.fragment(run_every="0.2s")
def dashboard():

    workers = st.session_state.workers

    if not workers:

        st.info(
            "Enable cameras and click Start / Restart."
        )

        return

    columns = st.columns(
        2,
        gap="medium"
    )

    snapshots = {}

    for column, name in zip(
        columns,
        VIDEO_PATHS
    ):

        with column:

            st.subheader(name)

            worker = workers.get(name)

            if worker is None:

                st.caption(
                    "Camera disabled or video unavailable"
                )

                continue

            snap = worker.snapshot()

            snapshots[name] = snap

            if snap["image"] is not None:

                st.image(
                    snap["image"],
                    channels="RGB",
                    use_container_width=True
                )

            else:

                st.info(
                    "Loading video..."
                )

            if worker.config["counting"]:

                st.write(
                    f'**IN:** {snap["in"]}  |  '
                    f'**OUT:** {snap["out"]}'
                )

            else:

                st.write(
                    "**Counting:** off"
                )

            st.write(
                f'**People detected:** {snap["people"]}  |  '
                f'**Processing FPS:** {snap["fps"]:.1f}  |  '
                f'**Latency:** {snap["latency"]:.1f} ms'
            )

            if (
                worker.config["detection"]
                and snap["people"] > worker.threshold
            ):

                st.warning(
                    "Crowd threshold exceeded: "
                    f'{snap["people"]} people'
                )

            if snap["error"]:

                st.error(
                    snap["error"]
                )

            elif snap["done"]:

                st.caption(
                    "Video finished"
                )

    if snapshots and all(
        s["done"]
        for s in snapshots.values()
    ):

        st.session_state.running = False

        st.success(
            "Video processing finished."
        )


dashboard()