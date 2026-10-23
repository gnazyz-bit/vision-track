"""VisionTrack Streamlit dashboard. Run: python -m streamlit run app.py."""
from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import threading

import streamlit as st
from ultralytics import YOLO

from utils.data_loader import save_uploaded_video
from utils.multi_stream_tracking_helpers import VideoWorker

ROOT = Path(__file__).resolve().parent

VIDEO_PATHS = {
    "Crosswalk": ROOT / "data" / "raw_videos" / "crosswalk.mp4",
    "Mall": ROOT / "data" / "raw_videos" / "mall.mp4",
}

MODEL_PATH = ROOT / "yolov8n.pt"  # Demo selection justified in README

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
st.caption(
    "Two-camera person detection, tracking and line counting"
)


def load_model(path: str):
    return YOLO(path), threading.Lock()


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

                source = save_uploaded_video(
                    cfg["uploaded"],
                    directory.name,
                    name
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