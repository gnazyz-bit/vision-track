from pathlib import Path
import time

import cv2
import streamlit as st
import supervision as sv
from ultralytics import YOLO


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="VisionTrack",
    page_icon="🎥",
    layout="wide"
)

st.title("VisionTrack")
st.caption(
    "Multi-stream person detection, tracking and counting"
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

VIDEO_PATHS = {
    "Crosswalk": (
        PROJECT_ROOT
        / "data"
        / "raw_videos"
        / "crosswalk.mp4"
    ),

    "Mall": (
        PROJECT_ROOT
        / "data"
        / "raw_videos"
        / "mall.mp4"
    ),
}


# --------------------------------------------------
# Constants
# --------------------------------------------------

PERSON_CLASS_ID = 0


# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------

st.sidebar.header("Controls")

confidence_threshold = st.sidebar.slider(
    "Detection confidence",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05
)

enable_tracking = st.sidebar.toggle(
    "Tracking",
    value=True
)

enable_counting = st.sidebar.toggle(
    "Counting",
    value=True
)

start_button = st.sidebar.button(
    "Start processing",
    type="primary"
)


# --------------------------------------------------
# Load YOLO model
# --------------------------------------------------

@st.cache_resource
def load_model():

    return YOLO("yolov8n.pt")


model = load_model()


# --------------------------------------------------
# Create stream
# --------------------------------------------------

def create_stream(name, video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        return None


    frame_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    # --------------------------------------------------
    # Crosswalk counting line
    # --------------------------------------------------

    if name == "Crosswalk":

        line_y = int(
            frame_height * 0.51
        )

        line_start = sv.Point(
            x=0,
            y=line_y
        )

        line_end = sv.Point(
            x=int(
                frame_width * 0.58
            ),
            y=line_y
        )


    # --------------------------------------------------
    # Mall counting line
    # --------------------------------------------------

    elif name == "Mall":

        line_y = int(
            frame_height * 0.93
        )

        line_start = sv.Point(
            x=int(
                frame_width * 0.03
            ),
            y=line_y
        )

        line_end = sv.Point(
            x=int(
                frame_width * 0.91
            ),
            y=line_y
        )


    else:

        raise ValueError(
            f"Unknown stream: {name}"
        )


    # Each stream gets its own tracker
    tracker = sv.ByteTrack()


    # Each stream gets its own counting zone
    line_zone = sv.LineZone(
        start=line_start,
        end=line_end,
        triggering_anchors=[
            sv.Position.BOTTOM_CENTER
        ]
    )


    return {
        "cap": cap,
        "tracker": tracker,
        "line_zone": line_zone,
        "line_start": line_start,
        "line_end": line_end,
        "finished": False,
    }


# --------------------------------------------------
# Process one frame
# --------------------------------------------------

def process_frame(
    frame,
    stream,
    confidence_threshold,
    enable_tracking,
    enable_counting
):

    start_time = time.perf_counter()


    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    results = model(
        frame,
        classes=[PERSON_CLASS_ID],
        conf=confidence_threshold,
        verbose=False
    )

    result = results[0]

    detections = (
        sv.Detections.from_ultralytics(
            result
        )
    )


    # --------------------------------------------------
    # Tracking
    # --------------------------------------------------

    if enable_tracking:

        detections = (
            stream["tracker"]
            .update_with_detections(
                detections
            )
        )


    # --------------------------------------------------
    # Counting
    # --------------------------------------------------

    if (
        enable_tracking
        and enable_counting
    ):

        stream["line_zone"].trigger(
            detections
        )


    # --------------------------------------------------
    # Draw detections with tracker IDs
    # --------------------------------------------------

    if (
        enable_tracking
        and detections.tracker_id
        is not None
    ):

        for (
            xyxy,
            confidence,
            tracker_id
        ) in zip(
            detections.xyxy,
            detections.confidence,
            detections.tracker_id
        ):

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"ID {tracker_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------
    # Draw detections without tracking
    # --------------------------------------------------

    else:

        for (
            xyxy,
            confidence
        ) in zip(
            detections.xyxy,
            detections.confidence
        ):

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            label = (
                f"Person "
                f"{confidence:.2f}"
            )

            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(
                        y1 - 10,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------
    # Draw counting line
    # --------------------------------------------------

    if enable_counting:

        line_start = (
            stream["line_start"]
        )

        line_end = (
            stream["line_end"]
        )

        cv2.line(
            frame,
            (
                line_start.x,
                line_start.y
            ),
            (
                line_end.x,
                line_end.y
            ),
            (0, 0, 255),
            3
        )


    # --------------------------------------------------
    # Performance metrics
    # --------------------------------------------------

    latency_ms = (
        time.perf_counter()
        - start_time
    ) * 1000


    if latency_ms > 0:

        fps = (
            1000.0
            / latency_ms
        )

    else:

        fps = 0.0


    return (
        frame,
        fps,
        latency_ms
    )


# --------------------------------------------------
# Dashboard placeholders
# --------------------------------------------------

dashboard_placeholder = st.empty()


with dashboard_placeholder.container():

    col_crosswalk, col_mall = (
        st.columns(2)
    )


    # --------------------------------------------------
    # Crosswalk UI
    # --------------------------------------------------

    with col_crosswalk:

        st.subheader(
            "Crosswalk"
        )

        crosswalk_video = (
            st.empty()
        )

        crosswalk_metrics = (
            st.empty()
        )


    # --------------------------------------------------
    # Mall UI
    # --------------------------------------------------

    with col_mall:

        st.subheader(
            "Mall"
        )

        mall_video = (
            st.empty()
        )

        mall_metrics = (
            st.empty()
        )


# Separate placeholder for status messages
status_placeholder = st.empty()


# --------------------------------------------------
# Start processing
# --------------------------------------------------

if start_button:

    status_placeholder.info(
        "Processing videos..."
    )


    streams = {}


    # --------------------------------------------------
    # Initialize streams
    # --------------------------------------------------

    for name, path in (
        VIDEO_PATHS.items()
    ):

        stream = create_stream(
            name,
            path
        )


        if stream is None:

            status_placeholder.error(
                f"Could not open "
                f"{name} video: "
                f"{path}"
            )

            st.stop()


        streams[name] = stream


    # --------------------------------------------------
    # Main processing loop
    # --------------------------------------------------

    while True:

        all_finished = True


        for (
            name,
            stream
        ) in streams.items():

            if stream["finished"]:

                continue


            success, frame = (
                stream["cap"].read()
            )


            # ------------------------------------------
            # End of video
            # ------------------------------------------

            if not success:

                stream["finished"] = True

                continue


            all_finished = False


            # ------------------------------------------
            # Process frame
            # ------------------------------------------

            (
                processed_frame,
                fps,
                latency_ms
            ) = process_frame(
                frame,
                stream,
                confidence_threshold,
                enable_tracking,
                enable_counting
            )


            # ------------------------------------------
            # Convert OpenCV BGR to RGB
            # ------------------------------------------

            processed_frame = (
                cv2.cvtColor(
                    processed_frame,
                    cv2.COLOR_BGR2RGB
                )
            )


            # ------------------------------------------
            # Get counting results
            # ------------------------------------------

            line_zone = (
                stream["line_zone"]
            )


            if enable_counting:

                in_count = (
                    line_zone.in_count
                )

                out_count = (
                    line_zone.out_count
                )

            else:

                in_count = 0
                out_count = 0


            # ------------------------------------------
            # Crosswalk display
            # ------------------------------------------

            if name == "Crosswalk":

                crosswalk_video.image(
                    processed_frame,
                    channels="RGB",
                    width="stretch"
                )

                crosswalk_metrics.markdown(
                    f"""
**IN:** {in_count}  
**OUT:** {out_count}  
**FPS:** {fps:.1f}  
**Latency:** {latency_ms:.1f} ms
"""
                )


            # ------------------------------------------
            # Mall display
            # ------------------------------------------

            elif name == "Mall":

                mall_video.image(
                    processed_frame,
                    channels="RGB",
                    width="stretch"
                )

                mall_metrics.markdown(
                    f"""
**IN:** {in_count}  
**OUT:** {out_count}  
**FPS:** {fps:.1f}  
**Latency:** {latency_ms:.1f} ms
"""
                )


        # --------------------------------------------------
        # Stop when both videos finish
        # --------------------------------------------------

        if all_finished:

            break


    # --------------------------------------------------
    # Release videos
    # --------------------------------------------------

    for stream in streams.values():

        stream["cap"].release()


    # --------------------------------------------------
    # Final message
    # --------------------------------------------------

    status_placeholder.success(
        "Video processing finished."
    )