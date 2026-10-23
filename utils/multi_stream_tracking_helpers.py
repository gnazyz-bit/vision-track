"""Per-camera worker with independent tracking, shared model and thread-safe snapshots."""
import logging
from pathlib import Path
import threading
import time

import cv2
import supervision as sv

from utils.preprocessing import display_frame
from utils.data_loader import open_video
from utils.counting_logic import (
    counting_line,
    make_counting_zone,
    update_line_count,
    draw_counting_line
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

        self.cap = open_video(self.source)

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

        self.p1, self.p2 = counting_line(
            name,
            self.width,
            self.height
        )

        self.tracker = sv.ByteTrack(
            frame_rate=int(round(self.source_fps))
        )

        self.zone = make_counting_zone(
            self.p1,
            self.p2
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

                update_line_count(
                    self.zone,
                    detections
                )

                draw_counting_line(
                    output,
                    self.p1,
                    self.p2
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