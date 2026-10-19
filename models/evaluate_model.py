
"""
VisionTrack - Model Evaluation

Models:
1. Pre-trained YOLOv8n
2. Fine-tuned best.pt
3. INT8 quantized best_quantized.onnx

Outputs:
reports/performance_metrics.json

Run:
python3 models/evaluate_model.py
"""

import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

DATASET_YAML = (
    PROJECT_ROOT / "data" / "person_dataset" / "data.yaml"
)

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_PATH = REPORT_DIR / "performance_metrics.json"

VIDEO_PATHS = [
    PROJECT_ROOT / "data" / "raw_videos" / "crosswalk.mp4",
    PROJECT_ROOT / "data" / "raw_videos" / "mall.mp4",
]

MODELS = {
    "pretrained_yolov8n": PROJECT_ROOT / "yolov8n.pt",
    "fine_tuned": CHECKPOINT_DIR / "best.pt",
    "quantized_onnx": CHECKPOINT_DIR / "best_quantized.onnx",
}

# --------------------------------------------------
# Evaluation settings
# --------------------------------------------------

IMAGE_SIZE = 640
CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50

DEVICE = "cpu"

# Number of frames from each video for benchmarking
BENCHMARK_FRAMES = 60

# Use the pretrained model as the selected
# application model.
SELECTED_MODEL = "pretrained_yolov8n"


# --------------------------------------------------
# Detection metrics
# --------------------------------------------------

def evaluate_detection(model):

    results = model.val(
        data=str(DATASET_YAML),
        split="val",
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False,
        plots=False,
        save_json=False,
        project=str(REPORT_DIR / "validation_runs"),
        name="model_evaluation",
        exist_ok=True,
    )

    precision = float(results.box.mp)
    recall = float(results.box.mr)

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "map50": round(float(results.box.map50), 4),
        "map50_95": round(float(results.box.map), 4),
    }


# --------------------------------------------------
# Performance benchmark: two video streams
# --------------------------------------------------

def benchmark_model(model):

    import cv2

    captures = []

    for path in VIDEO_PATHS:

        if not path.is_file():
            raise FileNotFoundError(
                f"Video not found: {path}"
            )

        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Cannot open video: {path}"
            )

        captures.append(cap)

    frame_counts = [0] * len(captures)
    latencies = []
    active = [True] * len(captures)

    try:

        # Model warmup
        dummy = np.zeros(
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            dtype=np.uint8
        )

        for _ in range(3):
            model.predict(
                dummy,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device=DEVICE,
                verbose=False,
            )

        start_time = time.perf_counter()

        while any(active):

            for i, cap in enumerate(captures):

                if not active[i]:
                    continue

                if frame_counts[i] >= BENCHMARK_FRAMES:
                    active[i] = False
                    continue

                ok, frame = cap.read()

                if not ok:
                    active[i] = False
                    continue

                t0 = time.perf_counter()

                model.predict(
                    frame,
                    imgsz=IMAGE_SIZE,
                    conf=CONFIDENCE,
                    classes=[0],
                    device=DEVICE,
                    verbose=False,
                )

                latency_ms = (
                    time.perf_counter() - t0
                ) * 1000.0

                latencies.append(latency_ms)
                frame_counts[i] += 1

        total_time = time.perf_counter() - start_time

    finally:

        for cap in captures:
            cap.release()

    if total_time <= 0 or not latencies:
        raise RuntimeError(
            "Benchmark did not process any frames."
        )

    fps_per_stream = [
        round(count / total_time, 2)
        for count in frame_counts
    ]

    return {
        "frames_processed_per_stream": frame_counts,
        "fps_per_stream": fps_per_stream,
        "average_fps_per_stream": round(
            float(np.mean(fps_per_stream)), 2
        ),
        "average_latency_ms": round(
            float(np.mean(latencies)), 2
        ),
        "total_benchmark_seconds": round(
            total_time, 2
        ),
    }


# --------------------------------------------------
# Validate paths
# --------------------------------------------------

def check_files():

    if not DATASET_YAML.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATASET_YAML}"
        )

    for name, path in MODELS.items():

        if not path.is_file():
            raise FileNotFoundError(
                f"Model missing ({name}): {path}"
            )

        if path.stat().st_size == 0:
            raise RuntimeError(
                f"Model is empty ({name}): {path}"
            )


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    check_files()

    print("=" * 55)
    print("VisionTrack - Model Evaluation")
    print("=" * 55)

    print(f"Dataset: {DATASET_YAML}")
    print(f"Device: {DEVICE}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"IoU: {IOU_THRESHOLD}")
    print()

    all_results = {}

    for model_name, model_path in MODELS.items():

        print("\n" + "-" * 55)
        print(f"Evaluating: {model_name}")
        print("-" * 55)

        try:

            model = YOLO(str(model_path))

            print("\n[1/2] Detection evaluation...")

            detection = evaluate_detection(model)

            print("Detection metrics:")
            print(json.dumps(detection, indent=2))

            print("\n[2/2] Performance benchmark...")

            performance = benchmark_model(model)

            print("Performance:")
            print(json.dumps(performance, indent=2))

            all_results[model_name] = {
                "status": "success",
                "model_path": str(model_path),
                "detection": detection,
                "performance": performance,
            }

        except Exception as exc:

            print(f"\nERROR in {model_name}: {exc}")

            all_results[model_name] = {
                "status": "failed",
                "model_path": str(model_path),
                "error": str(exc),
            }

    # --------------------------------------------------
    # Build audit report
    # --------------------------------------------------

    selected = all_results.get(SELECTED_MODEL, {})

    detection = selected.get("detection", {})
    performance = selected.get("performance", {})

    report = {
        "selected_model": SELECTED_MODEL,
        "evaluation_dataset": str(DATASET_YAML),
        "evaluation_split": "val",
        "evaluation_device": DEVICE,
        "confidence_threshold": CONFIDENCE,
        "iou_threshold": IOU_THRESHOLD,

        "detection_precision": detection.get("precision"),
        "detection_recall": detection.get("recall"),
        "f1_score": detection.get("f1_score"),

        "average_fps_per_stream": performance.get(
            "average_fps_per_stream"
        ),
        "average_latency_ms": performance.get(
            "average_latency_ms"
        ),

        "models": all_results,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 55)
    print(f"Report saved: {REPORT_PATH}")
    print("=" * 55)

    successful = [
        name
        for name, result in all_results.items()
        if result["status"] == "success"
    ]

    print(
        f"Successfully evaluated: "
        f"{len(successful)}/{len(MODELS)} models"
    )

    if len(successful) != len(MODELS):
        raise RuntimeError(
            "Evaluation incomplete. "
            "Review errors in performance_metrics.json."
        )

    print("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
