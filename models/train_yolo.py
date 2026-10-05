from pathlib import Path

import torch
from ultralytics import YOLO


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_YAML = (
    PROJECT_ROOT
    / "data"
    / "person_dataset"
    / "data.yaml"
)


# --------------------------------------------------
# Device selection
# --------------------------------------------------

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# --------------------------------------------------
# Load pretrained YOLO model
# --------------------------------------------------

model = YOLO("yolov8n.pt")


# --------------------------------------------------
# Fine-tuning
# --------------------------------------------------

results = model.train(
    data=str(DATASET_YAML),

    epochs=20,
    imgsz=640,
    batch=8,

    optimizer="AdamW",
    lr0=0.001,

    device=device,

    patience=7,

    project=str(PROJECT_ROOT / "runs"),
    name="visiontrack_person_v2",
)