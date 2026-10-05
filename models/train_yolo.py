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

    epochs=30,
    imgsz=768,
    batch=8,

    optimizer="AdamW",
    lr0=0.0005,
    lrf=0.01, 
    cos_lr=True,

    weight_decay=0.0005,
    momentum=0.937,

    patience=10,

    # Data augmentation
    fliplr=0.5,
    translate=0.1,
    scale=0.4,
    mosaic=1.0,
    mixup=0.1,

    device=device,

    project=str(PROJECT_ROOT / "runs"),
    name="visiontrack_person_v4",
)