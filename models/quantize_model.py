
from pathlib import Path
import shutil

import cv2
import numpy as np
import onnx
import onnxruntime as ort

from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

from ultralytics import YOLO


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"

PT_MODEL = CHECKPOINT_DIR / "best.pt"

ONNX_MODEL = CHECKPOINT_DIR / "best.onnx"

QUANTIZED_MODEL = CHECKPOINT_DIR / "best_quantized.onnx"

DATASET_DIR = PROJECT_ROOT / "data" / "person_dataset"

IMAGE_SIZE = 640

CALIBRATION_IMAGES = 50


# --------------------------------------------------
# Export PyTorch model to ONNX
# --------------------------------------------------

def export_to_onnx():

    print("\n[1/4] Exporting YOLO to ONNX...")

    if not PT_MODEL.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {PT_MODEL}"
        )

    model = YOLO(str(PT_MODEL))

    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=IMAGE_SIZE,
            opset=17,
            dynamic=False,
            simplify=False
        )
    )

    if exported_path.resolve() != ONNX_MODEL.resolve():
        shutil.copy2(exported_path, ONNX_MODEL)

    onnx_model = onnx.load(str(ONNX_MODEL))
    onnx.checker.check_model(onnx_model)

    print(f"ONNX model saved: {ONNX_MODEL}")


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Cannot read image: {image_path}"
        )

    height, width = image.shape[:2]

    scale = min(
        IMAGE_SIZE / height,
        IMAGE_SIZE / width
    )

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Letterbox padding
    canvas = np.full(
        (IMAGE_SIZE, IMAGE_SIZE, 3),
        114,
        dtype=np.uint8
    )

    x = (IMAGE_SIZE - new_width) // 2
    y = (IMAGE_SIZE - new_height) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    # BGR to RGB
    image = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    )

    # Normalize
    image = image.astype(np.float32) / 255.0

    # HWC to CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return np.ascontiguousarray(
        image,
        dtype=np.float32
    )


# --------------------------------------------------
# Find calibration images
# --------------------------------------------------

def find_calibration_images():

    print("\n[2/4] Preparing calibration images...")

    image_extensions = {
        ".jpg", ".jpeg", ".png", ".bmp", ".webp"
    }

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_DIR}"
        )

    # Try common YOLO dataset structures.
    candidates = [
        DATASET_DIR / "images" / "train",
        DATASET_DIR / "train" / "images",
        DATASET_DIR / "images" / "val",
        DATASET_DIR / "valid" / "images",
    ]

    images = []

    for folder in candidates:
        if folder.is_dir():
            images = sorted(
                p for p in folder.rglob("*")
                if p.suffix.lower() in image_extensions
            )

            if images:
                break

    if not images:
        raise FileNotFoundError(
            "No calibration images found. "
            f"Check dataset structure: {DATASET_DIR}"
        )

    # Deterministic sampling across the folder
    indices = np.linspace(
        0,
        len(images) - 1,
        min(CALIBRATION_IMAGES, len(images)),
        dtype=int
    )

    selected = [images[i] for i in indices]

    print(f"Calibration images: {len(selected)}")

    return selected


# --------------------------------------------------
# Calibration Data Reader
# --------------------------------------------------

class YOLOCalibrationReader(CalibrationDataReader):

    def __init__(self, images, input_name):

        self.input_name = input_name

        self.data = []

        for image_path in images:
            try:
                tensor = preprocess_image(image_path)

                self.data.append({
                    self.input_name: tensor
                })

            except Exception as exc:
                print(
                    f"Skipping {image_path.name}: {exc}"
                )

        if not self.data:
            raise RuntimeError(
                "No valid calibration images available."
            )

        self.iterator = iter(self.data)

    def get_next(self):
        return next(self.iterator, None)

    def rewind(self):
        self.iterator = iter(self.data)


# --------------------------------------------------
# Static INT8 quantization
# --------------------------------------------------

def quantize_model(images):

    print("\n[3/4] Applying INT8 quantization...")

    session = ort.InferenceSession(
        str(ONNX_MODEL),
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    reader = YOLOCalibrationReader(
        images,
        input_name
    )

    quantize_static(
        model_input=str(ONNX_MODEL),
        model_output=str(QUANTIZED_MODEL),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Conv", "MatMul"],
    )

    onnx_model = onnx.load(str(QUANTIZED_MODEL))
    onnx.checker.check_model(onnx_model)

    print(
        f"Quantized model saved: {QUANTIZED_MODEL}"
    )


# --------------------------------------------------
# Validate ONNX Runtime inference
# --------------------------------------------------

def validate_models(image_path):

    print("\n[4/4] Validating ONNX Runtime...")

    tensor = preprocess_image(image_path)

    for model_path in [
        ONNX_MODEL,
        QUANTIZED_MODEL
    ]:

        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"]
        )

        input_name = session.get_inputs()[0].name

        outputs = session.run(
            None,
            {input_name: tensor}
        )

        if not outputs:
            raise RuntimeError(
                f"No outputs from {model_path.name}"
            )

        if not all(
            np.all(np.isfinite(output))
            for output in outputs
        ):
            raise RuntimeError(
                f"Invalid output values in {model_path.name}"
            )

        print(
            f"{model_path.name}: OK"
        )

        print(
            "Output shapes:",
            [output.shape for output in outputs]
        )

        print(
            "Model size:",
            round(
                model_path.stat().st_size / (1024 * 1024),
                2
            ),
            "MB"
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 50)
    print("VisionTrack - ONNX INT8 Quantization")
    print("=" * 50)

    print("ONNX Runtime version:", ort.__version__)

    export_to_onnx()

    images = find_calibration_images()

    quantize_model(images)

    validate_models(images[0])

    print("\n" + "=" * 50)
    print("Quantization completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
