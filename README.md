
# VisionTrack: Multi-Stream Person Detection, Tracking and Counting

VisionTrack is a computer vision application for detecting, tracking, and counting people across multiple video streams.

The project uses YOLOv8 for person detection, ByteTrack for multi-object tracking, Supervision LineZone for entry/exit counting, and Streamlit for interactive monitoring.

VisionTrack was developed as a proof-of-concept smart city surveillance system for public spaces such as pedestrian crossings, transit stations, and shopping centers.

## 1. Project Objectives

The main objectives are:

- Detect people using YOLOv8.
- Track individuals using ByteTrack.
- Maintain tracking IDs between consecutive frames.
- Count people crossing predefined lines.
- Process two video sources simultaneously.
- Develop an interactive Streamlit dashboard.
- Display FPS, latency, and crowd monitoring alerts.
- Fine-tune a pretrained YOLOv8 model.
- Export the trained model to ONNX.
- Apply INT8 quantization.
- Compare model accuracy and inference speed.
- Generate demonstration images and videos.

## 2. Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Ultralytics YOLOv8 | Person detection and training |
| PyTorch | Model training and inference |
| Supervision | ByteTrack and LineZone |
| OpenCV | Video processing |
| Streamlit | Interactive dashboard |
| ONNX | Model export |
| ONNX Runtime | ONNX inference and quantization |
| NumPy | Image processing |
| Pandas | Experimental analysis |
| Matplotlib | Charts and visualization |
| Jupyter Notebook | Experiment documentation |
| imageio-ffmpeg | H.264 video export |

## 3. System Architecture

The processing pipeline is:

```text
Video Sources (Crosswalk / Mall)
            |
            v
      Frame Acquisition
            |
            v
     YOLOv8 Detection
            |
            v
    Supervision Detections
            |
            v
     ByteTrack Tracking
            |
            v
      LineZone Counting
            |
            v
   Visualization and Metrics
            |
            v
     Streamlit Dashboard
```

Each camera uses an independent ByteTrack tracker and LineZone counter.

The application uses separate video processing worker threads and a shared YOLO model protected by a lock.

Tracking IDs are local to each video stream. Cross-camera re-identification is not implemented.

## 4. Main Features

### 4.1 Person Detection

YOLOv8 detects people in video frames.

Only the `person` class is selected:

```python
classes=[0]
```

The default detection confidence is 0.35.

The main Streamlit application uses pretrained YOLOv8n weights.

### 4.2 Person Tracking

ByteTrack tracks detected people across frames and assigns tracking IDs.

Each video stream uses its own tracker to maintain independent tracking state.

Bounding boxes and tracking IDs are displayed on the processed video.

### 4.3 Entry and Exit Counting

Supervision LineZone registers crossings of a predefined line.

Two directional counters are maintained:

- IN: crossings in the first direction.
- OUT: crossings in the opposite direction.

The demonstration cameras use the following line positions:

**Crosswalk**

- Vertical position: 51% of frame height.
- Horizontal coverage: 0% to 58% of frame width.

**Mall**

- Vertical position: 93% of frame height.
- Horizontal coverage: 3% to 91% of frame width.

Counting is based on line crossings, not polygon occupancy.

### 4.4 Multi-Stream Processing

Two default videos are supported:

```text
data/raw_videos/crosswalk.mp4
data/raw_videos/mall.mp4
```

The Streamlit dashboard displays Crosswalk and Mall side by side.

Each camera has independent tracking and counting states.

### 4.5 Streamlit Dashboard

The interface provides:

- Independent camera activation.
- MP4, MOV, and AVI video uploads.
- Detection controls per camera.
- Tracking controls per camera.
- Counting controls per camera.
- Detection confidence slider.
- Crowd threshold configuration.
- Start, Restart, and Stop buttons.
- Bounding boxes and tracking IDs.
- IN and OUT counters.
- Detected-person counts.
- Processing FPS and latency.
- Crowd threshold warnings.

Changes to processing settings are applied when the workers are restarted.

### 4.6 Crowd Alerts

VisionTrack displays a warning when the detected number of people exceeds the configured threshold.

The alert represents a count of detected people within the camera frame. It does not estimate physical crowd density per square meter.

## 5. Project Structure

```text
vision-track/
├── app.py
├── data/
│   ├── raw_videos/
│   │   ├── crosswalk.mp4
│   │   └── mall.mp4
│   └── person_dataset/
│       ├── train/
│       ├── valid/
│       └── data.yaml
├── models/
│   ├── train_yolo.py
│   ├── quantize_model.py
│   ├── evaluate_model.py
│   └── checkpoints/
│       ├── best.pt
│       ├── best.onnx
│       ├── best_quantized.onnx
│       └── config.yaml
├── scripts/
│   └── create_demo.py
├── utils/
├── notebooks/
│   └── VisionTrack_Analysis.ipynb
├── reports/
│   ├── performance_metrics.json
│   ├── experiments/
│   └── demo_results/
│       ├── roi_counting_example.png
│       └── multi_stream_demo.mp4
├── logs/
│   └── app_errors.log
├── requirements.txt
└── README.md
```

Additional training logs, experiments, and helper scripts may be present.

## 6. Installation

### Requirements

- Python 3.9 or newer with compatible package versions.
- pip.
- CPU or supported hardware accelerator.
- Sufficient disk space for model weights and video files.

The project was developed and evaluated on an Apple M4 computer.

CUDA is not required to run the application.

### Create a Virtual Environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

Ensure that VS Code uses the Python interpreter from the same virtual environment.

## 7. Running VisionTrack

From the project root:

```bash
streamlit run app.py
```

Alternatively:

```bash
python3 -m streamlit run app.py
```

Open the local Streamlit URL displayed in the terminal.

Usually:

```text
http://localhost:8501
```

### How to Use

1. Open the Streamlit application.
2. Configure detection confidence.
3. Set the crowd threshold.
4. Enable Crosswalk, Mall, or both cameras.
5. Optionally upload a video for either camera.
6. Enable Detection, Tracking, and Counting.
7. Click Start / Restart.
8. Observe bounding boxes, tracking IDs, counts, FPS, and latency.
9. Click Stop to terminate processing.

For custom video uploads, the counting line follows the predefined relative coordinates for the selected camera slot.

## 8. YOLO Training and Transfer Learning

### 8.1 Baseline Model

The baseline detector is YOLOv8n pretrained on the COCO dataset.

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

### 8.2 Dataset

A custom person detection dataset was used for transfer learning.

| Split | Images |
|---|---:|
| Training | 376 |
| Validation | 96 |

The validation dataset contains 249 annotated person instances.

The dataset uses YOLO-format annotations.

### 8.3 Training Configuration

Training is implemented in:

```text
models/train_yolo.py
```

Configured parameters:

| Parameter | Value |
|---|---|
| Model | YOLOv8n |
| Epochs | 30 |
| Image size | 768 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.0005 |
| Final learning rate factor | 0.01 |
| Cosine LR scheduler | Enabled |
| Weight decay | 0.0005 |
| Momentum | 0.937 |
| Early stopping patience | 10 |
| Horizontal flip | 0.5 |
| Translation | 0.1 |
| Scale | 0.4 |
| Mosaic | 1.0 |
| MixUp | 0.1 |

These are the configured training settings. The actual best epoch and total completed epochs can be verified using the saved training logs.

The resulting model checkpoint is:

```text
models/checkpoints/best.pt
```

### Run Training

```bash
python3 models/train_yolo.py
```

Training supports device selection based on CUDA, Apple MPS, or CPU availability.

## 9. ONNX Export and INT8 Quantization

The fine-tuned model was exported to ONNX and quantized using ONNX Runtime.

Optimization script:

```text
models/quantize_model.py
```

Run:

```bash
python3 models/quantize_model.py
```

The optimization process:

1. Loads `best.pt`.
2. Exports the model to ONNX.
3. Prepares 50 calibration images.
4. Applies static INT8 quantization.
5. Saves the quantized model.
6. Validates ONNX Runtime inference.

Quantization uses the QDQ format and a CPU execution provider.

### Model Size Results

| Model | Size |
|---|---:|
| Original ONNX | 11.67 MB |
| Quantized ONNX | 3.22 MB |

The quantized model is approximately 72.4% smaller.

Model artifacts:

```text
models/checkpoints/best.pt
models/checkpoints/best.onnx
models/checkpoints/best_quantized.onnx
```

The ONNX Runtime validation checks confirmed that the original and quantized models successfully produced numerical outputs.

## 10. Evaluation Methodology

The evaluation script is:

```text
models/evaluate_model.py
```

Run:

```bash
python3 models/evaluate_model.py
```

Three models were evaluated:

1. Pretrained YOLOv8n.
2. Fine-tuned YOLOv8n.
3. INT8 quantized ONNX.

### Evaluation Configuration

| Parameter | Value |
|---|---|
| Dataset split | Validation |
| Images | 96 |
| Ground-truth person instances | 249 |
| Image size | 640 |
| Confidence threshold | 0.35 |
| Evaluator IoU setting | 0.50 |
| Device | Apple M4 CPU |
| Ultralytics | 8.4.182 |
| ONNX Runtime | 1.19.2 |

Precision and Recall are taken from aggregate Ultralytics validation statistics.

F1-score is calculated as:

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

The results describe performance on the validation dataset, not an independent test dataset.

## 11. Detection Evaluation Results

| Model | Precision | Recall | F1-score | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Pretrained YOLOv8n | 0.7538 | 0.7470 | 0.7504 | 0.7311 | 0.5308 |
| Fine-tuned YOLOv8n | 0.8578 | 0.7590 | 0.8054 | 0.7629 | 0.5865 |
| INT8 Quantized ONNX | 0.8729 | 0.7349 | 0.7980 | 0.7459 | 0.5575 |

### Evaluation Analysis

The fine-tuned YOLOv8n model achieved the highest F1-score, mAP50, and mAP50-95 on the validation dataset.

The F1-score increased from 0.7504 to 0.8054 after fine-tuning, an improvement of approximately 5.5 percentage points.

The quantized model retained much of the fine-tuned model's detection quality.

Although the quantized model achieved the highest Precision, its Recall and F1-score were lower than those of the fine-tuned PyTorch model.

## 12. Performance Benchmark

A separate benchmark measured inference performance while processing frames from two video sources.

The benchmark processed 60 frames from Crosswalk and 60 frames from Mall.

| Model | Average FPS per stream | Average latency |
|---|---:|---:|
| Pretrained YOLOv8n | 22.02 | 19.63 ms |
| Fine-tuned YOLOv8n | 23.67 | 18.08 ms |
| INT8 Quantized ONNX | 16.35 | 27.32 ms |

The fine-tuned model achieved the highest measured throughput.

The quantized ONNX model was slower on the tested Apple M4 CPU configuration.

Therefore, quantization improved storage efficiency but did not provide an inference speedup in this benchmark.

The FPS values reflect benchmark throughput under a two-stream workload. They are not the actual rendering FPS of the Streamlit dashboard.

The results are stored in:

```text
reports/performance_metrics.json
```

## 13. Confidence Threshold Experiments

Four experiments compared pretrained and fine-tuned YOLOv8n models on the same demonstration video sources.

| Model | Confidence | Crosswalk IN | Crosswalk OUT | Mall IN | Mall OUT |
|---|---:|---:|---:|---:|---:|
| Fine-tuned YOLOv8n | 0.50 | 0 | 4 | 7 | 7 |
| Fine-tuned YOLOv8n | 0.35 | 0 | 9 | 7 | 10 |
| Pretrained YOLOv8n | 0.50 | 3 | 16 | 9 | 12 |
| Pretrained YOLOv8n | 0.35 | 5 | 21 | 8 | 12 |

The pretrained detector registered more line-crossing events on the demonstration videos.

However, a larger number of recorded crossings does not automatically imply higher accuracy.

Manual ground-truth counting is needed to determine which model produces the most accurate entry/exit counts.

The fine-tuned model achieved better detection metrics on the labeled validation dataset, while the pretrained model produced more detections and recorded crossings in the two demonstration scenes.

These observations highlight the importance of evaluating computer vision models under different deployment conditions.

The complete experiment analysis is available in:

```text
notebooks/VisionTrack_Analysis.ipynb
```

## 14. Demonstration Results

The demonstration generator is:

```text
scripts/create_demo.py
```

Run:

```bash
python3 scripts/create_demo.py
```

The script processes Crosswalk and Mall with YOLOv8n, ByteTrack, and LineZone.

It generates:

### ROI Counting Example

```text
reports/demo_results/roi_counting_example.png
```

This image includes:

- A real video frame.
- Bounding boxes around detected people.
- Tracking IDs.
- A visible counting line.
- IN, OUT, and tracked-person statistics.

### Multi-Stream Demo

```text
reports/demo_results/multi_stream_demo.mp4
```

The demonstration video shows:

- Crosswalk and Mall side by side.
- YOLO person detection.
- ByteTrack IDs.
- LineZone counting lines.
- Dynamic counting statistics.
- Two video panels.

The MP4 is encoded using H.264 through imageio-ffmpeg.

The demonstration is generated for a configured maximum duration of 30 seconds.

It is intended to demonstrate the system visually and is not a substitute for quantitative tracking or counting validation.

## 15. Error Handling

The Streamlit application includes basic handling for:

- Missing video files.
- Video initialization failures.
- Invalid frame dimensions.
- Model inference exceptions.
- Video worker termination.
- Stopping and restarting processing.

Errors are logged to:

```text
logs/app_errors.log
```

The system runs on CPU when CUDA is unavailable.

The evaluation results in this report were obtained using CPU inference on an Apple M4 device.

## 16. Limitations

The current implementation has the following limitations:

1. Tracking IDs are independent for each camera.
2. Cross-camera person re-identification is not implemented.
3. Counting is based on line crossings, not arbitrary polygon occupancy.
4. Counting accuracy depends on detection and tracking stability.
5. Line coordinates are predefined and may require adjustment for custom video sources.
6. Crowd warnings are based on detected-person counts, not physical crowd density.
7. Validation results come from 96 labeled images, not a separate independent test dataset.
8. INT8 quantization reduced model size but did not increase inference speed on Apple M4 CPU.
9. Benchmark FPS does not represent actual Streamlit display FPS.
10. Manual validation of line-crossing counts is still required for a quantitative counting-accuracy claim.

## 17. Reproducibility

The project can be reproduced using the following commands.

### Train Model

```bash
python3 models/train_yolo.py
```

### Export and Quantize ONNX

```bash
python3 models/quantize_model.py
```

### Evaluate Models

```bash
python3 models/evaluate_model.py
```

### Generate Demonstration Results

```bash
python3 scripts/create_demo.py
```

### Run Streamlit

```bash
python3 -m streamlit run app.py
```

### Open Analysis Notebook

```text
notebooks/VisionTrack_Analysis.ipynb
```

All measured evaluation results are saved in `reports/performance_metrics.json`.

Training settings and model artifacts are stored in the corresponding model and checkpoint directories.

## 18. Conclusion

VisionTrack demonstrates an end-to-end computer vision workflow combining YOLOv8 person detection, ByteTrack multi-object tracking, Supervision LineZone counting, and Streamlit-based visualization.

The application supports two video sources with independent tracking and counting states.

Fine-tuning improved detection quality on the labeled validation dataset, increasing F1-score from 0.7504 to 0.8054.

INT8 quantization reduced the ONNX model size by approximately 72.4%, while maintaining an F1-score of 0.7980.

However, ONNX INT8 inference was slower than PyTorch inference on the tested Apple M4 CPU setup.

These results demonstrate the practical trade-offs between detection accuracy, model size, and performance.

VisionTrack provides a foundation for future work in crowd monitoring, custom ROI configuration, improved tracking stability, and larger-scale multi-camera video analytics.
