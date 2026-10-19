# VisionTrack

Multi-stream person detection, tracking and counting with YOLO, supervision and Streamlit.

## Run (with environment dependencies installed)

```bash
python3 -m streamlit run app.py
```

The demo expects `data/raw_videos/crosswalk.mp4` and `data/raw_videos/mall.mp4`. Model used depends on the current configuration in `app.py`; this archive retains the original app code.

## Model comparison experiment (2026-10-19)

See [VisionTrack Analysis Notebook](notebooks/VisionTrack_Analysis.ipynb) and [experiment data](reports/experiments/model_comparison.csv). Four Streamlit runs compare pretrained `yolov8n.pt` and fine-tuned `models/checkpoints/best.pt` at confidence 0.50 and 0.35. Screenshots are archived in `reports/experiments/screenshots/`.

**Caution:** the experiment reports observed ROI crossing counts and UI FPS/latency. No human-labeled ground truth is available for those videos, so the data cannot establish precision, recall, F1, or verified counting accuracy. The required evaluation and ONNX quantization remain separate pending tasks.
