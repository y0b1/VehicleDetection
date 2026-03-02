# BenchmarkCV — Vehicle Detection Benchmarking App

Compare YOLOv8 and EfficientDet (solo and ensemble) across accuracy and speed metrics on your own video or image data.

## What it does

Runs 4 detection configurations on identical input frames and produces a head-to-head comparison:

| Configuration | Models |
|---|---|
| YOLOv8 | ultralytics YOLOv8n |
| EfficientDet | EfficientDet-D0 (effdet) |
| NMS Ensemble | YOLOv8 + EfficientDet with IoU-NMS fusion |
| WBF Ensemble | YOLOv8 + EfficientDet with Weighted Box Fusion |

**Metrics reported:** Precision · Recall · F1 · mAP@0.5 · mAP@[.5:.95] · Avg Inference (ms) · FPS

---

## Quick Start

### 1 — Backend

```bash
cd benchmark/backend
pip install -r requirements.txt
python app.py
```

Backend runs on **http://localhost:5000**

> **Note:** EfficientDet weights download automatically (~50 MB) on first run via `timm`.
> If `effdet` installation is problematic, swap it with YOLOv9: replace `EfficientDetRunner` calls in `benchmarker.py` with another `YOLOv8Runner(model_size='s')` instance.

### 2 — Frontend

```bash
cd benchmark/frontend
npm install
npm run dev
```

Frontend runs on **http://localhost:5173**

---

## Using the UI

1. **Test immediately (no models needed):** Click **"Load Sample Data"** in the sidebar — the full dashboard renders with mock results.
2. **Real benchmark:** Drag a video (`.mp4`, `.avi`, `.mov`, `.mkv`) or image (`.jpg`, `.png`) onto the upload zone → click **"Run Benchmark"** → watch the 4-config progress tracker.
3. Results appear automatically: **Winners card**, **Metrics table**, **mAP@0.5 bar chart**, **Speed vs Accuracy scatter**, and **Annotated frame previews**.

---

## Project Structure

```
benchmark/
├── backend/
│   ├── app.py                      # Flask API (5 endpoints)
│   ├── models/
│   │   ├── yolov8_runner.py        # YOLOv8 inference wrapper
│   │   ├── efficientdet_runner.py  # EfficientDet inference wrapper
│   │   └── ensemble.py             # NMS + WBF fusion (ensemble_boxes)
│   ├── evaluation/
│   │   ├── metrics.py              # Precision/Recall/F1/mAP calculator
│   │   └── benchmarker.py          # Orchestrates all 4 configs
│   ├── uploads/                    # Uploaded files stored here
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx                 # Root: state, polling, layout
    │   ├── components/             # 8 UI components
    │   └── lib/api.js              # All fetch calls
    ├── package.json
    ├── vite.config.js              # Proxy: /api → localhost:5000
    └── tailwind.config.js
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/upload` | Upload video or image → `{ job_id }` |
| POST | `/api/benchmark/<job_id>` | Start benchmark in background thread |
| GET | `/api/status/<job_id>` | Poll progress `{ progress, current_config, status }` |
| GET | `/api/results/<job_id>` | Full metrics for all 4 configs |
| GET | `/api/sample` | Mock results (no models needed) |
| GET | `/api/preview/<job_id>/<config>` | Annotated frame as base64 image |

---

## Requirements

**Python:** 3.9+
**Node:** 18+

**Key Python packages:** `flask`, `ultralytics`, `effdet`, `timm`, `ensemble-boxes`, `opencv-python`, `torch`

**Key npm packages:** `react`, `recharts`, `lucide-react`, `tailwindcss`
