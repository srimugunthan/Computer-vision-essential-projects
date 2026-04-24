# Project Design: Privacy-First Home Office Tracker

**Version:** 1.0  
**Status:** Draft  
**Author:** Srimugunthan  
**Domain:** Edge AI · Privacy Engineering · Optimized Inference

---

## 1. Overview

The **Privacy-First Home Office Tracker** is a real-time, on-device video analytics pipeline that detects people and sensitive objects (phones, documents) via a webcam feed and automatically applies Gaussian blurring to detected regions before any data is logged or stored. The system is designed to run entirely on a standard laptop CPU with no GPU dependency, making it a portable demonstration of edge-native AI with privacy-by-design principles.

The project serves a dual purpose: it is a working portfolio artifact demonstrating skills in Edge AI, computer vision inference pipelines, and agentic analytics extensions — and it is a productivity analytics tool that can be used daily.

---

## 2. Goals and Non-Goals

### Goals

- Run real-time object detection at 20–30 FPS on a standard Intel/AMD laptop CPU.
- Apply bounding-box-level Gaussian blur to detected persons and sensitive items before any frame is persisted.
- Log anonymized detection events (timestamps, class labels, counts) to a local SQLite database.
- Support optional model quantization (INT8 via OpenVINO) for performance uplift.
- Support optional agentic extension: querying detection logs via a local LLM (Ollama).

### Non-Goals

- Cloud upload of any raw video frames.
- Face recognition or identity tracking.
- Multi-camera support (out of scope for v1).
- Real-time streaming to remote dashboards.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Laptop (Edge Device)                       │
│                                                                     │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────────┐  │
│  │  Webcam    │───▶│  Frame Capture  │───▶│  YOLO11n Inference   │  │
│  │  (OpenCV)  │    │  (cv2.VideoCapture)  │  (Ultralytics / ONNX │  │
│  └────────────┘    └─────────────────┘    │   / OpenVINO)        │  │
│                                           └──────────┬───────────┘  │
│                                                      │              │
│                                           ┌──────────▼───────────┐  │
│                                           │  Blur Post-Processor │  │
│                                           │  (cv2.GaussianBlur   │  │
│                                           │   on ROI per bbox)   │  │
│                                           └──────────┬───────────┘  │
│                                                      │              │
│                          ┌───────────────────────────┤              │
│                          │                           │              │
│               ┌──────────▼──────────┐   ┌───────────▼───────────┐  │
│               │  Display Window     │   │  Analytics Logger     │  │
│               │  (blurred preview)  │   │  (SQLite + CSV)       │  │
│               └─────────────────────┘   └───────────┬───────────┘  │
│                                                      │              │
│                                         ┌────────────▼───────────┐  │
│                                         │  Agentic Query Layer   │  │
│                                         │  (Ollama / local LLM)  │  │
│                                         └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 Frame Capture (`capture.py`)

- **Library:** `cv2.VideoCapture(0)` — reads from the default webcam.
- **Resolution:** 640×480 at 30 FPS (configurable via `config.yaml`).
- **Output:** Raw BGR frame as a NumPy array, passed to the inference engine.

**Key design decision:** Frames are never written to disk in their raw form. The only thing persisted is the blurred frame (optional) and the detection metadata.

---

### 4.2 Inference Engine (`detector.py`)

- **Model:** YOLO11n (Nano) — ~2.6M parameters, ~6 MB on disk.
- **Runtime options (selectable via config):**

| Runtime | Use Case | Notes |
|---|---|---|
| `ultralytics` (PyTorch) | Development / baseline | Easiest to set up |
| `ONNX Runtime` | Cross-platform speedup | Export once with `model.export(format='onnx')` |
| `OpenVINO` (INT8) | Intel CPU maximum perf | Requires calibration dataset for quantization |

- **Target classes:** `person`, `cell phone`, `book`, `laptop` (COCO class IDs: 0, 67, 73, 63).
- **Confidence threshold:** 0.45 (configurable).
- **Output:** List of bounding boxes `[(x_min, y_min, x_max, y_max, class_id, confidence)]`.

**Inference flow:**

```
frame (H×W×3)
    → model(frame)
    → results.boxes  [xyxy, cls, conf]
    → filter by target classes and confidence threshold
    → return detection list
```

---

### 4.3 Blur Post-Processor (`blurrer.py`)

This is the privacy-enforcement layer. For every bounding box returned by the detector:

1. Clip the box coordinates to the frame boundary (guard against out-of-frame detections).
2. Extract the ROI: `roi = frame[y_min:y_max, x_min:x_max]`.
3. Apply `cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)` where `blur_kernel` is configurable (default: 51).
4. Paste the blurred ROI back: `frame[y_min:y_max, x_min:x_max] = blurred_roi`.
5. Optionally draw a thin bounding box outline and class label on the blurred region for debug mode.

**Blur kernel sizing guidance:**

| Kernel Size | Effect |
|---|---|
| 21 | Light blur (faces partly visible) |
| 51 | Strong blur (recommended default) |
| 99 | Heavy pixelation (maximum privacy) |

---

### 4.4 Analytics Logger (`logger.py`)

Detection events are written to a local SQLite database (`tracker.db`) with the following schema:

```sql
CREATE TABLE detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   DATETIME NOT NULL,
    class_name  TEXT NOT NULL,
    confidence  REAL NOT NULL,
    x_min       INTEGER,
    y_min       INTEGER,
    x_max       INTEGER,
    y_max       INTEGER,
    session_id  TEXT NOT NULL
);

CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    start_time  DATETIME NOT NULL,
    end_time    DATETIME,
    notes       TEXT
);
```

**Derived analytics queries (pre-built views):**

```sql
-- Desk occupancy: minutes per hour with a person detected
SELECT strftime('%Y-%m-%d %H', timestamp) AS hour,
       COUNT(DISTINCT strftime('%Y-%m-%d %H:%M', timestamp)) AS occupied_minutes
FROM detections
WHERE class_name = 'person'
GROUP BY hour;

-- Phone pickup frequency: distinct pickup events (gap > 60s = new event)
SELECT COUNT(*) AS pickup_count, DATE(timestamp) AS day
FROM (
    SELECT timestamp,
           LAG(timestamp) OVER (ORDER BY timestamp) AS prev_ts
    FROM detections WHERE class_name = 'cell phone'
)
WHERE (julianday(timestamp) - julianday(prev_ts)) * 86400 > 60
   OR prev_ts IS NULL
GROUP BY day;
```

A companion `export_csv.py` script dumps the last N hours to a flat CSV for notebook analysis.

---

### 4.5 Agentic Query Layer (`agent.py`) — Extension

An optional module that wraps the SQLite analytics in a tool-calling interface consumed by a local LLM via Ollama.

**Architecture:**

```
User question (CLI or simple web UI)
    → agent.py builds context summary from tracker.db
    → sends to Ollama (e.g., llama3.2 or mistral)
    → LLM responds with natural language summary
```

**Example prompts:**
- "Summarize my desk activity for the last 3 hours."
- "How many times did I pick up my phone today?"
- "What was my longest uninterrupted focus session this week?"

**Implementation note:** The agent does not send raw video or bounding box pixel data to the LLM — only the structured text summary of detection events. This preserves privacy even at the agentic layer.

---

## 5. Data Flow (Privacy Boundary)

```
Raw Frame
    │
    ▼
YOLO11n Inference        ← model weights never leave device
    │
    ▼
Bounding Boxes           ← coordinates only, no pixel content
    │
    ├──▶ Blur ROIs in-place on the frame
    │         │
    │         ▼
    │    Blurred Frame   ← only this is displayed; raw frame is discarded
    │
    └──▶ Log metadata to SQLite
              │
              ▼
         (class, timestamp, bbox, confidence)  ← no image stored
```

**Privacy guarantee:** No raw frame is ever written to disk. The SQLite database contains only structured metadata. The blur is applied before the frame reaches the display buffer.

---

## 6. Performance Targets

| Metric | Target | Measurement |
|---|---|---|
| Inference FPS (PyTorch) | ≥ 20 FPS | Averaged over 100 frames |
| Inference FPS (OpenVINO INT8) | ≥ 40 FPS | Averaged over 100 frames |
| End-to-end latency (capture → display) | < 50 ms | p95 |
| SQLite write throughput | ≥ 30 inserts/sec | Not a bottleneck |
| Memory footprint | < 500 MB RSS | Steady state |

Profiling will be done with a simple decorator-based timer and logged to `perf.log` for baseline vs. quantized comparison.

---

## 7. Model Quantization Plan (OpenVINO INT8)

Quantization is a first-class feature, not an afterthought, because comparing FP32 vs INT8 performance is itself a portfolio demonstration.

**Steps:**

1. Export YOLO11n to ONNX: `model.export(format='onnx', imgsz=640)`.
2. Convert ONNX to OpenVINO IR: `mo --input_model model.onnx`.
3. Quantize to INT8 using NNCF with a 200-frame calibration dataset: `nncf.quantize(model, calibration_dataset)`.
4. Benchmark: use `benchmark_app` from OpenVINO toolkit to measure FPS before and after.
5. Validate accuracy: run both models on 50 held-out frames, compare mAP@0.5 to confirm < 2% degradation.

**Expected outcome:** ~1.8–2.2× FPS uplift on a modern Intel Core CPU (8th gen or later).

---

## 8. Project Structure

```
privacy-edge-monitor/
├── config.yaml                  # Runtime settings (model, thresholds, blur kernel, db path)
├── main.py                      # Entry point — orchestrates the pipeline loop
├── capture.py                   # Webcam frame capture
├── detector.py                  # YOLO11n inference wrapper
├── blurrer.py                   # Gaussian blur post-processor
├── logger.py                    # SQLite detection logger
├── agent.py                     # (Optional) Ollama-based agentic query interface
├── export_csv.py                # Export detections to CSV for notebook analysis
├── quantize/
│   ├── export_onnx.py           # YOLO → ONNX export
│   ├── convert_openvino.sh      # ONNX → OpenVINO IR conversion
│   └── quantize_int8.py         # NNCF INT8 quantization script
├── notebooks/
│   └── analytics_eda.ipynb      # Occupancy and productivity EDA
├── models/
│   ├── yolo11n.pt               # Downloaded at runtime (gitignored)
│   └── yolo11n_int8/            # OpenVINO INT8 model (generated)
├── data/
│   └── tracker.db               # SQLite database (gitignored)
├── tests/
│   └── test_blurrer.py          # Unit test: blur applied correctly within bbox
├── pyproject.toml               # uv-managed dependencies
└── README.md
```

---

## 9. Setup and Run

```bash
# 1. Create environment with uv
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install ultralytics opencv-python onnxruntime

# 3. Optional: OpenVINO for quantized inference
uv pip install openvino nncf

# 4. Optional: Agentic layer
uv pip install ollama
ollama pull llama3.2

# 5. Run the tracker
python main.py

# 6. Query the agent (after collecting data)
python agent.py --query "Summarize my activity for the last 2 hours"
```

---

## 10. Configuration (`config.yaml`)

```yaml
capture:
  device_id: 0
  width: 640
  height: 480
  fps: 30

inference:
  model_path: models/yolo11n.pt
  runtime: ultralytics          # options: ultralytics | onnx | openvino
  confidence_threshold: 0.45
  target_classes:               # COCO class names
    - person
    - cell phone
    - book
    - laptop

blur:
  kernel_size: 51               # must be odd
  debug_overlay: false          # draw bbox outlines on blurred regions

logging:
  db_path: data/tracker.db
  log_every_n_frames: 3         # avoid duplicate logging on static scenes

agent:
  enabled: false
  model: llama3.2
  ollama_base_url: http://localhost:11434
```

---

## 11. Key Learning Objectives Mapped to Implementation

| Learning Objective | Where Implemented |
|---|---|
| Live video inference pipeline | `main.py` + `capture.py` + `detector.py` |
| Bounding box extraction and manipulation | `detector.py` → `blurrer.py` |
| Gaussian blur on ROI (Street View-style) | `blurrer.py` → `cv2.GaussianBlur` |
| Model quantization (FP32 → INT8) | `quantize/` directory |
| Edge deployment with lean environment | `uv` + `pyproject.toml` |
| Structured data logging (DS workload) | `logger.py` + `tracker.db` |
| Productivity analytics (SQL + pandas) | `notebooks/analytics_eda.ipynb` |
| Agentic LLM integration (extension) | `agent.py` + Ollama |

---

## 12. Portfolio Narrative

This project can be positioned as a demonstration of four distinct competencies:

**Edge AI and Optimized Inference:** YOLO11n selection rationale (parameter count, CPU suitability), ONNX/OpenVINO export pipeline, INT8 quantization with accuracy benchmarking.

**Privacy Engineering:** Privacy-by-design architecture where blurring is applied in the inference loop before any output path — not as a post-hoc filter. This mirrors production privacy controls in fintech data pipelines.

**Data Science:** The SQLite logging + EDA notebook transforms a CV demo into a quantitative productivity study. Occupancy curves, phone pickup frequency, and focus session distributions are real deliverables.

**Agentic AI:** The Ollama integration demonstrates RAG-over-structured-data at the edge — a pattern directly transferable to AML transaction summarization or credit risk audit agents.

---

## 13. Open Questions and Future Work

- **v1.1:** Add a simple FastAPI endpoint to expose the blurred stream as an MJPEG feed for viewing from another device on the local network (no cloud required).
- **v1.2:** Explore TFLite or Core ML export for potential mobile deployment (demonstrates cross-runtime portability).
- **v2.0:** Swap detection-only for a Vision-Language Model (e.g., LLaVA via Ollama) to generate natural language descriptions of desk state — "desk is clear," "person is reading," "phone in use."
- **Research angle:** Benchmark adversarial perturbations against YOLO11n (small sticker-style patches) to connect this to your adversarial ML background.
