# Project Design: AI-Powered Exercise Rep Counter and Form Evaluator

**Version:** 1.0  
**Status:** Draft  
**Author:** Srimugunthan  
**Domain:** Computer Vision · Pose Estimation · Temporal State Machines · Edge Inference

---

## 1. Overview

The **AI-Powered Rep Counter and Form Evaluator** is a real-time, webcam-based exercise analysis pipeline that uses lightweight pose estimation to extract joint coordinates, computes anatomically meaningful joint angles using trigonometry, and runs a finite state machine to count exercise repetitions and grade form quality. The system operates entirely offline on a laptop CPU at 30+ FPS with no model training required — all intelligence lives in the heuristic engine layered on top of a pre-trained skeleton tracker.

The project demonstrates temporal reasoning over CV data: not just detecting a pose at a single frame, but understanding how body geometry changes across time — a skill that transfers directly to time-series anomaly detection, signal processing, and sequence modeling in fraud and AML domains.

---

## 2. Goals and Non-Goals

### Goals

- Extract 3D joint landmarks from a live webcam feed in real time using MediaPipe Pose or YOLO11n-pose.
- Smooth noisy coordinate streams via Exponential Moving Average (EMA) to eliminate phantom reps.
- Compute joint angles (knee, hip, elbow) using the law of cosines and classify pose states (UP / TRANSITION / DOWN).
- Count repetitions using a strict UP → DOWN → UP state transition machine.
- Grade form quality per rep using configurable anatomical rules (depth check, back angle, knee alignment).
- Overlay live feedback (angle, rep count, form grade) on the video stream using OpenCV.
- Log per-rep analytics to SQLite for post-session review.
- Benchmark inference latency at 10 FPS vs. 30 FPS to document the latency-accuracy trade-off.

### Non-Goals

- Training a custom pose model (pre-trained models are used as-is).
- Multi-person tracking (single-user, single-webcam setup).
- Exercise classification (the exercise type is selected at launch, not detected automatically — that is a v2 extension).
- Cloud sync or server-side processing of any kind.

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Laptop (Edge Device)                           │
│                                                                         │
│  ┌────────────┐    ┌──────────────────┐    ┌────────────────────────┐   │
│  │  Webcam    │───▶│  Frame Sampler   │───▶│  Pose Estimator        │   │
│  │ (OpenCV)   │    │  (configurable   │    │  (MediaPipe / YOLO11n- │   │
│  └────────────┘    │   FPS gate)      │    │   pose)                │   │
│                    └──────────────────┘    └───────────┬────────────┘   │
│                                                        │                │
│                                            ┌───────────▼────────────┐   │
│                                            │  Landmark Smoother     │   │
│                                            │  (EMA per joint coord) │   │
│                                            └───────────┬────────────┘   │
│                                                        │                │
│                                            ┌───────────▼────────────┐   │
│                                            │  Angle Calculator      │   │
│                                            │  (Law of Cosines)      │   │
│                                            └───────────┬────────────┘   │
│                                                        │                │
│                                            ┌───────────▼────────────┐   │
│                                            │  State Machine Engine  │   │
│                                            │  (UP / DOWN / TRANS)   │   │
│                                            └──────┬────────┬────────┘   │
│                                                   │        │            │
│                                     ┌─────────────▼──┐  ┌─▼──────────┐ │
│                                     │  Rep Counter   │  │  Form      │ │
│                                     │  (transition   │  │  Grader    │ │
│                                     │   logic)       │  │  (rules)   │ │
│                                     └─────────┬──────┘  └─────┬──────┘ │
│                                               │                │        │
│                                     ┌─────────▼────────────────▼──────┐ │
│                                     │  UI Overlay (cv2.putText)       │ │
│                                     │  + Analytics Logger (SQLite)    │ │
│                                     └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Pose Estimation Backend

### 4.1 Model Options

| Model | Landmarks | CPU FPS (est.) | Notes |
|---|---|---|---|
| **MediaPipe Pose (Lite)** | 33 3D landmarks | 30–60 FPS | Best for laptop; built-in smoothing |
| **MediaPipe Pose (Full)** | 33 3D landmarks | 20–30 FPS | Slightly more accurate torso tracking |
| **YOLO11n-pose** | 17 COCO keypoints | 25–40 FPS | Easier ONNX/OpenVINO export path |

**Recommended default:** MediaPipe Pose Lite. It ships with built-in temporal filtering that complements the EMA layer, and has zero training overhead.

### 4.2 Landmark Index Reference

MediaPipe landmark indices used by this project:

| Joint | Landmark ID | Used In |
|---|---|---|
| Left Hip | 23 | Squat depth, back angle |
| Right Hip | 24 | Squat depth, back angle |
| Left Knee | 25 | Squat angle, knee cave check |
| Right Knee | 26 | Squat angle, knee cave check |
| Left Ankle | 27 | Squat angle baseline |
| Right Ankle | 28 | Squat angle baseline |
| Left Shoulder | 11 | Push-up back angle |
| Right Shoulder | 12 | Push-up back angle |
| Left Elbow | 13 | Push-up / curl angle |
| Right Elbow | 14 | Push-up / curl angle |
| Left Wrist | 15 | Push-up angle |
| Right Wrist | 16 | Push-up angle |

YOLO11n-pose uses COCO 17-keypoint format; a mapping layer (`landmark_adapter.py`) translates between the two so the downstream components are model-agnostic.

---

## 5. Component Design

### 5.1 Frame Sampler (`sampler.py`)

Controls the processing rate independently of the camera's native FPS.

- **Camera capture:** `cv2.VideoCapture(0)` reads at the camera's native rate (typically 30 FPS).
- **Processing gate:** Only every `N`-th frame is forwarded to the pose estimator, where `N = camera_fps / target_fps`.
- **Display rate:** The raw camera frame (with overlaid graphics) is always shown at native FPS for smooth UI. Only the pose estimation and state machine run at the reduced rate.

**Rationale for configurable FPS:**

| Processing FPS | CPU Load | Rep Count Accuracy | Battery Impact |
|---|---|---|---|
| 30 FPS | High | Maximum | High |
| 15 FPS | Medium | Good (exercises < 2 reps/sec) | Medium |
| 10 FPS | Low | Acceptable (exercises < 1.5 reps/sec) | Low |

A squat at normal tempo takes ~2 seconds per rep, so 10 FPS captures ~20 frames per rep — more than sufficient for state transitions. The benchmark comparison between 10 and 30 FPS is a documented deliverable.

---

### 5.2 Landmark Smoother (`smoother.py`)

Raw landmark coordinates from MediaPipe jitter frame-to-frame due to camera noise, motion blur, and model uncertainty. Without smoothing, small jitters around the UP/DOWN angle thresholds cause phantom rep counts.

**Exponential Moving Average (EMA):**

```
smoothed[t] = α × raw[t] + (1 - α) × smoothed[t-1]
```

Applied independently to `x`, `y`, and `z` for each landmark. `α` is configurable (default: 0.4).

**Effect of α:**

| α | Lag | Noise Rejection |
|---|---|---|
| 0.8 | Low (responsive) | Low |
| 0.4 | Medium | Medium (recommended) |
| 0.2 | High (sluggish) | High |

**Additional guard: visibility threshold.** MediaPipe provides a `visibility` score per landmark (0–1). Landmarks with visibility < 0.5 are excluded from angle computation for that frame; the previous valid angle is carried forward instead.

---

### 5.3 Angle Calculator (`angle_calculator.py`)

Computes the interior angle at a joint given three landmark coordinates using the law of cosines.

**Formula:** Given points P1 (proximal joint), P2 (target joint, vertex), P3 (distal joint):

```
a = ||P2 - P3||   (distance from vertex to distal)
b = ||P1 - P2||   (distance from proximal to vertex)
c = ||P1 - P3||   (distance from proximal to distal)

θ = arccos((a² + b² - c²) / (2ab))
```

The result `θ` is the interior angle at P2 in degrees.

```python
def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Computes interior angle at p2 (the vertex joint).
    p1, p2, p3 are (x, y) or (x, y, z) coordinate arrays.
    Returns angle in degrees [0, 180].
    """
    a = np.linalg.norm(p2 - p3)
    b = np.linalg.norm(p1 - p2)
    c = np.linalg.norm(p1 - p3)
    cosine = np.clip((a**2 + b**2 - c**2) / (2 * a * b), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))
```

**Angles computed per exercise:**

| Exercise | Primary Angle | Supporting Angles |
|---|---|---|
| Squat | Knee angle (Hip–Knee–Ankle) | Hip flexion, back lean |
| Push-up | Elbow angle (Shoulder–Elbow–Wrist) | Back straightness (Shoulder–Hip–Ankle) |
| Bicep Curl | Elbow angle (Shoulder–Elbow–Wrist) | Shoulder stability (elbow drift) |

---

### 5.4 State Machine Engine (`state_machine.py`)

The core temporal logic component. Implements a finite state machine (FSM) per exercise that tracks position states and triggers rep counts only on valid complete transitions.

**States:**

```
UP ──────────────────────────────▶ TRANSITION_DOWN
(knee > 160°)      (knee between 90° and 160°)
                                          │
                                          ▼
                                        DOWN
                                    (knee < 90°)
                                          │
                                          ▼
                              TRANSITION_UP (knee between 90° and 160°)
                                          │
                                          ▼
                                   UP ← rep counted here
```

**State dataclass:**

```python
@dataclass
class ExerciseState:
    current_state: str          # UP | TRANSITION_DOWN | DOWN | TRANSITION_UP
    rep_count: int
    last_transition_time: float
    min_rep_duration_sec: float = 0.8   # guard: ignore transitions faster than this
```

**Anti-bounce logic:** A rep is only counted if the full DOWN phase lasted at least `min_rep_duration_sec` seconds. This prevents a single oscillation around the DOWN threshold from counting as two reps.

**Configurable thresholds per exercise:**

```yaml
exercises:
  squat:
    primary_joint: knee
    up_threshold_deg: 160
    down_threshold_deg: 90
    min_rep_duration_sec: 0.8
  pushup:
    primary_joint: elbow
    up_threshold_deg: 155
    down_threshold_deg: 90
    min_rep_duration_sec: 0.6
  bicep_curl:
    primary_joint: elbow
    up_threshold_deg: 150
    down_threshold_deg: 50
    min_rep_duration_sec: 0.5
```

---

### 5.5 Form Grader (`form_grader.py`)

Evaluated once per rep at the moment the DOWN state is reached — the point of maximum effort where form breaks most commonly occur.

**Squat form rules:**

| Rule | Check | Failure Label |
|---|---|---|
| Depth check | Hip y-coordinate must be ≤ knee y-coordinate at bottom | "LOW DEPTH" |
| Knee cave (valgus) | Left knee x must not drift medially past left ankle x by > 10% frame width | "KNEE CAVE" |
| Back angle | Shoulder–Hip vector angle from vertical must be < 45° | "FORWARD LEAN" |
| Symmetry | Left knee angle vs. right knee angle difference < 15° | "UNEVEN SQUAT" |

**Push-up form rules:**

| Rule | Check | Failure Label |
|---|---|---|
| Back straightness | Shoulder–Hip–Ankle angle must be within 10° of 180° | "SAGGING HIPS" |
| Full lockout | Elbow angle at top must exceed 150° | "INCOMPLETE LOCKOUT" |
| Head position | Nose y-coordinate must stay roughly aligned with shoulder y | "HEAD DROP" |

**Form grade per rep:**

```
PASS   → 0 failures
WARN   → 1 failure (non-critical)
FAIL   → 2+ failures or any critical failure (e.g., KNEE CAVE)
```

A per-rep `FormResult` is stored with the list of triggered rules, enabling post-session form trend analysis.

---

### 5.6 UI Overlay (`overlay.py`)

Renders live feedback on the video frame using OpenCV drawing primitives.

**Overlay elements:**

- **Skeleton:** Draw MediaPipe pose connections on the frame (colored lines between landmark pairs).
- **Angle badge:** Small circle at the knee/elbow joint displaying the current angle in degrees. Color: green if within good range, yellow in transition, red at extremes.
- **Rep counter:** Large text top-left: `REPS: 7`.
- **State indicator:** Current FSM state displayed as a colored bar: UP (blue), DOWN (orange), TRANSITION (gray).
- **Form grade:** Last rep's grade displayed top-right: `FORM: PASS ✓` or `FORM: FAIL ✗ [KNEE CAVE]`.
- **FPS meter:** Bottom-right: current processing FPS vs. display FPS.

**Color scheme:**

| Element | Color (BGR) |
|---|---|
| Skeleton lines | White `(255, 255, 255)` |
| Angle: good range | Green `(0, 200, 0)` |
| Angle: warning range | Yellow `(0, 200, 255)` |
| Angle: critical | Red `(0, 0, 255)` |
| Form PASS | Green |
| Form WARN | Yellow |
| Form FAIL | Red |

---

### 5.7 Analytics Logger (`logger.py`)

Logs per-rep data to a local SQLite database for post-session review and trend analysis.

**Schema:**

```sql
CREATE TABLE sessions (
    session_id   TEXT PRIMARY KEY,
    exercise     TEXT NOT NULL,
    start_time   DATETIME NOT NULL,
    end_time     DATETIME,
    target_fps   INTEGER,
    pose_model   TEXT
);

CREATE TABLE reps (
    rep_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    rep_number      INTEGER NOT NULL,
    timestamp       DATETIME NOT NULL,
    duration_sec    REAL NOT NULL,
    min_angle_deg   REAL NOT NULL,      -- deepest point reached
    max_angle_deg   REAL NOT NULL,      -- top of movement
    form_grade      TEXT NOT NULL,      -- PASS | WARN | FAIL
    form_failures   TEXT,               -- JSON array of triggered rule names
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

**Derived analytics (available via `analytics.py`):**

```sql
-- Average form grade trend over last 7 sessions
SELECT session_id, start_time,
       SUM(CASE WHEN form_grade = 'PASS' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS pass_rate
FROM reps GROUP BY session_id ORDER BY start_time DESC LIMIT 7;

-- Rep cadence: average duration per rep per session
SELECT session_id, AVG(duration_sec) AS avg_rep_duration
FROM reps GROUP BY session_id;

-- Worst form failures by frequency
SELECT json_each.value AS failure, COUNT(*) AS occurrences
FROM reps, json_each(form_failures)
WHERE form_failures IS NOT NULL
GROUP BY failure ORDER BY occurrences DESC;
```

---

## 6. Performance Engineering

### 6.1 Latency-Accuracy Trade-off Benchmark

A documented benchmark comparing 10 FPS vs. 30 FPS processing rates across three dimensions:

| Metric | 10 FPS | 30 FPS |
|---|---|---|
| CPU utilization | ~15% | ~45% |
| Rep count accuracy (vs. manual count) | ≥ 98% | ≥ 99.5% |
| Phantom rep rate | < 1% | < 0.5% |
| Battery draw (est., 1-hour session) | Low | High |
| End-to-end latency (capture → display) | < 35 ms | < 35 ms |

**Conclusion baked into the design:** 10 FPS is the recommended default for battery-constrained laptop use. 30 FPS is available via config for users with a power adapter or a dedicated GPU.

### 6.2 Profiling Breakdown (per frame at 30 FPS)

| Stage | Target Latency |
|---|---|
| Frame capture | < 5 ms |
| Pose estimation (MediaPipe Lite) | < 15 ms |
| EMA smoothing (33 landmarks) | < 1 ms |
| Angle computation (3–6 angles) | < 1 ms |
| State machine update | < 1 ms |
| Form grading | < 2 ms |
| Overlay rendering | < 5 ms |
| **Total pipeline** | **< 30 ms** |

All stages except pose estimation are sub-millisecond and do not require optimization. Pose estimation is the only bottleneck and is addressed by the FPS gate.

---

## 7. Exercise Configuration System

The heuristic engine is fully data-driven via `config.yaml`. Adding a new exercise requires only a new YAML block — no code changes.

**Example: extending to deadlift**

```yaml
exercises:
  deadlift:
    primary_joint: hip
    landmarks:
      primary: [shoulder, hip, knee]   # hip angle
      secondary: [hip, knee, ankle]    # knee angle
    up_threshold_deg: 170
    down_threshold_deg: 90
    min_rep_duration_sec: 1.2
    form_rules:
      - name: back_angle
        check: shoulder_hip_vertical_angle
        threshold_deg: 30
        failure_label: "ROUNDED BACK"
      - name: bar_path
        check: wrist_x_drift_from_ankle_x
        threshold_px_fraction: 0.05
        failure_label: "BAR DRIFT"
```

This design makes the system extensible to any bilateral joint-angle exercise without modifying the core state machine or angle calculator.

---

## 8. Project Structure

```
rep-counter/
├── config.yaml                  # Exercise definitions, thresholds, FPS, model selection
├── main.py                      # CLI entry point and main loop
├── sampler.py                   # FPS gate and frame capture
├── pose_estimator.py            # MediaPipe / YOLO11n-pose wrapper
├── landmark_adapter.py          # Normalizes landmark format across model backends
├── smoother.py                  # EMA coordinate smoothing
├── angle_calculator.py          # Law of cosines angle computation
├── state_machine.py             # FSM: UP / TRANSITION / DOWN states + rep counting
├── form_grader.py               # Per-rep anatomical rule evaluation
├── overlay.py                   # OpenCV UI rendering
├── logger.py                    # SQLite session and rep logging
├── analytics.py                 # Post-session query helpers
├── benchmark.py                 # FPS vs. accuracy benchmark runner
├── data/
│   └── sessions.db              # SQLite database (gitignored)
├── notebooks/
│   └── session_analytics.ipynb  # Rep quality trend visualization
├── tests/
│   ├── test_angle_calculator.py # Unit: known triangle → expected angle
│   ├── test_state_machine.py    # Unit: angle sequence → correct rep count
│   └── test_form_grader.py      # Unit: landmark positions → expected form grade
├── pyproject.toml               # uv-managed dependencies
└── README.md
```

---

## 9. Setup and Run

```bash
# 1. Create environment
uv venv
source .venv/bin/activate

# 2. Install dependencies
uv pip install mediapipe opencv-python numpy

# 3. Optional: YOLO11n-pose backend
uv pip install ultralytics

# 4. Run squat counter (default)
python main.py --exercise squat --fps 10 --model mediapipe

# 5. Run push-up counter at full FPS
python main.py --exercise pushup --fps 30 --model mediapipe

# 6. Run latency benchmark
python benchmark.py --exercise squat --fps 10 30 --duration 60

# 7. Launch post-session analytics notebook
jupyter lab notebooks/session_analytics.ipynb
```

---

## 10. Configuration (`config.yaml`)

```yaml
capture:
  device_id: 0
  native_fps: 30

pose:
  model: mediapipe               # options: mediapipe | yolo11n_pose
  mediapipe:
    complexity: 0                # 0=Lite, 1=Full, 2=Heavy
    min_detection_confidence: 0.7
    min_tracking_confidence: 0.5
  visibility_threshold: 0.5     # landmarks below this are skipped

smoothing:
  alpha: 0.4                    # EMA smoothing factor (0=max smooth, 1=no smooth)

processing_fps: 10              # target FPS for pose estimation and state machine

exercise: squat                 # active exercise (overridden by CLI --exercise flag)

exercises:
  squat:
    primary_joint: knee
    up_threshold_deg: 160
    down_threshold_deg: 90
    min_rep_duration_sec: 0.8
    form_rules:
      - name: depth_check
        failure_label: "LOW DEPTH"
      - name: knee_cave
        threshold_x_fraction: 0.10
        failure_label: "KNEE CAVE"
      - name: back_angle
        threshold_deg: 45
        failure_label: "FORWARD LEAN"
  pushup:
    primary_joint: elbow
    up_threshold_deg: 155
    down_threshold_deg: 90
    min_rep_duration_sec: 0.6
    form_rules:
      - name: back_straightness
        threshold_deg: 10
        failure_label: "SAGGING HIPS"
      - name: full_lockout
        threshold_deg: 150
        failure_label: "INCOMPLETE LOCKOUT"

logging:
  db_path: data/sessions.db
```

---

## 11. Learning Objectives Mapped to Implementation

| Learning Objective | Where Implemented |
|---|---|
| Temporal reasoning over CV data | `state_machine.py` — rep detection requires tracking state across frames, not just detecting pose in a single frame |
| Coordinate extraction from pose model | `pose_estimator.py` + `landmark_adapter.py` |
| EMA smoothing of noisy time-series | `smoother.py` — directly analogous to signal smoothing in fraud detection |
| Trigonometric angle computation | `angle_calculator.py` — law of cosines on joint triplets |
| Finite state machine design | `state_machine.py` — UP / TRANSITION / DOWN with anti-bounce guard |
| Rule-based heuristic engine | `form_grader.py` — anatomical constraints as executable business logic |
| Latency vs. accuracy trade-off | `benchmark.py` — documented FPS comparison with accuracy measurement |
| Time-series analytics | `analytics.py` + `session_analytics.ipynb` — rep quality trends over sessions |

---

## 12. Transferable Skills Framing

This project is deliberately framed for the finserv domain because the core engineering patterns map directly to production ML work in fraud and AML.

**EMA smoothing → Signal denoising in transaction streams.** The same technique used to prevent phantom reps from jittery coordinates applies to smoothing noisy transaction velocity signals in fraud detection — a high-frequency time series where single-point spikes cause false positives.

**Finite state machine → Sequence-based fraud rules.** The UP → DOWN → UP transition logic is structurally identical to multi-step fraud pattern detection: e.g., Account Opened → Small Test Transaction → Large Withdrawal. Both are FSMs over a time-ordered event sequence.

**Form grading rules → Business rule engines.** The form checker is a lightweight rule engine that evaluates a set of named conditions against structured data and returns a grade with explanations. This is the same pattern as a credit decisioning engine or an AML alert enrichment layer.

**Per-rep analytics → Per-transaction analytics.** The SQLite schema (session → reps) maps cleanly to (account → transactions), and the derived queries (pass rate trends, failure frequency) are the same aggregations used in fraud investigation dashboards.

---

## 13. Portfolio Narrative

**Temporal CV over raw coordinate reasoning:** This project shows understanding of the difference between pose detection (what frame-level models do) and pose understanding (what the heuristic engine does). The state machine is the deliverable, not the MediaPipe wrapper.

**Privacy by Design:** The entire system runs offline with no external API calls. No video frames leave the laptop. The only persistent data is the structured metadata in SQLite — angles, grades, timestamps. This is the same architectural principle as the Privacy-First Home Office Tracker, reinforcing a consistent design philosophy across the portfolio.

**Performance engineering as a first-class concern:** The FPS benchmark with documented accuracy trade-offs demonstrates awareness of the latency-resource-quality triangle that governs every production ML deployment decision.

**Extensible by design:** The YAML-driven exercise configuration means the system is not a one-trick squat counter — it is a generic joint-angle state machine. The README can frame it as a foundation for any bilateral resistance exercise, making it relevant to fitness tech, physical rehabilitation monitoring, and sports performance analytics.

---

## 14. Open Questions and Future Work

- **v1.1:** Add audio feedback (text-to-speech via `pyttsx3`) for real-time form cues without looking at the screen — "KNEE CAVE" spoken aloud mid-rep is more useful than text on screen.
- **v1.2:** Export rep-level data to a Pandas DataFrame and generate a session PDF report (form score trend, deepest angle per rep, cadence chart).
- **v2.0:** Add exercise auto-detection: instead of specifying the exercise at launch, classify the exercise from the first 5 seconds of movement using a lightweight 1D CNN or rule-based motion classifier over joint angle time series.
- **v2.1:** Connect session analytics to a local LLM via Ollama — "Summarize my squat form over the last 10 sessions and identify the most common failure pattern."
- **Research angle:** Treat the per-frame joint angle time series as a multivariate signal and apply DTW (Dynamic Time Warping) to compare a user's rep against a reference "perfect rep" template. This connects to time-series similarity search — a technique used in market microstructure analysis and behavioral biometrics in finserv.
