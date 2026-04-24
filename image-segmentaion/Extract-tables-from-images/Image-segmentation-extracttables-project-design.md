# Project Design: Smart Scraper – Intelligent Document Layout Segmenter

**Version:** 1.0  
**Status:** Draft  
**Author:** Srimugunthan  
**Domain:** Computer Vision · Document AI · Privacy Engineering · Edge Inference

---

## 1. Overview

**Smart Scraper** is a local, CPU-friendly document image segmentation pipeline that transforms unstructured document scans — bank statements, invoices, research papers, KYC forms — into structured, machine-readable zones. It identifies and masks functional regions (Headers, Paragraphs, Tables, Figures, Signatures, Profile Pictures) at the pixel level, extracts tabular data into dataframes, and applies privacy-compliant redaction to sensitive areas before any downstream processing.

The project is designed to run entirely on a laptop without a GPU, using lightweight segmentation models (MobileSAM or a fine-tuned U-Net), and is positioned as a production-grade document intelligence pipeline relevant to financial services document processing workflows.

---

## 2. Goals and Non-Goals

### Goals

- Segment document images into labeled functional zones using instance segmentation.
- Extract tabular regions and convert them to structured Pandas dataframes via Camelot or pdfplumber.
- Apply pixel-level privacy redaction (Gaussian blur) to detected Signature and Profile Picture regions.
- Benchmark inference latency across model variants (MobileSAM vs. U-Net) on CPU.
- Evaluate segmentation quality using mIoU against ground-truth masks.
- Support both scanned image input (PNG/JPEG) and PDF-to-image conversion.

### Non-Goals

- Full OCR pipeline (this project produces segments as inputs to OCR, not OCR output itself).
- Cloud deployment or multi-GPU inference.
- Real-time video-feed processing (static document images only).
- Training from scratch on large datasets (fine-tuning only, or zero-shot with SAM).

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Laptop (Edge Device)                            │
│                                                                          │
│  ┌──────────────────┐    ┌─────────────────────┐                         │
│  │  Document Input  │───▶│  Preprocessing      │                         │
│  │  (PDF / Image)   │    │  (pdf2image / PIL)  │                         │
│  └──────────────────┘    └──────────┬──────────┘                         │
│                                     │                                    │
│                          ┌──────────▼──────────┐                         │
│                          │  Segmentation Engine │                         │
│                          │  (MobileSAM / U-Net) │                         │
│                          └──────────┬──────────┘                         │
│                                     │                                    │
│                     ┌───────────────┼────────────────┐                   │
│                     │               │                │                   │
│          ┌──────────▼──────┐  ┌─────▼──────┐  ┌─────▼────────────────┐  │
│          │  Mask Classifier │  │ CRF Refiner│  │  Privacy Redactor    │  │
│          │  (zone labeling) │  │ (edge smth)│  │  (blur sig / photo)  │  │
│          └──────────┬──────┘  └─────┬──────┘  └─────┬────────────────┘  │
│                     │               │                │                   │
│                     └───────────────┴────────────────┘                   │
│                                     │                                    │
│                          ┌──────────▼──────────┐                         │
│                          │  Structured Extractor│                         │
│                          │  (Camelot / pandas)  │                         │
│                          └──────────┬──────────┘                         │
│                                     │                                    │
│               ┌─────────────────────┴──────────────────────┐             │
│               │                                            │             │
│    ┌──────────▼──────────┐                    ┌────────────▼──────────┐  │
│    │  Visualization Layer │                    │  Output Store         │  │
│    │  (OpenCV overlays)   │                    │  (JSON metadata +     │  │
│    └──────────────────────┘                    │   CSV tables + masks) │  │
│                                                └───────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Segmentation Model Selection

### 4.1 Model Options and Trade-offs

| Model | Type | Params | CPU FPS (est.) | Notes |
|---|---|---|---|---|
| **MobileSAM** | Instance (prompt-based) | ~5M | 3–8 FPS | Zero-shot; no training needed |
| **U-Net (ResNet18 backbone)** | Semantic | ~15M | 5–10 FPS | Requires fine-tuning on PubLayNet |
| **Mask R-CNN (ResNet50-FPN)** | Instance | ~44M | 1–3 FPS | Heaviest; best accuracy |
| **DocSegTr / DiT** | Transformer-based | ~28M | 2–4 FPS | State-of-art but slower on CPU |

**Recommended default:** MobileSAM for zero-shot exploration + fine-tuned U-Net for production accuracy. Both are benchmarked as part of the project.

### 4.2 Instance vs. Semantic Segmentation Distinction

- **Semantic segmentation** answers: "Which pixels are tables?" — useful for counting area.
- **Instance segmentation** answers: "Where exactly is Table 1 and where is Table 2?" — required for downstream extraction because each table is cropped and processed independently.

MobileSAM produces instance masks (one mask per detected region). U-Net in its default form produces semantic masks, but can be adapted with connected-component labeling to generate instance IDs post-hoc.

---

## 5. Component Design

### 5.1 Document Preprocessor (`preprocessor.py`)

Handles ingestion of both PDFs and raster images.

- **PDF input:** Convert pages to 300 DPI PNG using `pdf2image` (wraps Poppler).
- **Image input:** Load with PIL, convert to RGB, resize to a model-compatible resolution (e.g., 1024×1024 for SAM, 512×512 for U-Net).
- **Contrast normalization:** Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) via OpenCV to improve segmentation of low-contrast scanned documents.
- **Output:** Normalized NumPy array (H×W×3) + original DPI metadata for coordinate remapping.

```python
# Core preprocessing signature
def preprocess(input_path: str, dpi: int = 300) -> tuple[np.ndarray, dict]:
    # Returns: (normalized_image, metadata)
    # metadata: {original_size, dpi, page_count, input_type}
```

---

### 5.2 Segmentation Engine (`segmenter.py`)

Abstracted interface over multiple backends, selectable via `config.yaml`.

**MobileSAM path:**
1. Load `MobileSAM` checkpoint (~40 MB).
2. Generate automatic grid prompts across the document image (SAM's `SamAutomaticMaskGenerator`).
3. Filter masks by area (discard masks < 1% or > 70% of image area to remove noise and background).
4. Return raw mask list with bbox and predicted IoU score per mask.

**U-Net path:**
1. Load fine-tuned U-Net weights from checkpoint.
2. Forward pass on preprocessed image tile(s).
3. Apply argmax across class channels to get per-pixel class label.
4. Use `cv2.connectedComponentsWithStats` to convert semantic map to instance masks.

**Output format (common interface):**

```python
@dataclass
class SegmentMask:
    mask_id: int
    class_label: str         # header | paragraph | table | figure | signature | photo | unknown
    confidence: float
    bbox: tuple              # (x_min, y_min, x_max, y_max) in pixel coords
    pixel_mask: np.ndarray   # binary H×W mask
    area_fraction: float     # fraction of total image area
```

---

### 5.3 Zone Classifier (`classifier.py`)

MobileSAM produces unlabeled masks. A lightweight classifier assigns functional zone labels to each mask.

**Approach — rule-based heuristics (v1, fast, interpretable):**

| Feature | Heuristic |
|---|---|
| Aspect ratio > 5:1 (wide) | → Header or separator |
| Regular grid of lines detected via Hough transform | → Table |
| Area fraction 5–30%, positioned top-center | → Header |
| Dense uniform text (detected via pixel variance) | → Paragraph |
| Contains embedded raster region (low text density) | → Figure |
| Small area, bottom-right quadrant, cursive-like stroke width | → Signature |
| Small area, circular bounding region, face-like structure | → Profile Photo |

**Approach — ML classifier (v2, optional upgrade):**
Train a lightweight CNN (MobileNetV2) on PubLayNet crop-level labels. Input: cropped mask region resized to 224×224. Output: 6-class softmax. This adds ~3 MB to the model footprint.

---

### 5.4 CRF Post-Processor (`crf_refiner.py`)

Conditional Random Fields smooth the jagged edges of raw segmentation masks, particularly at table cell boundaries and figure borders.

**Library:** `pydensecrf` (DenseCRF by Philipp Krähenbühl).

**Configuration:**

```python
crf_params = {
    "gaussian_sxy": 3,      # spatial smoothness kernel
    "bilateral_sxy": 80,    # color-sensitive smoothness
    "bilateral_srgb": 13,   # RGB similarity sensitivity
    "num_iter": 5           # inference iterations (5 is sufficient for documents)
}
```

**When to apply:** CRF is applied only to masks with confidence < 0.7 to avoid degrading high-confidence clean masks. It adds ~50–100 ms per page so it is run selectively.

**Expected impact:** 2–4 point mIoU improvement on table boundaries, based on published results for document segmentation tasks.

---

### 5.5 Privacy Redactor (`redactor.py`)

Applies Gaussian blur to any mask classified as `signature` or `photo`, in-place on the output image.

```python
def redact(image: np.ndarray, masks: list[SegmentMask],
           target_classes: list[str] = ["signature", "photo"],
           kernel_size: int = 61) -> np.ndarray:
    for mask in masks:
        if mask.class_label in target_classes:
            x1, y1, x2, y2 = mask.bbox
            roi = image[y1:y2, x1:x2]
            image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    return image
```

The redactor operates on the output image before visualization and before any crop is passed downstream. This mirrors a production data pipeline where PII redaction is a gate, not an afterthought.

---

### 5.6 Structured Extractor (`extractor.py`)

For each mask classified as `table`:
1. Crop the original (unblurred) image using the bounding box.
2. Save the crop to a temporary file.
3. Pass to `Camelot` (for grid-based tables) or `pdfplumber` (if PDF input is available).
4. Return a `pd.DataFrame` with detected rows and columns.
5. If Camelot fails (e.g., borderless table), fall back to `img2table` which uses OpenCV line detection.

```python
@dataclass
class ExtractionResult:
    mask_id: int
    dataframe: pd.DataFrame
    confidence: float         # Camelot accuracy score
    method: str               # camelot | img2table | fallback_manual
    row_count: int
    col_count: int
```

---

### 5.7 Visualization Layer (`visualizer.py`)

Renders the segmented document with color-coded overlays using OpenCV and Matplotlib.

**Color map:**

| Zone | Color (BGR) |
|---|---|
| Header | Blue `(255, 100, 0)` |
| Paragraph | Green `(0, 200, 0)` |
| Table | Orange `(0, 165, 255)` |
| Figure | Purple `(200, 0, 200)` |
| Signature | Red `(0, 0, 255)` (shown as blurred box) |
| Photo | Red `(0, 0, 220)` (shown as blurred box) |

**Output modes:**
- `overlay`: Transparent color mask overlaid on original document (alpha = 0.4).
- `contour`: Only the boundary contours of each segment drawn on the original.
- `panel`: Side-by-side original vs. segmented image for comparison.

---

### 5.8 Output Store (`output_writer.py`)

Per-document outputs written to `outputs/{doc_name}/`:

```
outputs/
└── bank_statement_jan2025/
    ├── page_01_segmented.png        # Visualized overlay
    ├── page_01_redacted.png         # Privacy-redacted version
    ├── page_01_masks.json           # All SegmentMask metadata as JSON
    ├── table_01.csv                 # Extracted table as CSV
    ├── table_02.csv
    └── benchmark.json               # Inference timing for this document
```

`masks.json` schema:
```json
{
  "document": "bank_statement_jan2025",
  "page": 1,
  "dpi": 300,
  "masks": [
    {
      "mask_id": 3,
      "class_label": "table",
      "confidence": 0.82,
      "bbox": [120, 340, 890, 670],
      "area_fraction": 0.18,
      "extraction_result": "table_01.csv"
    }
  ]
}
```

---

## 6. Evaluation Framework (`evaluate.py`)

### 6.1 mIoU (Mean Intersection over Union)

The primary segmentation quality metric. For each class `c`:

```
IoU_c = (Predicted_c ∩ Ground_Truth_c) / (Predicted_c ∪ Ground_Truth_c)
mIoU  = mean(IoU_c for all c)
```

Ground truth can be sourced from:
- **PubLayNet:** 360,000+ annotated document images with polygon-level annotations (Header, Paragraph, Figure, Table, List). A subset of 500 images is sufficient for local evaluation.
- **Manual annotation:** Use `labelme` to annotate 20–30 of your own PDFs for a domain-specific evaluation.

### 6.2 Table Extraction Accuracy

For tables with known ground truth (e.g., re-typed from source):
- Cell-level precision and recall.
- Camelot's built-in `accuracy` and `whitespace` scores.

### 6.3 Inference Benchmarking

```python
# benchmark.py — measures per-stage latency
stages = {
    "preprocessing": time_preprocessing,
    "segmentation": time_segmentation,
    "crf_refinement": time_crf,
    "extraction": time_extraction,
    "visualization": time_visualization
}
# Results logged to benchmark.json per document
```

**Target benchmarks (single A4 page, laptop CPU):**

| Stage | Target Latency |
|---|---|
| Preprocessing | < 200 ms |
| MobileSAM segmentation | < 3 s |
| U-Net segmentation | < 1 s |
| CRF refinement | < 150 ms |
| Table extraction (Camelot) | < 2 s |
| Full pipeline (MobileSAM) | < 7 s |
| Full pipeline (U-Net) | < 4 s |

---

## 7. Dataset Strategy

### 7.1 PubLayNet (Primary Evaluation Dataset)

- 335,703 document images derived from PubMed Open Access papers.
- 5 classes: Text, Title, List, Table, Figure.
- COCO-format annotations (polygon masks + bounding boxes).
- Download a 2,000-image subset for local development: ~1.2 GB.

### 7.2 Custom Financial Documents (Domain Adaptation)

For financial services relevance, annotate 30–50 documents from:
- Publicly available annual reports (SEC EDGAR).
- Synthetic bank statements generated with `Faker` + `ReportLab`.

This creates a small domain-specific test set and a compelling portfolio narrative: you evaluated on both academic benchmarks and synthetic finserv documents.

### 7.3 Zero-Shot Baseline (MobileSAM, No Training)

Run MobileSAM on PubLayNet images without any fine-tuning. Measure mIoU as the zero-shot baseline. This becomes the comparison anchor for the fine-tuned U-Net.

---

## 8. Project Structure

```
smart-scraper/
├── config.yaml                   # Runtime settings (model, thresholds, paths)
├── main.py                       # CLI entry point
├── preprocessor.py               # PDF/image ingestion and normalization
├── segmenter.py                  # Unified segmentation interface (SAM + U-Net)
├── classifier.py                 # Zone label assignment (rule-based + optional CNN)
├── crf_refiner.py                # DenseCRF mask smoothing
├── redactor.py                   # Signature/photo privacy blurring
├── extractor.py                  # Table-to-dataframe extraction
├── visualizer.py                 # OpenCV overlay rendering
├── output_writer.py              # Structured output (JSON, CSV, PNG)
├── evaluate.py                   # mIoU computation and benchmarking
├── models/
│   ├── mobile_sam.pt             # MobileSAM checkpoint (gitignored, downloaded at runtime)
│   └── unet_publaynet.pth        # Fine-tuned U-Net checkpoint (gitignored)
├── data/
│   ├── publaynet_sample/         # Evaluation subset (gitignored)
│   └── custom_docs/              # Your annotated financial documents
├── outputs/                      # Per-document results (gitignored)
├── notebooks/
│   ├── 01_eda_publaynet.ipynb    # Dataset exploration
│   ├── 02_model_comparison.ipynb # MobileSAM vs U-Net benchmark comparison
│   └── 03_table_extraction.ipynb # Table quality analysis
├── tests/
│   ├── test_segmenter.py         # Unit: mask output schema validation
│   ├── test_redactor.py          # Unit: blur applied only to target classes
│   └── test_extractor.py         # Unit: CSV shape matches ground truth table
├── pyproject.toml                # uv-managed dependencies
└── README.md
```

---

## 9. Setup and Run

```bash
# 1. Create environment
uv venv
source .venv/bin/activate

# 2. Core dependencies
uv pip install segment-anything-fast opencv-python matplotlib \
               pdf2image pillow pydensecrf camelot-py pandas \
               torch torchvision

# 3. Optional: img2table fallback for borderless tables
uv pip install img2table

# 4. Download MobileSAM checkpoint
python -c "from mobile_sam import sam_model_registry; ..."

# 5. Run on a single document
python main.py --input data/custom_docs/statement.pdf \
               --model mobilesam \
               --redact signature photo \
               --output outputs/

# 6. Run evaluation on PubLayNet subset
python evaluate.py --dataset data/publaynet_sample/ \
                   --model unet \
                   --report outputs/eval_report.json

# 7. Benchmark model comparison
python evaluate.py --benchmark --models mobilesam unet \
                   --input data/custom_docs/statement.pdf
```

---

## 10. Configuration (`config.yaml`)

```yaml
input:
  dpi: 300
  supported_formats: [pdf, png, jpg, tiff]

segmentation:
  model: mobilesam               # options: mobilesam | unet | mask_rcnn
  mobilesam:
    checkpoint: models/mobile_sam.pt
    points_per_side: 32          # grid prompt density
    min_mask_area_frac: 0.01
    max_mask_area_frac: 0.70
  unet:
    checkpoint: models/unet_publaynet.pth
    input_size: [512, 512]
    num_classes: 6

classification:
  mode: rules                    # options: rules | cnn
  cnn_checkpoint: models/zone_classifier.pth  # only if mode=cnn

crf:
  enabled: true
  apply_below_confidence: 0.70
  num_iterations: 5

redaction:
  enabled: true
  target_classes: [signature, photo]
  kernel_size: 61

extraction:
  primary: camelot               # options: camelot | img2table
  camelot_flavor: lattice        # lattice (bordered) | stream (borderless)
  fallback: img2table

output:
  save_overlay: true
  save_redacted: true
  save_masks_json: true
  save_extracted_tables: true
  overlay_alpha: 0.4
```

---

## 11. Learning Objectives Mapped to Implementation

| Learning Objective | Where Implemented |
|---|---|
| Instance vs. semantic segmentation | `segmenter.py` — SAM produces instance masks; U-Net produces semantic maps converted via connected components |
| mIoU metric computation | `evaluate.py` — per-class IoU averaged across test set |
| Coordinate mapping pixel → crop | `extractor.py` — bbox pixel coords used to crop table ROI |
| Downstream OCR handoff | `extractor.py` — cropped table passed to Camelot / img2table |
| CRF post-processing | `crf_refiner.py` — pydensecrf on low-confidence masks |
| Privacy redaction logic | `redactor.py` — Gaussian blur on signature / photo masks |
| Inference benchmarking | `evaluate.py --benchmark` — per-stage latency logged to JSON |
| Visualization with masks | `visualizer.py` — OpenCV transparent overlay with class color map |

---

## 12. Senior-Level Complexity Layers

### 12.1 Model Quantization

Export fine-tuned U-Net to ONNX and apply dynamic INT8 quantization via `torch.quantization`:

```python
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Conv2d, torch.nn.Linear}, dtype=torch.qint8
)
```

Expected benefit: ~1.5–2× latency reduction on CPU, < 2 point mIoU drop.

### 12.2 Agentic Extension

Connect the structured extractor output to a local LLM via Ollama:

- Input: Extracted CSV tables + `masks.json` metadata.
- Prompt: "This is a bank statement. Identify any unusual transactions in the table below."
- Output: Natural language anomaly summary.

This directly maps to AML document review use cases — a compelling narrative for finserv interviews.

### 12.3 Confidence-Gated Processing

Implement a quality gate: if the average mask confidence on a document falls below a threshold (e.g., 0.55), the pipeline automatically falls back from MobileSAM to U-Net. This mirrors production ML systems with graceful degradation logic.

---

## 13. Portfolio Narrative

**Document AI meets Financial Services:** The ability to segment bank statements, KYC documents, and loan applications into structured zones is directly applicable to AML workflows, credit underwriting automation, and regulatory compliance reporting. This project demonstrates end-to-end document intelligence, not just model inference.

**Privacy by Design:** The redaction component mirrors GDPR and RBI data minimization requirements. Signatures and photos are blurred before any data leaves the segmentation stage — the same architectural decision that would be made in a production document processing pipeline.

**Rigorous Evaluation:** mIoU on PubLayNet plus a custom finserv test set gives two evaluation anchors — academic benchmark comparability and domain-specific validation. The benchmark comparison (MobileSAM vs. U-Net vs. quantized U-Net) demonstrates awareness of the accuracy-latency trade-off that matters in production.

**Agentic Extension:** The Ollama integration turns a CV pipeline into a Document Intelligence Agent, connecting computer vision output to language model reasoning — exactly the kind of multimodal agentic system being built in enterprise AI today.

---

## 14. Open Questions and Future Work

- **v1.1:** Add `layoutparser` as an alternative detection backend (Detectron2-based, pre-trained on PubLayNet) for direct comparison with MobileSAM.
- **v1.2:** Build a minimal Gradio UI for interactive document upload and segmentation visualization — lowers the barrier for stakeholder demos.
- **v2.0:** Fine-tune a Vision-Language Model (e.g., LLaVA via Ollama) on document Q&A: given a segmented document, answer "What is the closing balance?" by reasoning over the extracted table.
- **Research angle:** Explore adversarial patches that fool the zone classifier into misclassifying a Table as a Paragraph — connects directly to your adversarial ML background and raises interesting questions about document AI robustness in fraud scenarios.
