# Project Design: Smart-Sort AI – Personalized Waste Intelligence System

**Version:** 1.0  
**Status:** Draft  
**Author:** Srimugunthan  
**Domain:** Transfer Learning · Image Classification · Model Optimization · Responsible AI

---

## 1. Overview

**Smart-Sort AI** is a local, laptop-optimized waste classification system that fine-tunes a lightweight convolutional neural network (MobileNetV3-Small or EfficientNet-B0) on the TrashNet dataset to classify household waste images into four actionable categories: **Recyclable**, **Compostable**, **Hazardous**, and **Landfill**. The system includes cost-sensitive evaluation (misclassifying Hazardous as Recyclable is not equivalent to other errors), Grad-CAM visual explanations for trust and interpretability, INT8 quantization for edge-optimized inference, a FastAPI inference endpoint for local service integration, and a Gradio drag-and-drop demo UI.

The project demonstrates the complete MLOps arc — data preparation, transfer learning, evaluation beyond accuracy, model optimization, interpretability, and deployment — on hardware accessible to any individual developer.

---

## 2. Goals and Non-Goals

### Goals

- Fine-tune MobileNetV3-Small and EfficientNet-B0 on the TrashNet dataset mapped to 4 waste categories.
- Evaluate with confusion matrix, per-class F1-score, and a cost-weighted error matrix that penalizes high-severity misclassifications.
- Generate Grad-CAM heatmaps per prediction to visualize which image regions drive classification decisions.
- Quantize the best model to INT8 via PyTorch static quantization or OpenVINO NNCF.
- Benchmark FP32 vs. INT8 on CPU: latency, model size, and accuracy delta.
- Serve predictions via a FastAPI REST endpoint (local).
- Build a Gradio UI supporting drag-and-drop image input and webcam live classification.
- Run all training and inference entirely on a laptop CPU (no GPU required).

### Non-Goals

- Multi-label classification (each item is assigned exactly one category).
- Video-stream waste sorting at conveyor-belt speeds (static image classification only).
- Cloud deployment or containerized serving (local endpoint only for v1).
- Detection of multiple waste items in a single image (single-item classification; detection is a v2 extension).

---

## 3. System Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE: Training Pipeline                         │
│                                                                           │
│  ┌──────────────┐    ┌───────────────────┐    ┌─────────────────────────┐ │
│  │  TrashNet    │───▶│  Data Preprocessor│───▶│  Transfer Learning      │ │
│  │  Dataset     │    │  (augment, split, │    │  (MobileNetV3 /         │ │
│  │  (~2,500 img)│    │   normalize)      │    │   EfficientNet-B0)      │ │
│  └──────────────┘    └───────────────────┘    └───────────┬─────────────┘ │
│                                                           │               │
│                                               ┌───────────▼─────────────┐ │
│                                               │  Evaluation Suite       │ │
│                                               │  (confusion matrix,     │ │
│                                               │   F1, cost matrix,      │ │
│                                               │   Grad-CAM)             │ │
│                                               └───────────┬─────────────┘ │
│                                                           │               │
│                                               ┌───────────▼─────────────┐ │
│                                               │  Quantization Pipeline  │ │
│                                               │  (FP32 → INT8)          │ │
│                                               └───────────┬─────────────┘ │
│                                                           │               │
│                                               ┌───────────▼─────────────┐ │
│                                               │  Model Store            │ │
│                                               │  (checkpoint, ONNX,     │ │
│                                               │   OpenVINO IR, INT8)    │ │
│                                               └─────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│                        ONLINE: Inference Pipeline                         │
│                                                                           │
│  ┌──────────────────┐    ┌────────────────────┐                           │
│  │  Image Input     │───▶│  Preprocessor      │                           │
│  │  (upload/webcam) │    │  (resize, normalize│                           │
│  └──────────────────┘    └─────────┬──────────┘                           │
│                                    │                                      │
│                          ┌─────────▼──────────┐                           │
│                          │  INT8 Model         │                           │
│                          │  (MobileNetV3-Small)│                           │
│                          └─────────┬──────────┘                           │
│                                    │                                      │
│              ┌─────────────────────┼──────────────────────┐               │
│              │                     │                      │               │
│   ┌──────────▼──────┐  ┌───────────▼───────┐  ┌──────────▼────────────┐  │
│   │  Class + Score  │  │  Grad-CAM Heatmap │  │  Disposal Instruction │  │
│   │  (top-1 + top-3)│  │  (overlay on img) │  │  (rule-based lookup)  │  │
│   └──────────┬──────┘  └───────────┬───────┘  └──────────┬────────────┘  │
│              │                     │                      │               │
│              └─────────────────────┴──────────────────────┘               │
│                                    │                                      │
│              ┌─────────────────────┴──────────────────────┐               │
│              │                                            │               │
│   ┌──────────▼──────────────┐           ┌────────────────▼─────────────┐  │
│   │  FastAPI REST Endpoint  │           │  Gradio / Streamlit UI       │  │
│   │  POST /classify         │           │  (drag-drop + webcam)        │  │
│   └─────────────────────────┘           └──────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Dataset: TrashNet Mapping

### 4.1 Original TrashNet Categories (6 classes)

TrashNet ships with six classes: glass, paper, cardboard, plastic, metal, trash. These must be remapped to the four target categories that reflect real-world disposal decisions.

**Remapping:**

| TrashNet Class | Target Category | Rationale |
|---|---|---|
| glass | Recyclable | Accepted by most municipal recycling programs |
| paper | Recyclable | Standard recyclable material |
| cardboard | Recyclable | Standard recyclable material |
| plastic | Recyclable | Most plastics 1–2 are recyclable |
| metal | Recyclable | Aluminum and steel are universally recyclable |
| trash | Landfill | Non-specific waste with no recycling path |

This 6→4 mapping means **Compostable** and **Hazardous** have no TrashNet source images and require augmentation:

- **Compostable:** Scrape or source ~200 images of food waste, fruit peels, coffee grounds, yard waste. Alternatively, use a subset from OpenImages V7 with organic material labels.
- **Hazardous:** Source ~150 images of batteries, paint cans, fluorescent bulbs, aerosol cans. These are underrepresented in public datasets and critical to get right.

### 4.2 Class Distribution After Mapping

| Category | Approx. Images | Source |
|---|---|---|
| Recyclable | ~2,000 | TrashNet (glass + paper + cardboard + plastic + metal) |
| Landfill | ~500 | TrashNet (trash class) |
| Compostable | ~200 | Supplemental scrape / OpenImages |
| Hazardous | ~150 | Supplemental scrape / web |

**Class imbalance is significant** (Recyclable:Hazardous ≈ 13:1) and must be handled explicitly — see Section 5.2.

### 4.3 Train / Validation / Test Split

Stratified split maintaining class proportions:

```
Total: ~2,850 images
Train:      70% (~2,000 images)
Validation: 15% (~425 images)
Test:       15% (~425 images)
```

Test set is held out completely until final evaluation. All hyperparameter decisions are made against the validation set.

---

## 5. Component Design

### 5.1 Data Preprocessor and Augmenter (`data_pipeline.py`)

**Preprocessing (train + val + test):**
- Resize to 224×224 (MobileNetV3 input resolution).
- Normalize with ImageNet mean and std: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

**Augmentation (train only):**

```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
    transforms.RandomGrayscale(p=0.1),      # simulate photos under bad lighting
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

**Rationale:** Waste photos are taken in varied lighting (kitchen, street, office) and at arbitrary orientations. Color jitter and grayscale simulate real-world capture conditions. RandomResizedCrop simulates partial occlusion by bins or hands.

### 5.2 Class Imbalance Handling

Three strategies applied in combination:

**Weighted sampler:** During training, oversample minority classes (Hazardous, Compostable) using `WeightedRandomSampler`. Each sample's weight is `1 / class_count`. This ensures each training batch has roughly balanced class representation.

**Class-weighted loss:** Use `CrossEntropyLoss(weight=class_weights)` where `class_weights[c] = total_samples / (n_classes × class_count[c])`. This penalizes errors on minority classes more heavily.

**Cost-sensitive weight amplification for Hazardous:** Beyond standard weighting, Hazardous receives an additional 2× multiplier in the loss weight, reflecting the real-world consequence asymmetry — a battery in the recycling stream contaminates an entire batch of recyclables and poses fire risk in sorting facilities.

---

### 5.3 Model Architecture (`model.py`)

**Primary model: MobileNetV3-Small**

```python
import torchvision.models as models

backbone = models.mobilenet_v3_small(weights='DEFAULT')

# Freeze all backbone layers
for param in backbone.parameters():
    param.requires_grad = False

# Replace classifier head (output: 4 classes instead of 1000)
backbone.classifier = nn.Sequential(
    nn.Linear(576, 256),
    nn.Hardswish(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 4)       # 4 waste categories
)
```

**Parameter count:** MobileNetV3-Small backbone = 1.5M parameters (frozen). New classifier head = ~150K parameters (trained). Total trained parameters: ~150K — fine-tunable in under 1 hour on a laptop CPU.

**Comparison model: EfficientNet-B0**

```python
backbone = models.efficientnet_b0(weights='DEFAULT')
backbone.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 4)
)
```

Both models are trained and compared. The best checkpoint (by validation F1-macro) advances to quantization and deployment.

### 5.4 Training Configuration (`trainer.py`)

**Phase 1 — Head only (epochs 1–5):**
- Only the classifier head is trained. Backbone weights are frozen.
- Learning rate: 1e-3 with AdamW optimizer.
- This stabilizes the new head quickly before the backbone is unfrozen.

**Phase 2 — Fine-tuning (epochs 6–10):**
- Unfreeze the last 2 blocks of the backbone.
- Reduce learning rate to 1e-4 with cosine annealing scheduler.
- This allows the backbone to adapt its higher-level features to waste texture patterns without destroying the ImageNet pretraining.

```python
training_config = {
    "epochs_phase1": 5,
    "epochs_phase2": 5,
    "optimizer": "AdamW",
    "lr_phase1": 1e-3,
    "lr_phase2": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "batch_size": 32,
    "early_stopping_patience": 3,   # stop if val F1 doesn't improve for 3 epochs
    "checkpoint_metric": "f1_macro"
}
```

**Expected training time (laptop CPU):** Phase 1 ~15 min, Phase 2 ~20 min. Total under 40 minutes.

---

### 5.5 Evaluation Suite (`evaluate.py`)

#### Confusion Matrix

```
                  Predicted
                  Recycle  Compost  Hazardous  Landfill
Actual Recycle  [  TP     |   ...  |   ...    |   ...  ]
       Compost  [  ...    |   TP   |   ...    |   ...  ]
       Hazardous[  ...    |   ...  |   TP     |   ...  ]
       Landfill [  ...    |   ...  |   ...    |   TP   ]
```

The confusion matrix is the primary diagnostic tool. Hazardous→Recyclable errors are the most critical failure mode and are highlighted in red in the visualization.

#### Per-Class Metrics

| Metric | Formula | Why It Matters Here |
|---|---|---|
| Precision (per class) | TP / (TP + FP) | How often a "Hazardous" call is correct |
| Recall (per class) | TP / (TP + FN) | How many actual hazardous items are caught |
| F1-Score (per class) | 2 × P × R / (P + R) | Harmonic balance; critical for imbalanced classes |
| F1-Macro | Mean F1 across all 4 classes | Primary checkpoint selection metric |
| F1-Weighted | F1 weighted by class support | Secondary metric for stakeholder reporting |

#### Cost-Weighted Error Matrix

A domain-specific evaluation layer that assigns monetary or environmental severity to each misclassification type.

```python
cost_matrix = np.array([
    # Predicted:  Recycle  Compost  Hazardous  Landfill
    [0,           1,       5,       2],    # Actual: Recyclable
    [1,           0,       4,       1],    # Actual: Compostable
    [10,          8,       0,       3],    # Actual: Hazardous  ← highest cost errors
    [1,           1,       4,       0],    # Actual: Landfill
])

# Cost-weighted error = sum(confusion_matrix * cost_matrix) / total_samples
```

**Rationale:** Hazardous→Recyclable (cost=10) represents a battery entering the recycling stream, causing facility contamination and fire risk. This is a substantially higher real-world cost than Landfill→Compostable (cost=1). Reporting the cost-weighted error alongside accuracy gives a realistic picture of model safety.

---

### 5.6 Grad-CAM Interpretability (`gradcam.py`)

Grad-CAM (Gradient-weighted Class Activation Mapping) generates a spatial heatmap showing which regions of the input image most influenced the model's classification decision.

**Mechanism:**
1. Register a forward hook on the last convolutional layer of the backbone.
2. Run a forward pass on the input image.
3. Compute the gradient of the target class score with respect to the feature map activations.
4. Weight each feature map channel by its mean gradient (global average pooling over spatial dims).
5. Apply ReLU to retain only positive contributions.
6. Upsample to input image resolution and overlay as a heatmap.

```python
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def generate(self, image_tensor, class_idx):
        output = self.model(image_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam / cam.max()                       # normalize to [0, 1]
        return cam.detach().numpy()
```

**Visualization:** Upsample CAM to 224×224, apply `cv2.COLORMAP_JET`, blend with original image at alpha=0.5.

**Qualitative test:** A battery classified as Hazardous should show high activation on the cylindrical body, not on the background. A plastic bottle classified as Recyclable should activate on the bottle shape and label. If activations are on backgrounds, this is a data leakage signal (e.g., all hazardous images have the same background) and triggers a dataset audit.

---

### 5.7 Quantization Pipeline (`quantize.py`)

Converts the best FP32 checkpoint to INT8 using PyTorch static quantization.

**Steps:**

1. **Fuse layers:** Fuse Conv+BN+ReLU sequences for efficiency.
   ```python
   model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
   ```

2. **Prepare:** Insert quantization observers at layer boundaries.
   ```python
   model_fused.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # x86 CPUs
   torch.quantization.prepare(model_fused, inplace=True)
   ```

3. **Calibrate:** Run 100–200 representative calibration images through the model to collect activation statistics.
   ```python
   for images, _ in calibration_loader:
       model_fused(images)
   ```

4. **Convert:**
   ```python
   model_int8 = torch.quantization.convert(model_fused, inplace=True)
   ```

5. **Benchmark:** Compare FP32 vs. INT8 on 100 test images.

**Expected results:**

| Metric | FP32 | INT8 | Delta |
|---|---|---|---|
| Model size | ~10 MB | ~2.5 MB | −75% |
| Inference latency (CPU) | ~80 ms | ~25 ms | −69% |
| Top-1 Accuracy | baseline | baseline − 1–2% | Acceptable |
| F1-Macro | baseline | baseline − 0.5–1.5% | Acceptable |

**Optional: OpenVINO path (Intel CPUs)**

For additional speedup on Intel CPUs, export to ONNX then convert to OpenVINO IR and quantize with NNCF:

```bash
# Export to ONNX
python -c "torch.onnx.export(model, dummy_input, 'model.onnx')"

# Convert to OpenVINO IR
mo --input_model model.onnx --output_dir openvino_ir/

# INT8 quantization via NNCF
python quantize_openvino.py --model openvino_ir/model.xml
```

OpenVINO INT8 on Intel Core i5/i7 typically achieves < 10 ms per image — effectively real-time.

---

### 5.8 FastAPI Inference Endpoint (`api.py`)

Wraps the INT8 model as a locally served REST API.

**Endpoint:**

```
POST /classify
Content-Type: multipart/form-data
Body: image file

Response:
{
  "category": "Hazardous",
  "confidence": 0.94,
  "top3": [
    {"category": "Hazardous", "confidence": 0.94},
    {"category": "Landfill",  "confidence": 0.04},
    {"category": "Recyclable","confidence": 0.02}
  ],
  "disposal_instruction": "Do not place in recycling. Take to a designated hazardous waste collection point.",
  "gradcam_heatmap_b64": "<base64 encoded PNG>",
  "inference_latency_ms": 23.4
}
```

**Additional endpoints:**

```
GET  /health              → {"status": "ok", "model": "MobileNetV3-Small-INT8"}
GET  /categories          → List of 4 categories with disposal instructions
POST /classify/batch      → Accept up to 10 images, return list of results
```

**Startup:** Model is loaded once into memory at server startup using FastAPI's `lifespan` context manager. Inference is synchronous (no async needed at this throughput level).

---

### 5.9 Gradio UI (`app.py`)

A drag-and-drop web interface for interactive testing.

**UI layout:**

```
┌─────────────────────────────────────────────────────────┐
│  Smart-Sort AI  🌱                                      │
├───────────────────────────────┬─────────────────────────┤
│  Upload or Webcam Capture     │  Classification Result  │
│  ┌─────────────────────────┐  │  ┌─────────────────┐   │
│  │                         │  │  │ HAZARDOUS ⚠️    │   │
│  │   [Drag image here]     │  │  │ Confidence: 94% │   │
│  │                         │  │  └─────────────────┘   │
│  └─────────────────────────┘  │                        │
│  [Upload] [Webcam Snapshot]   │  ┌─────────────────┐   │
│                               │  │ Grad-CAM        │   │
│                               │  │ [heatmap image] │   │
│                               │  └─────────────────┘   │
│                               │                        │
│                               │  ┌─────────────────┐   │
│                               │  │ Disposal:       │   │
│                               │  │ Take to hazmat  │   │
│                               │  │ collection.     │   │
│                               │  └─────────────────┘   │
│                               │  Latency: 23 ms        │
└───────────────────────────────┴─────────────────────────┘
```

**Webcam mode:** Captures a single frame on button click (not continuous stream). Frame is passed to the local FastAPI endpoint and results rendered in the same panel.

---

### 5.10 Disposal Instruction Engine (`instructions.py`)

A simple rule-based lookup table that maps predicted category to human-readable disposal guidance. This is the "last mile" that makes the system actionable rather than just a classifier.

```python
DISPOSAL_INSTRUCTIONS = {
    "Recyclable": {
        "action": "Place in the blue recycling bin.",
        "tip": "Rinse containers before recycling. Remove caps from bottles.",
        "icon": "♻️"
    },
    "Compostable": {
        "action": "Place in the green compost bin or home compost.",
        "tip": "Remove any stickers or rubber bands from produce.",
        "icon": "🌱"
    },
    "Hazardous": {
        "action": "Do NOT place in any household bin. Take to a designated hazardous waste collection facility.",
        "tip": "Keep in original container if possible. Check local authority for collection schedule.",
        "icon": "⚠️",
        "alert": True        # triggers red warning banner in UI
    },
    "Landfill": {
        "action": "Place in the general waste bin.",
        "tip": "Consider whether this item could be repaired or donated before disposal.",
        "icon": "🗑️"
    }
}
```

---

## 6. Evaluation Results Template

The `notebooks/04_evaluation_report.ipynb` produces a structured report with the following sections:

1. **Dataset statistics:** Class distribution (train/val/test), augmentation examples.
2. **Training curves:** Loss and F1-macro over epochs for both phases. Overfitting detection via train/val gap.
3. **Confusion matrix:** Annotated with high-cost error cells highlighted in red.
4. **Per-class metrics table:** Precision, Recall, F1 for all 4 categories for both models.
5. **Cost-weighted error:** Single number summarizing real-world misclassification severity.
6. **Grad-CAM gallery:** 16 examples (4 per class) showing correct activations and 4 failure cases.
7. **Quantization benchmark:** FP32 vs. INT8 comparison table.
8. **Model comparison:** MobileNetV3-Small vs. EfficientNet-B0 on F1-macro, latency, size.

---

## 7. Project Structure

```
smart-sort-ai/
├── config.yaml                     # Dataset paths, training hyperparams, model settings
├── train.py                        # CLI: full training pipeline (both phases)
├── evaluate.py                     # Confusion matrix, F1, cost-weighted error, Grad-CAM gallery
├── quantize.py                     # FP32 → INT8 quantization and benchmark
├── api.py                          # FastAPI inference endpoint
├── app.py                          # Gradio UI
├── data_pipeline.py                # Dataset loading, augmentation, WeightedRandomSampler
├── model.py                        # MobileNetV3-Small and EfficientNet-B0 definitions
├── trainer.py                      # Two-phase training loop with early stopping
├── gradcam.py                      # Grad-CAM implementation and visualization
├── instructions.py                 # Disposal instruction lookup table
├── data/
│   ├── trashnet/                   # Raw TrashNet dataset (gitignored)
│   ├── supplemental/               # Compostable and Hazardous augmentation images
│   ├── train/                      # Processed split (gitignored)
│   ├── val/
│   └── test/
├── models/
│   ├── mobilenetv3_fp32.pt         # Best FP32 checkpoint (gitignored)
│   ├── mobilenetv3_int8.pt         # INT8 quantized model
│   ├── efficientnet_fp32.pt
│   └── openvino_ir/                # Optional: OpenVINO export
├── notebooks/
│   ├── 01_eda_trashnet.ipynb       # Dataset exploration and remapping
│   ├── 02_training_analysis.ipynb  # Learning curves and hyperparameter sensitivity
│   ├── 03_gradcam_gallery.ipynb    # Visual explanation audit
│   └── 04_evaluation_report.ipynb  # Full metrics report
├── tests/
│   ├── test_model.py               # Unit: output shape, class count
│   ├── test_gradcam.py             # Unit: heatmap shape matches input resolution
│   ├── test_api.py                 # Integration: POST /classify returns expected schema
│   └── test_cost_matrix.py         # Unit: cost matrix shape and Hazardous weight
├── pyproject.toml                  # uv-managed dependencies
└── README.md
```

---

## 8. Setup and Run

```bash
# 1. Create environment
uv venv
source .venv/bin/activate

# 2. Core dependencies
uv pip install torch torchvision pillow scikit-learn matplotlib \
               gradio fastapi uvicorn python-multipart tqdm

# 3. Optional: OpenVINO path (Intel CPUs)
uv pip install openvino nncf

# 4. Prepare dataset
python data_pipeline.py --trashnet data/trashnet/ \
                         --supplemental data/supplemental/ \
                         --output data/

# 5. Train
python train.py --model mobilenetv3 --epochs 10 --output models/

# 6. Evaluate and generate Grad-CAM gallery
python evaluate.py --checkpoint models/mobilenetv3_fp32.pt \
                   --test-dir data/test/ \
                   --report outputs/eval_report.json

# 7. Quantize to INT8
python quantize.py --checkpoint models/mobilenetv3_fp32.pt \
                   --calibration-dir data/val/ \
                   --output models/mobilenetv3_int8.pt

# 8. Start API server
uvicorn api:app --host 127.0.0.1 --port 8000

# 9. Launch Gradio UI
python app.py
```

---

## 9. Configuration (`config.yaml`)

```yaml
data:
  trashnet_dir: data/trashnet/
  supplemental_dir: data/supplemental/
  output_dir: data/
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  image_size: 224
  num_workers: 2

model:
  architecture: mobilenetv3_small    # options: mobilenetv3_small | efficientnet_b0
  pretrained: true
  num_classes: 4
  dropout: 0.3

training:
  epochs_phase1: 5
  epochs_phase2: 5
  batch_size: 32
  lr_phase1: 0.001
  lr_phase2: 0.0001
  weight_decay: 0.01
  early_stopping_patience: 3
  checkpoint_metric: f1_macro
  hazardous_loss_multiplier: 2.0     # extra penalty for Hazardous errors

quantization:
  method: pytorch_static             # options: pytorch_static | openvino_nncf
  calibration_batches: 10           # 10 batches × 32 = 320 calibration images

api:
  host: "127.0.0.1"
  port: 8000
  max_batch_size: 10

categories:
  - Recyclable
  - Compostable
  - Hazardous
  - Landfill
```

---

## 10. Key Design Decisions

### 10.1 Two-Phase Fine-Tuning over Full Fine-Tuning

Unfreezing the entire backbone from epoch 1 risks catastrophic forgetting — the ImageNet representations that capture edges, textures, and shapes are overwritten by 2,500 domain images too quickly. The two-phase approach (frozen backbone → warm head → selectively unfrozen backbone) is standard practice for small-dataset transfer learning and produces 3–5% better F1-macro than single-phase fine-tuning in this regime.

### 10.2 Cost Matrix as a Primary Metric, Not a Secondary Footnote

In waste classification, accuracy is a misleading headline metric. A model that always predicts Recyclable achieves ~70% accuracy (given class distribution) while sending every hazardous item to the recycling stream. The cost-weighted error is the metric that captures real-world consequence and should be the first number reported in any stakeholder summary.

### 10.3 Static Quantization over Dynamic Quantization

Dynamic quantization quantizes only the weights at load time; activations are quantized at runtime per-batch. Static quantization quantizes both weights and activations using calibration data, producing better latency reduction (typically 2–4× vs. 1.5–2× for dynamic). For a fixed-distribution inference workload (product photos), static quantization is the right choice.

### 10.4 Grad-CAM as a Dataset Audit Tool, Not Just an Explanation Tool

Grad-CAM serves a dual purpose: user trust and model debugging. During development, if Grad-CAM shows consistent activation on image backgrounds (e.g., all hazardous images were photographed on a white surface while recyclables were on a green surface), this reveals spurious correlations in the dataset that must be corrected before deployment. Running a Grad-CAM audit on 20 images per class during development is as important as the confusion matrix.

---

## 11. Learning Objectives Mapped to Implementation

| Learning Objective | Where Implemented |
|---|---|
| Transfer learning with frozen backbone | `model.py` + `trainer.py` Phase 1 |
| Progressive unfreezing (two-phase fine-tuning) | `trainer.py` Phase 2 |
| Class imbalance: weighted sampler + weighted loss | `data_pipeline.py` + `trainer.py` |
| Evaluation beyond accuracy: F1, cost matrix | `evaluate.py` |
| Confusion matrix as diagnostic tool | `evaluate.py` + `notebooks/04` |
| Grad-CAM visual explanation | `gradcam.py` |
| INT8 static quantization | `quantize.py` |
| Model serving via REST API | `api.py` (FastAPI) |
| Interactive demo UI | `app.py` (Gradio) |
| Latency benchmarking (FP32 vs INT8) | `quantize.py --benchmark` |

---

## 12. Transferable Skills Framing

**Class imbalance handling → Fraud detection.** The exact techniques used here — weighted sampler, class-weighted cross-entropy loss, amplified minority class penalty — are the same techniques applied to fraud detection datasets where fraud:legitimate transaction ratios are 1:500 or worse. The framing transfers directly.

**Cost-weighted error matrix → Credit risk decision thresholds.** The cost matrix formalizes the asymmetry between false negatives (missing a hazardous item) and false positives (over-classifying recyclables). This is identical in structure to the confusion cost matrices used in credit decisioning, where a missed default (FN) has a different cost from a rejected good applicant (FP). The pattern of building domain-specific cost matrices and using them to select decision thresholds is a senior DS competency.

**Grad-CAM → Model governance and explainability in finserv.** Regulators in financial services (RBI, SEBI, SR 11-7) increasingly require model explainability. Grad-CAM is the CV analog of SHAP for tabular models — both answer "which input features drove this decision?" Demonstrating Grad-CAM competency signals awareness of the explainability requirements that govern production ML in regulated industries.

**Two-phase fine-tuning → Domain adaptation in NLP.** The frozen-backbone + trained-head → selective-unfreezing pattern in CV is structurally identical to BERT fine-tuning (freeze all → train classification head → unfreeze top transformer layers). Understanding this pattern at a conceptual level makes it easy to explain domain adaptation in any modality.

---

## 13. Portfolio Narrative

**Real-world impact with quantifiable stakes.** Waste misclassification is not an abstract benchmark problem — Hazardous→Recyclable errors create real facility contamination and safety incidents. Framing the cost matrix results in terms of "X% of hazardous items would have been incorrectly routed to recycling at baseline accuracy; our model reduces this to Y%" gives the project a concrete business impact narrative.

**Full MLOps arc in one project.** From data remapping through class imbalance handling, two-phase fine-tuning, cost-sensitive evaluation, Grad-CAM auditing, INT8 quantization, API serving, and UI deployment — this project covers more of the production ML lifecycle than most CV portfolio pieces. It demonstrates that the work doesn't stop at training accuracy.

**Lightweight stack with production-grade thinking.** MobileNetV3-Small at < 3 MB (INT8) with < 25 ms CPU inference is a credible edge AI story, not a toy demo. The OpenVINO extension (< 10 ms) makes it a genuine embedded deployment candidate. This resonates with the edge AI and agentic AI direction of current finserv ML work.

**Responsible AI by design.** Grad-CAM as a dataset audit tool (not just a UI feature), the cost matrix prioritizing Hazardous recall, and the explicit disposal instruction engine all demonstrate that the system was designed to be safe and trustworthy, not just accurate. This is the framing that resonates with senior hiring managers in regulated industries.

---

## 14. Open Questions and Future Work

- **v1.1:** Extend the class set to subcategories within Recyclable (paper vs. glass vs. plastic vs. metal) to enable bin-level routing instructions rather than just category-level guidance.
- **v1.2:** Add an uncertainty quantification layer (Monte Carlo Dropout or temperature scaling) so the system can say "I'm only 55% confident — please verify manually" on ambiguous items rather than always forcing a hard prediction.
- **v2.0:** Add a YOLOv11n detection head in front of the classifier to support images containing multiple waste items — segment each item, classify independently, return per-item disposal instructions.
- **v2.1:** Build a personal learning loop: log all user classifications with optional user feedback ("that was wrong"), fine-tune the head nightly on the correction set using continual learning to adapt to the user's specific waste stream.
- **Research angle:** Evaluate adversarial robustness — can a small perturbation to a battery image cause the model to classify it as Recyclable? Given that the cost of this specific error is the highest in the cost matrix, the model's vulnerability to adversarial inputs in the Hazardous class is a safety-critical research question that connects directly to your adversarial ML background.
