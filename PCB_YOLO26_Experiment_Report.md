# PCB Inspection — YOLO26 Training & Evaluation Report

> **Purpose:** Living experiment log for the PCB defect-detection project.
>
> This document is intentionally structured so that future experiments can be added without changing the existing records. Each experiment should receive a new `Run ID` and should record **model + dataset + training configuration + evaluation metrics + qualitative observations + changes from the previous run**.

---

## 1. Project Context

### Objective

Develop a PCB inspection pipeline using object detection / anomaly detection techniques.

The current supervised baseline uses **YOLO26s** fine-tuned on the **PKU-Market-PCB defect dataset**.

### Current supervised defect classes

1. `mouse_bite`
2. `spur`
3. `missing_hole`
4. `short`
5. `open_circuit`
6. `spurious_copper`

### Important distinction

The YOLO26 experiments documented here are **supervised defect detection** experiments.

They learn:

```text
PCB image
    ↓
YOLO26
    ↓
defect class + bounding box + confidence
```

They are not, by themselves, golden-image/template comparison or unsupervised anomaly detection experiments.

---

# 2. Dataset Information

## 2.1 PKU-Market-PCB

The current YOLO26 experiments use the **PKU-Market-PCB** dataset.

Previously recorded dataset information:

- Approximately **1,386 color images**
- Color information is retained.
- The dataset contains the six classes listed above.
- The project intentionally does **not** binarize the PKU color images merely to make them visually consistent with DeepPCB, because retaining color information is considered important.

### Dataset configuration used in the current run

```text
data: /content/drive/MyDrive/colab/data.yaml
split: val
```

The project dataset setup also contains train/validation/test YOLO splits.

### Dataset-related consideration for future experiments

A major point to verify in future evaluation is whether the validation/test split is sufficiently independent at the **PCB/board level**, rather than only being a random image-level split.

---

# 3. Experiment Index

| Run ID | Model | Dataset | Epochs | Image Size | Batch | Device | Key Modification | mAP@0.5 | F1 | Status |
|---|---|---|---:|---:|---:|---|---|---:|---:|---|
| `YOLO26S-001` | YOLO26s | PKU-Market | 50 | 640 | 16 | GPU 0 / Colab | Baseline pretrained fine-tuning | **0.988** | **0.98** | Complete |
| `YOLO26S-002` | YOLO26s | PKU-Market | TBD | 640 | TBD | TBD | Longer/convergence run | TBD | TBD | Planned |
| `YOLO26S-003` | YOLO26s | PKU-Market | TBD | TBD | TBD | TBD | Future module/augmentation experiment | TBD | TBD | Planned |

---

# 4. Run YOLO26S-001 — Baseline

## 4.1 Run identification

| Field | Value |
|---|---|
| Run ID | `YOLO26S-001` |
| Run name | `pku_yolo26s-3` |
| Task | `detect` |
| Model | `yolo26s.pt` |
| Pretrained | Yes |
| Dataset | PKU-Market-PCB |
| Dataset YAML | `/content/drive/MyDrive/colab/data.yaml` |
| Epochs | 50 |
| Batch size | 16 |
| Image size | 640 × 640 |
| Device | `0` |
| Environment | Google Colab |
| AMP | Enabled |
| Validation | Enabled |
| Validation split | `val` |
| Seed | 0 |
| Deterministic | Yes |
| Run directory | `/content/drive/MyDrive/colab/runs/detect/pku_yolo26s-3` |

---

# 5. Run YOLO26S-001 — Full Training Arguments

## 5.1 General

```yaml
task: detect
mode: train

model: /content/drive/MyDrive/colab/yolo26s.pt
data: /content/drive/MyDrive/colab/data.yaml

epochs: 50
time: null
patience: 100

batch: 16
imgsz: 640

save: true
save_period: -1

cache: false
device: '0'
workers: 8

project: /content/drive/MyDrive/colab/runs/detect
name: pku_yolo26s-3
exist_ok: false

pretrained: true
cls_remap: true
optimizer: auto

verbose: true
seed: 0
deterministic: true

single_cls: false
rect: false
cos_lr: false

close_mosaic: 10
resume: false

amp: true
fraction: 1.0

profile: false
freeze: null
multi_scale: 0.0
compile: false
channels_last: null
```

## 5.2 Validation / Inference Settings

```yaml
val: true
split: val

save_json: false
conf: null
iou: 0.7
max_det: 300

dnn: false
nms: null

source: null
vid_stride: 1
stream_buffer: false
visualize: false

classes: null
agnostic_nms: false
```

## 5.3 Visualization / Output

```yaml
plots: true

show: false
save_frames: false
save_txt: false
save_conf: false
save_crop: false

show_labels: true
show_conf: true
show_boxes: true

line_width: null
```

## 5.4 Export Settings

```yaml
format: torchscript
keras: false
optimize: false
dynamic: false
simplify: true
opset: null
workspace: null
```

These are export-related settings and should be considered separately from the actual training configuration.

## 5.5 Optimization / Loss Settings

```yaml
lr0: 0.01
lrf: 0.01

momentum: 0.937
weight_decay: 0.0005

warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

nbs: 64

dis: 6.0
box: 7.5
cls: 0.5
cls_pw: 0.0
dfl: 1.5

pose: 12.0
kobj: 1.0
rle: 1.0
angle: 1.0
dlog: 1.0
dgrad: 0.5
dlam: 1.0
```

## 5.6 Augmentation

```yaml
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4

degrees: 0.0
translate: 0.1
scale: 0.5

shear: 0.0
perspective: 0.0

flipud: 0.0
fliplr: 0.5

bgr: 0.0

mosaic: 1.0
mixup: 0.0
cutmix: 0.0

copy_paste: 0.0
copy_paste_mode: flip

auto_augment: randaugment
erasing: 0.4
```

Other supplied settings:

```yaml
overlap_mask: true
mask_ratio: 4
dropout: 0.0

distill_model: null
cfg: null
tracker: tracktrack.yaml
```

---

# 6. Run YOLO26S-001 — Quantitative Results

## 6.1 Precision-Recall result

Reported by the supplied Precision-Recall curve:

```text
All classes mAP@0.5 = 0.988
```

Per-class AP@0.5:

| Class | AP@0.5 |
|---|---:|
| `mouse_bite` | **0.991** |
| `spur` | **0.979** |
| `missing_hole` | **0.992** |
| `short` | **0.988** |
| `open_circuit` | **0.995** |
| `spurious_copper` | **0.981** |
| **All classes** | **0.988** |

---

# 7. Confusion Matrix

## 7.1 Raw confusion matrix

The supplied confusion matrix has:

- Rows = **Predicted**
- Columns = **True**

| Predicted ↓ / True → | mouse_bite | spur | missing_hole | short | open_circuit | spurious_copper | background |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mouse_bite` | 262 | 0 | 0 | 0 | 0 | 0 | 8 |
| `spur` | 0 | 271 | 0 | 0 | 0 | 0 | 9 |
| `missing_hole` | 0 | 0 | 283 | 0 | 0 | 0 | 6 |
| `short` | 0 | 0 | 0 | 270 | 0 | 0 | 4 |
| `open_circuit` | 0 | 0 | 0 | 0 | 263 | 0 | 2 |
| `spurious_copper` | 0 | 0 | 0 | 0 | 0 | 294 | 8 |
| `background` | 0 | 8 | 0 | 5 | 1 | 4 | — |

---

# 8. Confusion Matrix — Derived Statistics

The following are calculated directly from the supplied confusion matrix.

> These are **matrix-derived metrics** and should not be confused with a separate summary metric calculated at another confidence/IoU operating point.

## 8.1 Per-class recall

| Class | TP | FN | Recall |
|---|---:|---:|---:|
| `mouse_bite` | 262 | 0 | **100.0%** |
| `spur` | 271 | 8 | **97.1%** |
| `missing_hole` | 283 | 0 | **100.0%** |
| `short` | 270 | 5 | **98.2%** |
| `open_circuit` | 263 | 1 | **99.6%** |
| `spurious_copper` | 294 | 4 | **98.7%** |

```text
Total TP = 1643
Total FN = 18

Matrix-derived recall ≈ 98.92%
```

## 8.2 Per-class precision

| Class | TP | FP | Precision |
|---|---:|---:|---:|
| `mouse_bite` | 262 | 8 | **97.0%** |
| `spur` | 271 | 9 | **96.8%** |
| `missing_hole` | 283 | 6 | **97.9%** |
| `short` | 270 | 4 | **98.5%** |
| `open_circuit` | 263 | 2 | **99.2%** |
| `spurious_copper` | 294 | 8 | **97.4%** |

```text
Total TP = 1643
Total FP = 37

Matrix-derived precision ≈ 97.80%
```

## 8.3 Matrix-derived F1

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| `mouse_bite` | 97.0% | 100.0% | **98.50%** |
| `spur` | 96.8% | 97.1% | **96.96%** |
| `missing_hole` | 97.9% | 100.0% | **98.95%** |
| `short` | 98.5% | 98.2% | **98.36%** |
| `open_circuit` | 99.2% | 99.6% | **99.43%** |
| `spurious_copper` | 97.4% | 98.7% | **98.00%** |

> The supplied F1-confidence curve reports an all-class F1 of approximately **0.98 at confidence ≈ 0.454**. The matrix-derived values above are calculated from the displayed matrix and may use a different operating point.

---

# 9. Confusion Matrix Error Analysis

## False negatives

Defects missed and classified as background:

```text
spur              → 8
short             → 5
spurious_copper   → 4
open_circuit      → 1
mouse_bite        → 0
missing_hole      → 0

Total FN = 18
```

## False positives

Background regions predicted as defects:

```text
mouse_bite        ← 8
spur              ← 9
missing_hole      ← 6
short             ← 4
open_circuit      ← 2
spurious_copper   ← 8

Total FP = 37
```

## Defect-to-defect confusion

The supplied confusion matrix shows essentially **no off-diagonal defect-to-defect errors**.

The visible errors are predominantly:

```text
defect → background
background → defect
```

rather than:

```text
defect A → defect B
```

---

# 10. Normalized Confusion Matrix

The supplied normalized matrix shows approximately:

| Class | Normalized correct detection |
|---|---:|
| `mouse_bite` | **1.00** |
| `spur` | **0.97** |
| `missing_hole` | **1.00** |
| `short` | **0.98** |
| `open_circuit` | **0.99** |
| `spurious_copper` | **0.99** |

Background-related normalized values shown in the supplied plot include:

```text
Background → spur              ≈ 0.03
Background → short             ≈ 0.02
Background → spurious_copper   ≈ 0.01
```

The true-background column shows approximately:

```text
mouse_bite        0.22
spur              0.24
missing_hole      0.16
short             0.11
open_circuit      0.05
spurious_copper   0.22
```

---

# 11. Recall–Confidence Curve

The supplied curve reports:

```text
All classes:
Recall ≈ 0.99 at confidence = 0.000
```

Observed shape:

- Recall remains very high through a substantial confidence range.
- Recall begins dropping sharply around approximately 0.65–0.75.
- Recall approaches zero around approximately 0.8–0.85.

---

# 12. Precision–Confidence Curve

The supplied curve reports:

```text
All classes:
Precision = 1.00 at confidence ≈ 0.823
```

Observed behavior:

- Precision rises rapidly at low confidence.
- Most classes reach very high precision early.
- Precision remains close to 1.0 through much of the confidence range.
- The all-class curve reaches approximately 1.00 around confidence 0.823.

---

# 13. F1–Confidence Curve

The supplied curve reports:

```text
All classes:
F1 ≈ 0.98
at confidence ≈ 0.454
```

| Run | Best all-class F1 | Confidence at F1 optimum |
|---|---:|---:|
| `YOLO26S-001` | **0.98** | **0.454** |

This confidence value should be recorded in future runs because it can be useful when comparing confidence calibration after architecture or training changes.

---

# 14. Precision–Recall Curves

The supplied PR curve reports:

```text
mAP@0.5 = 0.988
```

Per-class:

```text
mouse_bite       = 0.991
spur             = 0.979
missing_hole     = 0.992
short            = 0.988
open_circuit     = 0.995
spurious_copper  = 0.981
```

---

# 15. Qualitative Prediction Results

The supplied prediction grids show detections for all six defect categories.

Examples shown include:

- `mouse_bite`
- `open_circuit`
- `short`
- `spur`
- `spurious_copper`
- `missing_hole`

The supplied examples generally show bounding boxes positioned around the defect regions.

A second set of supplied visualizations includes predictions with confidence values around `0.6–0.8`, including examples such as:

```text
mouse_bite 0.7
open_circuit 0.7
short 0.7
missing_hole 0.7–0.8
spur 0.7
spurious_copper 0.7–0.8
```

One supplied visualization has strong false-color / stripe-like appearance. The exact visualization-processing configuration was not provided, so this should be recorded as a visualization observation rather than as a model-property conclusion.

---

# 16. Baseline Interpretation

## What this run establishes

The run establishes that a pretrained YOLO26s model can be fine-tuned on PKU-Market-PCB and achieve very high performance on the supplied validation split.

Key values:

```text
mAP@0.5          = 0.988
Best F1           ≈ 0.98
F1 confidence     ≈ 0.454
Precision         = 1.00 at confidence ≈ 0.823
```

The confusion matrix shows very little defect-to-defect confusion.

## What this run does NOT establish

It does not yet establish:

- performance on genuinely unseen PCB layouts;
- robustness to new camera setups;
- robustness to substantially different lighting;
- real production-line performance;
- performance on the user's own PCB hardware/camera;
- golden-image/template comparison performance;
- unsupervised anomaly-detection performance;
- deployment latency/FPS;
- GPU/CPU power consumption;
- performance after export/quantization.

---

# 17. Dataset Generalization Caveat

The current reported result is:

```text
mAP@0.5 = 0.988
```

on the supplied validation evaluation.

Before using this number as the main project performance claim, future evaluation should verify that validation/test data are independent at the **PCB/board level**, rather than only being randomly split images.

A stronger future evaluation is:

```text
TRAIN:
PCB/board groups A, B, C, ...

VALIDATION:
different PCB/board groups

TEST:
completely held-out PCB/board groups
```

---

# 18. Current Baseline Configuration Summary

```text
Model:
    YOLO26s

Initialization:
    pretrained = true

Dataset:
    PKU-Market-PCB

Classes:
    mouse_bite
    spur
    missing_hole
    short
    open_circuit
    spurious_copper

Training:
    epochs = 50
    batch = 16
    imgsz = 640
    optimizer = auto
    lr0 = 0.01
    lrf = 0.01
    momentum = 0.937
    weight_decay = 0.0005
    warmup_epochs = 3
    AMP = true
    seed = 0
    deterministic = true

Main augmentation:
    HSV augmentation
    translation = 0.1
    scale = 0.5
    horizontal flip = 0.5
    mosaic = 1.0
    randaugment
    erasing = 0.4

Validation:
    split = val
    IoU = 0.7
    max_det = 300
    plots = true

Results:
    mAP@0.5 = 0.988
    best F1 ≈ 0.98 @ confidence ≈ 0.454
    precision = 1.00 @ confidence ≈ 0.823
```

---

# 19. Previously Recorded YOLO/PCB Experiment History

The project history contains these recorded milestones:

| Commit / Run reference | Description |
|---|---|
| `23deca7` | Earlier YOLO test run |
| `03ffb5c` | Dataset links |
| `0643974` | Updated dataset links |
| `fccc6c6` | YOLO26s test run |
| `5da7d4c` | Train script and plan |
| `e562dbb` | Environment requirements |
| `9505717` | Implemented early stopping and checkpointing |
| `9d8bc90` | First Train on PKU-Market Dataset |
| `6d553dd` | First train results |

Earlier run/output references mentioned in the project history include:

```text
pku_yolo26s-2
predict-2
predict-3
```

Exact quantitative metrics for those earlier runs were not included in the current test-results record, so their values should be filled in when their corresponding `results.csv`, plots, or evaluation outputs are available.

---

# 20. Future Experiment Template

Copy this section for every new run.

## RUN: `YOLO26S-XXX`

### 20.1 Identification

| Field | Value |
|---|---|
| Run ID | |
| Run name | |
| Date | |
| Commit | |
| Model | |
| Base weights | |
| Dataset | |
| Dataset version | |
| Dataset split | |
| Hardware | |
| Framework version | |
| Purpose | |

### 20.2 What changed?

```text
[Describe ONLY the changes from the previous run.]
```

Examples:

```text
- Added DINOv2 feature module
- Changed imgsz from 640 → 768
- Added P2 detection head
- Changed augmentation
- Changed backbone
- Changed loss
- Changed confidence threshold
```

### 20.3 Training configuration

```yaml
epochs:
batch:
imgsz:
device:
optimizer:
lr0:
lrf:
momentum:
weight_decay:
warmup_epochs:
seed:
deterministic:
amp:

mosaic:
mixup:
cutmix:
scale:
translate:
fliplr:
hsv_h:
hsv_s:
hsv_v:
```

### 20.4 Validation configuration

```yaml
split:
conf:
iou:
max_det:
```

### 20.5 Quantitative results

| Metric | Result |
|---|---:|
| mAP@0.5 | |
| mAP@0.5:0.95 | |
| Precision | |
| Recall | |
| Best F1 | |
| Confidence @ best F1 | |
| Inference FPS | |
| Inference latency | |
| Model parameters | |
| Model size | |

### 20.6 Per-class results

| Class | AP@0.5 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| `mouse_bite` | | | | |
| `spur` | | | | |
| `missing_hole` | | | | |
| `short` | | | | |
| `open_circuit` | | | | |
| `spurious_copper` | | | | |

### 20.7 Confusion matrix

| Predicted ↓ / True → | mouse_bite | spur | missing_hole | short | open_circuit | spurious_copper | background |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mouse_bite` | | | | | | | |
| `spur` | | | | | | | |
| `missing_hole` | | | | | | | |
| `short` | | | | | | | |
| `open_circuit` | | | | | | | |
| `spurious_copper` | | | | | | | |
| `background` | | | | | | | |

### 20.8 Qualitative observations

```text
Good detections:
-

Missed detections:
-

False positives:
-

Localization issues:
-

Small-defect performance:
-

Lighting sensitivity:
-

Other observations:
-
```

### 20.9 Resource / deployment measurements

```text
GPU:
CPU:
VRAM:
RAM:
Inference latency:
FPS:
Model size:
Export format:
Quantization:
Power:
```

### 20.10 Conclusion

```text
What did this experiment establish?

-

What changed relative to the previous run?

-

Did the modification improve the metrics?

-

What should be tested next?

-
```

---

# 21. Master Comparison Table

Keep one table like this near the top of the document and add one row for each completed experiment.

| Run | Model | Dataset | ImgSz | Epochs | Module / Change | mAP50 | mAP50-95 | Precision | Recall | F1 | FPS | Notes |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| `YOLO26S-001` | YOLO26s | PKU | 640 | 50 | Baseline | **0.988** | — | — | — | **0.98** | — | Baseline |
| `YOLO26S-002` | YOLO26s | PKU | 640 | | | | | | | | | |
| `YOLO26S-003` | YOLO26s | PKU | | | | | | | | | | |
| `YOLO26S-004` | | | | | | | | | | | | |

---

# 22. Suggested Experiment Progression

The experiment structure can progress from the clean baseline toward the complete inspection architecture:

```text
RUN 001
YOLO26s baseline
        ↓
RUN 002
Longer training / convergence verification
        ↓
RUN 003
Resolution experiment
        ↓
RUN 004
Model-size experiment
        ↓
RUN 005
Architecture/module addition
        ↓
RUN 006
Feature/anomaly branch
        ↓
RUN 007
Golden-image comparison
        ↓
RUN 008
Combined inspection pipeline
        ↓
RUN 009
Real PCB / camera evaluation
        ↓
RUN 010
Edge/deployment optimization
```

Each experiment should ideally change **one major variable at a time** so that improvements can be attributed to the modification.

---

# 23. Baseline Status

## Current baseline

**YOLO26s + pretrained weights + PKU-Market-PCB**

```text
Status: COMPLETE

mAP@0.5:          0.988
Best F1:          ≈ 0.98
F1 confidence:    ≈ 0.454
Precision=1.00:   confidence ≈ 0.823

Matrix-derived:
Recall:           ≈ 98.92%
Precision:        ≈ 97.80%

Primary matrix errors:
    spur → background             8
    short → background            5
    spurious_copper → background  4
    open_circuit → background     1

    background → spur             9
    background → mouse_bite       8
    background → spurious_copper  8
    background → missing_hole     6
    background → short            4
    background → open_circuit     2
```

### Baseline conclusion

The current run provides a strong **supervised YOLO26s reference point** for future experiments.
