# PCB Inspection

## Problem Statement

We are developing a computer vision system for detecting defects in Printed Circuit Boards (PCBs) using visual inspection.

Traditional PCB inspection systems can rely on relatively large or computationally expensive vision models, which can make deployment on **edge and mobile devices** challenging due to limitations in computation, memory, power consumption, and inference latency.

This project explores **YOLO26 as the base object-detection architecture** for PCB defect detection, with the goal of developing a lightweight and efficient inspection system suitable for practical edge deployment.

## Goals

* Detect common PCB defects reliably using computer vision.
* Use **YOLO26** as the base detection architecture.
* Train and fine-tune the model using publicly available PCB defect datasets.
* Optimize the model for resource-constrained edge and mobile devices.
* Maintain a practical balance between:

  * Detection accuracy
  * Inference speed
  * Model size
  * Computational requirements
* Develop a reproducible training and evaluation pipeline.

## Dataset

The current training pipeline uses the **PKU PCB Defect Dataset**.

The dataset is organized in the standard YOLO format and is configured through a `data.yaml` file containing:

* Training split
* Validation split
* Test split
* Six defect classes

### Defect Classes

| Class | Defect          |
| ----- | --------------- |
| 0     | Mouse Bite      |
| 1     | Spur            |
| 2     | Missing Hole    |
| 3     | Short           |
| 4     | Open Circuit    |
| 5     | Spurious Copper |

The dataset configuration is validated before training to ensure that all three splits (`train`, `val`, and `test`) are available and that exactly six classes are defined.

## Model

### Current Model

**YOLO26s**

* Base architecture: YOLO26
* Model variant: Small (`s`)
* Parameters: ~9.5M
* Training mode: Pretrained-weight fine-tuning

The training script loads pretrained YOLO26s weights and fine-tunes them on the PKU PCB defect dataset.

## Training

Training is handled by `train.py`.

The script provides configurable options for:

* Dataset YAML
* Model/checkpoint
* Number of epochs
* Image resolution
* Batch size
* Device
* Number of dataloader workers
* Experiment name
* Output directory
* Training resumption

### Example

```bash
python3 train.py \
    --epochs 200 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

For CPU training:

```bash
python3 train.py \
    --epochs 200 \
    --batch 16 \
    --imgsz 640 \
    --device cpu
```

The default dataset configuration is:

```text
compiled_datasets/PKU/data.yaml
```

and the default model is:

```text
yolo26s.pt
```

Training outputs are saved under:

```text
runs/detect/
```

## Training Pipeline

The current pipeline performs the following steps:

```text
PKU PCB Dataset
       ↓
Dataset Validation
       ↓
YOLO26s Pretrained Weights
       ↓
Fine-tuning
       ↓
Validation / Testing
       ↓
Defect Detection Model
       ↓
Edge Deployment Evaluation
```

Before training begins, the script validates:

1. The dataset YAML exists.
2. Exactly six classes are defined.
3. `train`, `val`, and `test` directories exist.
4. The specified YOLO26s model weights are available.

This prevents common dataset-configuration errors from occurring after training has already started.

## Current Status

### Dataset

**PKU PCB Defect Dataset**

Status: **Integrated into training pipeline**

### Model

**YOLO26s**

Status: **Training pipeline implemented**

Parameters: **~9.5M**

### Defect Classes

* Mouse Bite
* Spur
* Missing Hole
* Short
* Open Circuit
* Spurious Copper

## Planned Work

* [ ] Establish baseline YOLO26s performance on PKU.
* [ ] Evaluate precision, recall, mAP@50, and mAP@50:95.
* [ ] Benchmark inference latency on CPU and GPU.
* [ ] Measure model memory requirements.
* [ ] Compare accuracy/speed trade-offs across YOLO26 model sizes.
* [ ] Evaluate generalization using additional publicly available PCB datasets.
* [ ] Investigate PCB-specific preprocessing and augmentation strategies.
* [ ] Explore anomaly/template-based inspection methods for defects that may not be well represented in the training data.
* [ ] Optimize the trained model for edge deployment.
* [ ] Develop an end-to-end PCB inspection inference pipeline.

## Project Direction

The long-term objective is not simply to train a detector on a single PCB dataset, but to build a **practical PCB inspection system** that can operate under real-world deployment constraints.

The project will therefore focus on the trade-off between:

**Detection Accuracy ↔ Inference Speed ↔ Model Size ↔ Hardware Requirements**

rather than optimizing solely for benchmark accuracy.

## Repository Structure

```text
PCB-Inspection/
│
├── train.py
├── yolo26s.pt
│
├── compiled_datasets/
│   └── PKU/
│       ├── data.yaml
│       ├── train/
│       ├── val/
│       └── test/
│
├── runs/
│   └── detect/
│
└── README.md
```

## License

This project is intended for research and engineering experimentation using publicly available PCB datasets.
