#!/usr/bin/env python3
"""Evaluate a trained YOLO26s model on the PKU PCB defect dataset."""

import argparse
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "data.yaml"
EXPECTED_CLASS_COUNT = 6


def validate_dataset(data_yaml: Path) -> None:
    """Fail early when the dataset configuration does not match PKU."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    with data_yaml.open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    names = config.get("names")

    if isinstance(names, dict):
        names = list(names.values())

    if not isinstance(names, list) or len(names) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} classes in {data_yaml}, "
            f"found {names!r}"
        )

    dataset_root = Path(config.get("path", data_yaml.parent))

    if not dataset_root.is_absolute():
        dataset_root = (data_yaml.parent / dataset_root).resolve()

    split_path = dataset_root / config.get("test", "test")

    if not split_path.is_dir():
        raise FileNotFoundError(
            f"Test split directory not found: {split_path}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to the YOLO dataset YAML.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the trained model checkpoint, e.g. best.pt.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size used during evaluation.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device such as 0, or cpu.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers.",
    )

    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split to evaluate on (default: test).",
    )

    parser.add_argument(
        "--project",
        type=Path,
        default=PROJECT_DIR / "runs" / "detect",
        help="Directory where evaluation results are saved.",
    )

    parser.add_argument(
        "--name",
        default="pku_yolo26s_test",
        help="Name of the evaluation run.",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help=(
            "Confidence threshold. Leave unset to let Ultralytics "
            "use its default validation behaviour."
        ),
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
        help="Maximum number of detections per image.",
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save evaluation results in COCO-style JSON format.",
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use test-time augmentation during evaluation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_yaml = args.data.resolve()
    model_path = args.model.resolve()

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------

    validate_dataset(data_yaml)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}"
        )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    from ultralytics import YOLO

    print("=" * 70)
    print("YOLO26 MODEL EVALUATION")
    print("=" * 70)

    print(f"Model : {model_path}")
    print(f"Data  : {data_yaml}")
    print(f"Split : {args.split}")
    print(f"Image : {args.imgsz}")
    print(f"Batch : {args.batch}")
    print(f"Device: {args.device}")
    print("=" * 70)

    model = YOLO(str(model_path))

    # ------------------------------------------------------------------
    # Build validation arguments
    # ------------------------------------------------------------------

    eval_args = {
        "data": str(data_yaml),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "plots": True,
        "iou": args.iou,
        "max_det": args.max_det,
        "save_json": args.save_json,
        "augment": args.augment,
        "verbose": True,
    }

    if args.device is not None:
        eval_args["device"] = args.device

    if args.conf is not None:
        eval_args["conf"] = args.conf

    # ------------------------------------------------------------------
    # Run evaluation
    # ------------------------------------------------------------------

    metrics = model.val(**eval_args)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)

    try:
        print(f"mAP@0.50     : {metrics.box.map50:.4f}")
        print(f"mAP@0.50:0.95: {metrics.box.map:.4f}")
        print(f"Precision     : {metrics.box.mp:.4f}")
        print(f"Recall        : {metrics.box.mr:.4f}")
    except AttributeError:
        print("Metrics returned by Ultralytics:")
        print(metrics)

    print("=" * 70)
    print("Evaluation complete.")


if __name__ == "__main__":
    main()