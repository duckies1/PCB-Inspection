#!/usr/bin/env python3
"""Train or fine-tune YOLO26s on the PKU PCB defect dataset."""

import argparse
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = PROJECT_DIR / "compiled_datasets" / "PKU" / "data.yaml"
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

    for split in ("train", "val", "test"):
        split_path = dataset_root / config.get(split, split)
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"{split} split directory not found: {split_path}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to a YOLO dataset YAML (default: PKU data.yaml)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_DIR / "yolo26s.pt",
        help="Pretrained weights or a model checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Batch size; -1 lets Ultralytics auto-select it",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device such as 0, multiple devices such as 0,1, or cpu",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="pku_yolo26s")
    parser.add_argument("--project", type=Path, default=PROJECT_DIR / "runs" / "detect")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint supplied with --model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    model_path = args.model.resolve()

    validate_dataset(data_yaml)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    train_args = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "pretrained": True,
        "plots": True,
    }
    if args.device is not None:
        train_args["device"] = args.device
    if args.resume:
        train_args["resume"] = True

    model.train(**train_args)


if __name__ == "__main__":
    main()