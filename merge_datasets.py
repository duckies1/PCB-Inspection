#!/usr/bin/env python3

"""
General YOLO Dataset Merger

Merges multiple YOLO-format datasets into one.

Example:

python merge_datasets.py \
    --datasets PKU Strathclyde \
    --output merged_dataset

Expected dataset structure:

dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

The script also works if a dataset only has train/images + train/labels.

Output:

merged_dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
"""

from pathlib import Path
import argparse
import shutil
import yaml


# ============================================================
# CONFIG
# ============================================================

CLASS_NAMES = [
    "short",
    "spur",
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "spurious_copper",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# FIND DATASET SPLITS
# ============================================================

def find_split_dirs(dataset_path):
    """
    Find train/val/test directories.

    If no standard split directories are found, assume
    the dataset itself is a single split.
    """

    splits = {}

    for split in ["train", "val", "test"]:

        split_path = dataset_path / split

        if split_path.exists():
            splits[split] = split_path

    # Dataset itself may directly contain images/labels
    if not splits:

        if (
            (dataset_path / "images").exists()
            and
            (dataset_path / "labels").exists()
        ):
            splits["train"] = dataset_path

    return splits


# ============================================================
# FIND IMAGE FILES
# ============================================================

def find_images(images_dir):

    images = []

    for path in images_dir.rglob("*"):

        if (
            path.is_file()
            and
            path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


# ============================================================
# FIND CORRESPONDING LABEL
# ============================================================

def find_label(image_path, images_dir, labels_dir):

    relative_path = image_path.relative_to(images_dir)

    label_path = (
        labels_dir
        / relative_path.with_suffix(".txt")
    )

    return label_path


# ============================================================
# COPY DATASET
# ============================================================

def merge_dataset(
    dataset_path,
    dataset_name,
    output_path,
):

    print("\n" + "=" * 60)
    print(f"Processing: {dataset_name}")
    print(f"Path:       {dataset_path}")
    print("=" * 60)

    splits = find_split_dirs(dataset_path)

    if not splits:

        print(
            f"[WARNING] No valid YOLO structure found "
            f"in {dataset_path}"
        )

        return {
            "train": 0,
            "val": 0,
            "test": 0,
        }

    counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for split, split_dir in splits.items():

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():

            print(
                f"[WARNING] Missing images directory: "
                f"{images_dir}"
            )

            continue

        if not labels_dir.exists():

            print(
                f"[WARNING] Missing labels directory: "
                f"{labels_dir}"
            )

        # Output directories
        output_images = (
            output_path
            / split
            / "images"
        )

        output_labels = (
            output_path
            / split
            / "labels"
        )

        output_images.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_labels.mkdir(
            parents=True,
            exist_ok=True,
        )

        images = find_images(images_dir)

        print(
            f"{split}: {len(images)} images"
        )

        for image in images:

            # Preserve subdirectory structure
            relative = image.relative_to(images_dir)

            # Prefix filename to prevent collisions
            new_name = (
                f"{dataset_name}_"
                f"{relative.stem}"
                f"{relative.suffix.lower()}"
            )

            output_image = (
                output_images
                / new_name
            )

            # Corresponding label
            label = find_label(
                image,
                images_dir,
                labels_dir,
            )

            new_label = (
                output_labels
                / f"{dataset_name}_{relative.stem}.txt"
            )

            # Handle duplicate filenames inside
            # subdirectories
            counter = 1

            while output_image.exists():

                new_name = (
                    f"{dataset_name}_"
                    f"{relative.stem}_"
                    f"{counter}"
                    f"{relative.suffix.lower()}"
                )

                output_image = (
                    output_images
                    / new_name
                )

                new_label = (
                    output_labels
                    / f"{dataset_name}_"
                    f"{relative.stem}_"
                    f"{counter}.txt"
                )

                counter += 1

            # Copy image
            shutil.copy2(
                image,
                output_image,
            )

            # Copy label if it exists
            if label.exists():

                shutil.copy2(
                    label,
                    new_label,
                )

            else:

                # Create empty label file
                # for images without annotations
                new_label.touch()

                print(
                    f"[WARNING] No label found for:"
                    f" {image}"
                )

            counts[split] += 1

    return counts


# ============================================================
# CREATE DATA.YAML
# ============================================================

def create_yaml(output_path):

    data = {
        "path": str(output_path.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }

    yaml_path = output_path / "data.yaml"

    with yaml_path.open("w") as f:

        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
        )

    print(
        f"\nCreated data.yaml:"
        f"\n{yaml_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Merge multiple YOLO datasets."
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Paths to datasets to merge",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory",
    )

    args = parser.parse_args()

    datasets = [
        Path(path)
        for path in args.datasets
    ]

    # --------------------------------------------------------
    # Validate input datasets
    # --------------------------------------------------------

    for dataset in datasets:

        if not dataset.exists():

            raise FileNotFoundError(
                f"Dataset not found: {dataset}"
            )

    # --------------------------------------------------------
    # Prepare output directory
    # --------------------------------------------------------

    if args.output.exists():

        print(
            f"\nWARNING:"
            f"\nOutput directory already exists:"
            f"\n{args.output}"
        )

        response = input(
            "\nDelete it and start again? [y/N]: "
        )

        if response.lower() != "y":

            print("Aborted.")

            return

        shutil.rmtree(args.output)

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Merge datasets
    # --------------------------------------------------------

    total_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for dataset in datasets:

        dataset_name = dataset.name.lower()

        counts = merge_dataset(
            dataset_path=dataset,
            dataset_name=dataset_name,
            output_path=args.output,
        )

        for split in total_counts:

            total_counts[split] += counts[split]

    # --------------------------------------------------------
    # Create YAML
    # --------------------------------------------------------

    create_yaml(args.output)

    # --------------------------------------------------------
    # Print final statistics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MERGE COMPLETE")
    print("=" * 60)

    print(
        f"Train images : {total_counts['train']}"
    )

    print(
        f"Val images   : {total_counts['val']}"
    )

    print(
        f"Test images  : {total_counts['test']}"
    )

    print(
        f"Total        : "
        f"{sum(total_counts.values())}"
    )

    print(
        f"\nOutput:"
        f"\n{args.output.resolve()}"
    )

    print(
        f"\nData YAML:"
        f"\n{(args.output / 'data.yaml').resolve()}"
    )


if __name__ == "__main__":
    main()