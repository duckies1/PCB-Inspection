#!/usr/bin/env python3
"""
Generic YOLO Dataset Class-ID Remapper

Reads a YOLO dataset's data.yaml, determines its current class-ID
ordering, and remaps all YOLO annotation files to a target class
ordering.

The original dataset is never modified.

Example:

    python3 remap_yolo_classes.py \
        --data compiled_datasets/Roboflow/Strathclyde_University_Dataset/data.yaml \
        --target-names mouse_bite spur missing_hole short open_circuit spurious_copper

Or using a target YAML:

    python3 remap_yolo_classes.py \
        --data source/data.yaml \
        --target-yaml PKU/data.yaml

"""

from pathlib import Path
import argparse
import shutil
import sys

import yaml


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


SPLIT_NAMES = {
    "train",
    "val",
    "test",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Remap YOLO dataset class IDs according to data.yaml."
    )

    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to the source dataset data.yaml",
    )

    parser.add_argument(
        "--target-yaml",
        type=Path,
        default=None,
        help=(
            "Optional target data.yaml. "
            "Its class ordering will be used as the target ordering."
        ),
    )

    parser.add_argument(
        "--target-names",
        nargs="+",
        default=None,
        help=(
            "Target class ordering. "
            "Example: mouse_bite spur missing_hole short open_circuit spurious_copper"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Default: <source_dataset>_remapped"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow an existing output directory to be deleted.",
    )

    return parser.parse_args()


# ============================================================
# YAML HELPERS
# ============================================================

def load_yaml(path: Path) -> dict:

    if not path.is_file():

        raise FileNotFoundError(
            f"YAML file does not exist:\n{path}"
        )

    with path.open("r") as f:

        data = yaml.safe_load(f)

    if not isinstance(data, dict):

        raise ValueError(
            f"Invalid YAML structure:\n{path}"
        )

    return data


def extract_class_names(data: dict, yaml_path: Path) -> list[str]:

    names = data.get("names")

    if names is None:

        raise ValueError(
            f"No 'names' field found in:\n{yaml_path}"
        )

    # YOLO YAML may store names as either:
    #
    # names:
    #   0: mouse_bite
    #   1: spur
    #
    # OR:
    #
    # names:
    #   - mouse_bite
    #   - spur

    if isinstance(names, list):

        result = names

    elif isinstance(names, dict):

        try:

            result = [
                names[int(key)]
                for key in sorted(names, key=lambda x: int(x))
            ]

        except (ValueError, KeyError):

            raise ValueError(
                f"Could not parse class names in:\n{yaml_path}"
            )

    else:

        raise ValueError(
            f"Unsupported 'names' format in:\n{yaml_path}"
        )

    result = [str(name) for name in result]

    if len(result) == 0:

        raise ValueError(
            f"No classes found in:\n{yaml_path}"
        )

    return result


def resolve_dataset_root(
    yaml_data: dict,
    yaml_path: Path,
) -> Path:

    dataset_path = yaml_data.get("path")

    if dataset_path is None:

        # If no path is specified, use directory containing YAML.
        return yaml_path.parent.resolve()

    dataset_path = Path(dataset_path)

    if dataset_path.is_absolute():

        return dataset_path.resolve()

    return (yaml_path.parent / dataset_path).resolve()


# ============================================================
# TARGET ORDER
# ============================================================

def get_target_names(args) -> list[str]:

    if args.target_yaml is not None:

        target_data = load_yaml(args.target_yaml)

        target_names = extract_class_names(
            target_data,
            args.target_yaml,
        )

        print(
            f"Target class ordering loaded from:\n"
            f"  {args.target_yaml}"
        )

        return target_names

    if args.target_names is not None:

        return args.target_names

    raise ValueError(
        "You must provide either --target-yaml or --target-names."
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_class_lists(
    source_names: list[str],
    target_names: list[str],
):

    if len(source_names) != len(target_names):

        raise ValueError(
            "\nNumber of classes differs!\n\n"
            f"Source: {len(source_names)} classes\n"
            f"Target: {len(target_names)} classes\n"
        )

    source_set = set(source_names)
    target_set = set(target_names)

    if source_set != target_set:

        missing_from_target = source_set - target_set
        missing_from_source = target_set - source_set

        message = "\nClass sets do not match!\n"

        if missing_from_target:

            message += (
                "\nClasses present in source but not target:\n"
                + "\n".join(
                    f"  - {x}" for x in sorted(missing_from_target)
                )
            )

        if missing_from_source:

            message += (
                "\nClasses present in target but not source:\n"
                + "\n".join(
                    f"  - {x}" for x in sorted(missing_from_source)
                )
            )

        raise ValueError(message)

    if len(set(source_names)) != len(source_names):

        raise ValueError(
            "Source dataset contains duplicate class names."
        )

    if len(set(target_names)) != len(target_names):

        raise ValueError(
            "Target class ordering contains duplicate class names."
        )


# ============================================================
# MAPPING
# ============================================================

def create_mapping(
    source_names: list[str],
    target_names: list[str],
) -> dict[int, int]:

    target_ids = {
        name: index
        for index, name in enumerate(target_names)
    }

    mapping = {}

    for source_id, class_name in enumerate(source_names):

        mapping[source_id] = target_ids[class_name]

    return mapping


def print_mapping(
    source_names: list[str],
    target_names: list[str],
    mapping: dict[int, int],
):

    print()
    print("=" * 75)
    print("CLASS ID MAPPING")
    print("=" * 75)

    print(
        f"{'Old ID':<10}"
        f"{'Class':<25}"
        f"{'New ID':<10}"
        f"{'Status'}"
    )

    print("-" * 75)

    for old_id, class_name in enumerate(source_names):

        new_id = mapping[old_id]

        if old_id == new_id:

            status = "UNCHANGED"

        else:

            status = "REMAPPED"

        print(
            f"{old_id:<10}"
            f"{class_name:<25}"
            f"{new_id:<10}"
            f"{status}"
        )


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_label_file(
    source_file: Path,
    output_file: Path,
    mapping: dict[int, int],
    stats: dict,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_lines = []

    with source_file.open("r") as f:

        lines = f.readlines()

    for line_number, line in enumerate(lines, start=1):

        line = line.strip()

        if not line:

            continue

        parts = line.split()

        # Standard YOLO format:
        #
        # class x_center y_center width height
        #
        if len(parts) < 5:

            raise ValueError(
                f"\nMalformed YOLO annotation:\n"
                f"File: {source_file}\n"
                f"Line: {line_number}\n"
                f"Content: {line}"
            )

        try:

            old_id = int(parts[0])

        except ValueError:

            raise ValueError(
                f"\nInvalid class ID:\n"
                f"File: {source_file}\n"
                f"Line: {line_number}\n"
                f"Content: {line}"
            )

        if old_id not in mapping:

            raise ValueError(
                f"\nUnknown class ID {old_id}:\n"
                f"File: {source_file}\n"
                f"Line: {line_number}\n"
                f"Content: {line}"
            )

        new_id = mapping[old_id]

        parts[0] = str(new_id)

        output_lines.append(" ".join(parts))

        stats["objects"] += 1
        stats["old"][old_id] += 1
        stats["new"][new_id] += 1

    with output_file.open("w") as f:

        if output_lines:

            f.write("\n".join(output_lines))
            f.write("\n")


# ============================================================
# SPLIT HANDLING
# ============================================================

def find_split_directory(
    dataset_root: Path,
    split_name: str,
) -> Path | None:

    candidates = []

    if split_name == "val":

        candidates = [
            dataset_root / "val",
            dataset_root / "valid",
            dataset_root / "validation",
        ]

    else:

        candidates = [
            dataset_root / split_name
        ]

    for candidate in candidates:

        if candidate.exists():

            return candidate

    return None


def copy_images(
    source_dir: Path,
    output_dir: Path,
) -> int:

    if not source_dir.exists():

        return 0

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    for source_file in source_dir.iterdir():

        if not source_file.is_file():

            continue

        if source_file.suffix.lower() not in IMAGE_EXTENSIONS:

            continue

        shutil.copy2(
            source_file,
            output_dir / source_file.name,
        )

        count += 1

    return count


def process_split(
    dataset_root: Path,
    output_root: Path,
    split_name: str,
    mapping: dict[int, int],
    stats: dict,
):

    source_split = find_split_directory(
        dataset_root,
        split_name,
    )

    if source_split is None:

        print(
            f"\nWARNING: '{split_name}' split not found."
        )

        return

    output_split = output_root / split_name

    source_images = source_split / "images"
    source_labels = source_split / "labels"

    output_images = output_split / "images"
    output_labels = output_split / "labels"

    print()
    print("=" * 75)
    print(f"PROCESSING {split_name.upper()}")
    print("=" * 75)

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    image_count = copy_images(
        source_images,
        output_images,
    )

    print(f"Images copied : {image_count}")

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    if not source_labels.exists():

        print(
            f"WARNING: labels directory not found:\n"
            f"  {source_labels}"
        )

        return

    label_files = list(
        source_labels.glob("*.txt")
    )

    print(
        f"Labels found  : {len(label_files)}"
    )

    for source_label in label_files:

        output_label = (
            output_labels /
            source_label.name
        )

        convert_label_file(
            source_label,
            output_label,
            mapping,
            stats,
        )

    print(
        f"Labels converted: {len(label_files)}"
    )


# ============================================================
# YAML CREATION
# ============================================================

def create_output_yaml(
    output_root: Path,
    target_names: list[str],
):

    output_yaml = output_root / "data.yaml"

    data = {
        "path": str(output_root),
        "train": "train",
        "val": "val",
        "test": "test",
        "nc": len(target_names),
        "names": {
            index: name
            for index, name in enumerate(target_names)
        },
    }

    with output_yaml.open("w") as f:

        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    return output_yaml


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    source_names: list[str],
    target_names: list[str],
    stats: dict,
):

    print()
    print("=" * 75)
    print("CONVERSION SUMMARY")
    print("=" * 75)

    print(
        f"\nTotal objects processed: "
        f"{stats['objects']}"
    )

    print("\nClass counts:")

    print(
        f"{'ID':<6}"
        f"{'Class':<25}"
        f"{'Objects'}"
    )

    print("-" * 50)

    for new_id, name in enumerate(target_names):

        print(
            f"{new_id:<6}"
            f"{name:<25}"
            f"{stats['new'][new_id]}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Source YAML
    # --------------------------------------------------------

    source_yaml = args.data.resolve()

    source_data = load_yaml(
        source_yaml
    )

    source_names = extract_class_names(
        source_data,
        source_yaml,
    )

    dataset_root = resolve_dataset_root(
        source_data,
        source_yaml,
    )

    # --------------------------------------------------------
    # Target ordering
    # --------------------------------------------------------

    if args.target_yaml:
        target_names = get_target_names(
            args
        )
    else:
        target_names = ['mouse_bite', 'spur', 'missing_hole', 'short', 'open_circuit', 'spurious_copper']

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_class_lists(
        source_names,
        target_names,
    )

    # --------------------------------------------------------
    # Create mapping
    # --------------------------------------------------------

    mapping = create_mapping(
        source_names,
        target_names,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    if args.output:

        output_root = args.output.resolve()

    else:

        output_root = Path(
            str(dataset_root) + "_remapped"
        )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("=" * 75)
    print("YOLO DATASET CLASS-ID REMAPPER")
    print("=" * 75)

    print(
        f"\nSource YAML:\n  {source_yaml}"
    )

    print(
        f"\nSource dataset:\n  {dataset_root}"
    )

    print(
        f"\nOutput dataset:\n  {output_root}"
    )

    print(
        f"\nSource classes:\n  {source_names}"
    )

    print(
        f"\nTarget classes:\n  {target_names}"
    )

    print()
    print("Class mapping:")
    print_mapping(
        source_names,
        target_names,
        mapping,
    )

    # --------------------------------------------------------
    # Check whether anything actually needs changing
    # --------------------------------------------------------

    if all(
        mapping[i] == i
        for i in range(len(mapping))
    ):

        print()
        print(
            "NOTE: Source and target class ordering are "
            "already identical."
        )

    # --------------------------------------------------------
    # Output handling
    # --------------------------------------------------------

    if output_root.exists():

        if not args.force:

            print()
            print(
                "ERROR: Output directory already exists:"
            )
            print(
                f"  {output_root}"
            )
            print()
            print(
                "Use --force to replace it."
            )

            sys.exit(1)

        print(
            f"\nRemoving existing output directory:"
            f"\n  {output_root}"
        )

        shutil.rmtree(output_root)

    output_root.mkdir(
        parents=True
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "objects": 0,
        "old": {
            i: 0
            for i in range(len(source_names))
        },
        "new": {
            i: 0
            for i in range(len(target_names))
        },
    }

    # --------------------------------------------------------
    # Process splits
    # --------------------------------------------------------

    for split in SPLIT_NAMES:

        process_split(
            dataset_root,
            output_root,
            split,
            mapping,
            stats,
        )

    # --------------------------------------------------------
    # Create YAML
    # --------------------------------------------------------

    output_yaml = create_output_yaml(
        output_root,
        target_names,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        source_names,
        target_names,
        stats,
    )

    print()
    print("=" * 75)
    print("DONE")
    print("=" * 75)

    print(
        f"\nConverted dataset:"
        f"\n  {output_root}"
    )

    print(
        f"\nGenerated data.yaml:"
        f"\n  {output_yaml}"
    )

    print(
        "\nOriginal dataset was NOT modified."
    )


if __name__ == "__main__":
    main()