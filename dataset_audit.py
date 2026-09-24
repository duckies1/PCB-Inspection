#!/usr/bin/env python3

"""
Audit a YOLO object-detection dataset for:

1. Exact duplicate images
2. Near-duplicate images using perceptual hashing
3. Train/validation leakage
4. Image statistics
5. YOLO annotation statistics
6. Class distribution
7. Images with no annotations
8. Very small / unusual images

Usage:

    python dataset_audit.py /path/to/dataset

Example:

    python dataset_audit.py ./compiled_datasets/PKU

Dependencies:

    pip install pillow imagehash pandas tqdm
"""

from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json

import pandas as pd
from PIL import Image
import imagehash
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

LABEL_EXTENSIONS = {".txt"}

# Maximum Hamming distance between perceptual hashes
# for considering two images "near duplicates".
#
# Lower = stricter
#
# 0     = effectively identical perceptual hash
# 1-4   = extremely similar
# 5-8   = very similar
# 9-12  = somewhat similar
#
NEAR_DUPLICATE_THRESHOLD = 6

# Images smaller than this are flagged.
MIN_WIDTH = 256
MIN_HEIGHT = 256


# ============================================================
# HELPERS
# ============================================================

def sha256_file(path):
    """Calculate SHA256 hash of a file."""

    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def get_image_files(root):
    """Recursively find images."""

    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_split(path):
    """
    Try to determine whether an image belongs to train/val/test.
    """

    parts = [p.lower() for p in path.parts]

    if "train" in parts:
        return "train"

    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"

    if "test" in parts:
        return "test"

    return "unknown"


def get_label_path(image_path):
    """
    Convert:

        image.jpg

    into:

        image.txt

    """

    return image_path.with_suffix(".txt")


def read_labels(label_path):

    if not label_path.exists():
        return []

    labels = []

    with open(label_path, "r") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) >= 5:

                try:
                    cls = int(parts[0])
                    labels.append(cls)

                except ValueError:
                    pass

    return labels


def calculate_statistics(image_path):

    try:

        with Image.open(image_path) as img:

            width, height = img.size

            # Convert to RGB for statistics
            rgb = img.convert("RGB")

            # Resize to make statistics cheap
            small = rgb.resize((64, 64))

            pixels = list(small.getdata())

            mean_r = sum(p[0] for p in pixels) / len(pixels)
            mean_g = sum(p[1] for p in pixels) / len(pixels)
            mean_b = sum(p[2] for p in pixels) / len(pixels)

            return {
                "width": width,
                "height": height,
                "aspect_ratio": width / height,
                "mean_r": mean_r,
                "mean_g": mean_g,
                "mean_b": mean_b,
                "mean_brightness": (
                    0.299 * mean_r
                    + 0.587 * mean_g
                    + 0.114 * mean_b
                ),
            }

    except Exception as e:

        return {
            "error": str(e)
        }


# ============================================================
# MAIN AUDIT
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to YOLO dataset root"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=NEAR_DUPLICATE_THRESHOLD,
        help="Perceptual hash Hamming distance threshold"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_audit"),
        help="Output directory"
    )

    args = parser.parse_args()

    root = args.dataset.resolve()
    output = args.output

    output.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("PCB DATASET AUDIT")
    print("=" * 70)

    print(f"Dataset: {root}")
    print()

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = get_image_files(root)

    print(f"Images found: {len(images)}")
    print()

    if not images:

        print("ERROR: No images found.")
        return

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    records = []

    exact_hashes = defaultdict(list)

    phashes = {}

    class_counts = Counter()

    split_counts = Counter()

    no_annotation = []

    errors = []

    print("Processing images...")

    for image_path in tqdm(images):

        split = find_split(image_path)

        split_counts[split] += 1

        label_path = get_label_path(image_path)

        labels = read_labels(label_path)

        if not labels:
            no_annotation.append(str(image_path))

        for cls in labels:
            class_counts[cls] += 1

        # Exact hash
        try:

            sha = sha256_file(image_path)

            exact_hashes[sha].append(str(image_path))

        except Exception as e:

            errors.append({
                "image": str(image_path),
                "error": str(e)
            })

            continue

        # Perceptual hash
        try:

            with Image.open(image_path) as img:

                img = img.convert("RGB")

                phash = imagehash.phash(img)

                phashes[str(image_path)] = phash

        except Exception as e:

            errors.append({
                "image": str(image_path),
                "error": str(e)
            })

            continue

        stats = calculate_statistics(image_path)

        records.append({
            "image": str(image_path),
            "split": split,
            "width": stats.get("width"),
            "height": stats.get("height"),
            "aspect_ratio": stats.get("aspect_ratio"),
            "mean_r": stats.get("mean_r"),
            "mean_g": stats.get("mean_g"),
            "mean_b": stats.get("mean_b"),
            "brightness": stats.get("mean_brightness"),
            "annotations": len(labels),
        })

    # --------------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------------

    duplicate_groups = [
        paths
        for paths in exact_hashes.values()
        if len(paths) > 1
    ]

    duplicate_count = sum(
        len(group) - 1
        for group in duplicate_groups
    )

    print()
    print("=" * 70)
    print("1. EXACT DUPLICATES")
    print("=" * 70)

    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Duplicate images: {duplicate_count}")

    if duplicate_groups:

        print()
        print("Examples:")

        for group in duplicate_groups[:10]:

            print()

            for path in group:
                print("  ", path)

    # --------------------------------------------------------
    # Near duplicates
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2. NEAR DUPLICATES")
    print("=" * 70)

    print(
        f"Searching using perceptual hash "
        f"(threshold <= {args.threshold})..."
    )

    image_paths = list(phashes.keys())

    near_duplicates = []

    # Simple O(N²) comparison.
    #
    # This is intentionally straightforward.
    # For a few thousand images it is generally acceptable.
    #
    # If your dataset becomes >10k images, use an indexed method.

    for i in tqdm(range(len(image_paths))):

        path_a = image_paths[i]

        hash_a = phashes[path_a]

        for j in range(i + 1, len(image_paths)):

            path_b = image_paths[j]

            hash_b = phashes[path_b]

            distance = hash_a - hash_b

            if distance <= args.threshold:

                near_duplicates.append({
                    "image_a": path_a,
                    "image_b": path_b,
                    "hamming_distance": distance,
                    "split_a": find_split(Path(path_a)),
                    "split_b": find_split(Path(path_b)),
                    "cross_split": (
                        find_split(Path(path_a))
                        != find_split(Path(path_b))
                    ),
                })

    print()

    print(
        f"Near-duplicate pairs: "
        f"{len(near_duplicates)}"
    )

    cross_split_duplicates = [
        x
        for x in near_duplicates
        if x["cross_split"]
    ]

    print(
        f"Cross train/val/test near-duplicates: "
        f"{len(cross_split_duplicates)}"
    )

    # --------------------------------------------------------
    # Split leakage
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3. TRAIN / VALIDATION LEAKAGE")
    print("=" * 70)

    if cross_split_duplicates:

        print()
        print("Potential leakage detected:")
        print()

        for item in cross_split_duplicates[:30]:

            print(
                f"[distance={item['hamming_distance']}]"
            )

            print("  A:", item["image_a"])
            print("  B:", item["image_b"])
            print()

    else:

        print("No cross-split near-duplicates detected.")

    # --------------------------------------------------------
    # Image statistics
    # --------------------------------------------------------

    df = pd.DataFrame(records)

    df.to_csv(
        output / "image_statistics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Small images
    # --------------------------------------------------------

    small_images = df[
        (df["width"] < MIN_WIDTH)
        | (df["height"] < MIN_HEIGHT)
    ]

    print()
    print("=" * 70)
    print("4. IMAGE SIZE")
    print("=" * 70)

    print(
        f"Resolution range: "
        f"{df['width'].min()}x{df['height'].min()} "
        f"to "
        f"{df['width'].max()}x{df['height'].max()}"
    )

    print(
        f"Small images (< {MIN_WIDTH}x{MIN_HEIGHT}): "
        f"{len(small_images)}"
    )

    # --------------------------------------------------------
    # Dataset splits
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("5. DATASET SPLITS")
    print("=" * 70)

    for split, count in split_counts.items():

        print(f"{split:10s}: {count}")

    # --------------------------------------------------------
    # Annotation statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("6. ANNOTATION STATISTICS")
    print("=" * 70)

    print(
        f"Images with zero annotations: "
        f"{len(no_annotation)}"
    )

    print()
    print("Object instances per class:")

    for cls, count in sorted(class_counts.items()):

        print(
            f"Class {cls}: {count}"
        )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("7. IMAGE DIVERSITY / BASIC STATISTICS")
    print("=" * 70)

    print(
        f"Brightness mean: "
        f"{df['brightness'].mean():.2f}"
    )

    print(
        f"Brightness std: "
        f"{df['brightness'].std():.2f}"
    )

    print(
        f"Brightness min: "
        f"{df['brightness'].min():.2f}"
    )

    print(
        f"Brightness max: "
        f"{df['brightness'].max():.2f}"
    )

    print()

    print(
        f"Aspect ratio mean: "
        f"{df['aspect_ratio'].mean():.3f}"
    )

    print(
        f"Aspect ratio std: "
        f"{df['aspect_ratio'].std():.3f}"
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    with open(
        output / "exact_duplicates.json",
        "w"
    ) as f:

        # json.dump(
        #     duplicate_groups,
        #     f,
        #     indent=2
        # )
        json.dump(duplicate_groups, f, indent=2, default=lambda x: x.item())

    with open(
        output / "near_duplicates.json",
        "w"
    ) as f:

        # json.dump(
        #     near_duplicates,
        #     f,
        #     indent=2
        # )
        json.dump(near_duplicates, f, indent=2, default=lambda x: x.item())

    with open(
        output / "cross_split_duplicates.json",
        "w"
    ) as f:

        # json.dump(
        #     cross_split_duplicates,
        #     f,
        #     indent=2
        # )
        json.dump(cross_split_duplicates, f, indent=2, default=lambda x: x.item())

    with open(
        output / "no_annotations.txt",
        "w"
    ) as f:

        for path in no_annotation:
            f.write(path + "\n")

    with open(
        output / "errors.json",
        "w"
    ) as f:

        # json.dump(
        #     errors,
        #     f,
        #     indent=2
        # )
        json.dump(errors, f, indent=2, default=lambda x: x.item())
        

    print()
    print("=" * 70)
    print("REPORTS SAVED")
    print("=" * 70)

    print(f"Output directory: {output.resolve()}")

    print()
    print("Files:")

    print("  image_statistics.csv")
    print("  exact_duplicates.json")
    print("  near_duplicates.json")
    print("  cross_split_duplicates.json")
    print("  no_annotations.txt")
    print("  errors.json")

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()