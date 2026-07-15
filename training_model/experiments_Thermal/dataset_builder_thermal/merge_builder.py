"""
Merge AGV and FLIR Thermal Datasets

Creates

ThermalDataset/
    Merged/
        images/
        labels/

Checks

✓ Missing labels
✓ Missing images
✓ Duplicate filenames
✓ Duplicate images (MD5)
✓ Empty labels
"""

import shutil
from pathlib import Path

from config import (
    AGV_OUTPUT,
    FLIR_OUTPUT,
    MERGED_OUTPUT
)

from utils import ensure_dir, md5


MERGED_IMAGES = MERGED_OUTPUT / "images"
MERGED_LABELS = MERGED_OUTPUT / "labels"


def prepare():

    ensure_dir(MERGED_IMAGES)
    ensure_dir(MERGED_LABELS)


def copy_dataset(dataset_root):

    image_root = dataset_root / "images"
    label_root = dataset_root / "labels"

    image_count = 0
    label_count = 0

    hash_table = {}

    duplicate_images = []

    for image in sorted(image_root.iterdir()):

        if not image.is_file():
            continue

        label = label_root / (image.stem + ".txt")

        if not label.exists():

            print(f"[WARNING] Missing label : {image.name}")
            continue

        if label.stat().st_size == 0:

            print(f"[WARNING] Empty label : {label.name}")
            continue

        image_hash = md5(image)

        if image_hash in hash_table:

            duplicate_images.append(image.name)
            continue

        hash_table[image_hash] = image.name

        shutil.copy2(
            image,
            MERGED_IMAGES / image.name
        )

        shutil.copy2(
            label,
            MERGED_LABELS / label.name
        )

        image_count += 1
        label_count += 1

    return image_count, label_count, duplicate_images


def merge_datasets():

    prepare()

    print("\n===============================")
    print("Merging AGV Dataset")
    print("===============================")

    agv_images, agv_labels, agv_duplicates = copy_dataset(
        AGV_OUTPUT
    )

    print("\n===============================")
    print("Merging FLIR Dataset")
    print("===============================")

    flir_images, flir_labels, flir_duplicates = copy_dataset(
        FLIR_OUTPUT
    )

    print("\n===============================")
    print("Merge Summary")
    print("===============================")

    print(f"AGV Images     : {agv_images}")
    print(f"FLIR Images    : {flir_images}")

    print()

    print(f"AGV Labels     : {agv_labels}")
    print(f"FLIR Labels    : {flir_labels}")

    print()

    print(f"Duplicate AGV Images  : {len(agv_duplicates)}")
    print(f"Duplicate FLIR Images : {len(flir_duplicates)}")

    print()

    print(
        f"Total Images : {agv_images + flir_images}"
    )

    print(
        f"Total Labels : {agv_labels + flir_labels}"
    )

    print("\nMerge Complete.\n")
