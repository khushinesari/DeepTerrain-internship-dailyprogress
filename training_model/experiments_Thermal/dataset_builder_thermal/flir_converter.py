"""
flir_converter.py

Convert Official FLIR ADAS Thermal Dataset
to YOLO format using the common

person
vehicle
animal

taxonomy.
"""

import json
import shutil

from pathlib import Path
from collections import defaultdict

from config import (
    FLIR_ROOT,
    FLIR_OUTPUT,
    FLIR_SPLITS,
    KEEP_NEGATIVE_IMAGES,
)

from class_mapping import (
    CLASS_MAPPING,
    YOLO_CLASSES
)

from utils import ensure_dir
IMAGE_DIR = FLIR_OUTPUT / "images"
LABEL_DIR = FLIR_OUTPUT / "labels"

ensure_dir(IMAGE_DIR)
ensure_dir(LABEL_DIR)
def coco_to_yolo(box, width, height):

    x, y, w, h = box

    x = max(0, x)
    y = max(0, y)

    w = min(w, width - x)
    h = min(h, height - y)

    xc = (x + w / 2) / width
    yc = (y + h / 2) / height

    return xc, yc, w / width, h / height

def load_coco(json_file):

    with open(json_file) as f:

        return json.load(f)

def build_image_lookup(coco):

    lookup = {}

    for image in coco["images"]:

        lookup[image["id"]] = image

    return lookup

def build_annotation_lookup(coco):

    grouped = defaultdict(list)

    for ann in coco["annotations"]:

        grouped[
            ann["image_id"]
        ].append(ann)

    return grouped

def build_category_lookup(coco):

    lookup = {}

    for cat in coco["categories"]:

        name = cat["name"].lower().strip()

        if name not in CLASS_MAPPING:
            continue

        merged = CLASS_MAPPING[name]

        lookup[
            cat["id"]
        ] = YOLO_CLASSES[merged]

    return lookup

class DatasetStats:

    def __init__(self):

        self.images = 0
        self.labels = 0
        self.objects = 0
        self.empty = 0
        self.skipped = 0

    def print(self):

        print()

        print("="*60)

        print("FLIR Dataset Summary")

        print("="*60)

        print(f"Images     : {self.images}")
        print(f"Labels     : {self.labels}")
        print(f"Objects    : {self.objects}")
        print(f"Negatives  : {self.empty}")
        print(f"Skipped    : {self.skipped}")

        print("="*60)

def convert_split(split, stats):

    print("\n" + "=" * 60)
    print(f"Processing {split}")
    print("=" * 60)

    split_root = FLIR_ROOT / split

    coco_file = split_root / "coco.json"

    if not coco_file.exists():

        print(f"[ERROR] Missing {coco_file}")

        return

    coco = load_coco(coco_file)

    image_lookup = build_image_lookup(coco)

    annotation_lookup = build_annotation_lookup(coco)

    category_lookup = build_category_lookup(coco)

    print(f"Images      : {len(image_lookup)}")
    print(f"Annotations : {len(coco['annotations'])}")

    counter = 1

    for image in coco["images"]:

        image_id = image["id"]

        width = image["width"]

        height = image["height"]

        # ---------------------------------------------------------
        # IMPORTANT
        # file_name already contains "data/..."
        #
        # Example:
        #
        # data/video-XXXX-frame-00123.jpg
        #
        # Therefore:
        #
        # split_root / file_name
        #
        # NOT
        #
        # split_root/data/file_name
        # ---------------------------------------------------------

        src = split_root / image["file_name"]

        if not src.exists():

            # fallback search

            found = list(
                split_root.rglob(
                    Path(image["file_name"]).name
                )
            )

            if len(found):

                src = found[0]

            else:

                stats.skipped += 1

                continue

        ext = src.suffix.lower()

        new_name = (
            f"{split}_{counter:06d}{ext}"
        )

        shutil.copy2(
            src,
            IMAGE_DIR / new_name
        )

        label_file = (
            LABEL_DIR /
            (Path(new_name).stem + ".txt")
        )

        valid = []

        for ann in annotation_lookup.get(image_id, []):

            if ann["category_id"] not in category_lookup:

                continue

            valid.append(ann)

        # -----------------------------------------
        # Negative Image
        # -----------------------------------------

        if len(valid) == 0:
            if KEEP_NEGATIVE_IMAGES:

                label_file.touch()

                stats.images += 1
                stats.labels += 1
                stats.empty += 1
            else:
                (IMAGE_DIR / new_name).unlink(missing_ok=True)
                stats.skipped += 1

            counter += 1

            continue

        # -----------------------------------------

        with open(label_file, "w") as out:

            for ann in valid:

                cls = category_lookup[
                    ann["category_id"]
                ]

                xc, yc, w, h = coco_to_yolo(

                    ann["bbox"],
                    width,
                    height
                )

                out.write(

                    f"{cls} "

                    f"{xc:.6f} "

                    f"{yc:.6f} "

                    f"{w:.6f} "

                    f"{h:.6f}\n"

                )

                stats.objects += 1

        stats.images += 1

        stats.labels += 1

        counter += 1

    print(

        f"Finished {split}"

    )

    print(

        f"Images written : {counter-1}"

    )

# -------------------------------------------------------------
# Save class mapping
# -------------------------------------------------------------

def save_class_mapping():

    reverse = {}

    for name, idx in YOLO_CLASSES.items():

        reverse[idx] = name

    with open(
        FLIR_OUTPUT / "class_mapping.json",
        "w"
    ) as f:

        json.dump(
            reverse,
            f,
            indent=4
        )


# -------------------------------------------------------------
# Validation
# -------------------------------------------------------------

def validate_dataset():

    print("\nValidating FLIR dataset...")

    images = {
        p.stem
        for p in IMAGE_DIR.iterdir()
    }

    labels = {
        p.stem
        for p in LABEL_DIR.iterdir()
    }

    missing = images - labels

    if missing:

        print(f"Missing labels : {len(missing)}")

        for item in sorted(missing):

            print(item)

    else:

        print("✓ Every image has a label file.")

# -------------------------------------------------------------

def convert_flir_dataset():

    ensure_dir(IMAGE_DIR)
    ensure_dir(LABEL_DIR)

    stats = DatasetStats()

    print("\n" + "=" * 60)
    print("Converting FLIR COCO -> YOLO")
    print("=" * 60)

    for split in FLIR_SPLITS:

        convert_split(
            split,
            stats
        )

    save_class_mapping()

    validate_dataset()

    stats.print()

    print("\n" + "=" * 60)
    print("FLIR Conversion Completed")
    print("=" * 60)

# -------------------------------------------------------------

if __name__ == "__main__":

    convert_flir_dataset()
