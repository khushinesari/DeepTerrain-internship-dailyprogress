import json
import shutil
from pathlib import Path
from collections import defaultdict

from config import AGV_OUTPUT
from class_mapping import CLASS_MAPPING, YOLO_CLASSES
from utils import ensure_dir

IMAGE_DIR = AGV_OUTPUT / "images"
LABEL_DIR = AGV_OUTPUT / "labels"
ANNOTATION_DIR = AGV_OUTPUT / "annotations"
MAPPING_DIR = AGV_OUTPUT / "mappings"

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
def load_mapping(mapping_json):

    with open(mapping_json) as f:
        return json.load(f)
def load_coco(annotation_json):

    with open(annotation_json) as f:
        coco = json.load(f)

    return coco

def build_image_lookup(coco):

    images = {}

    for img in coco["images"]:

        images[img["id"]] = img

    return images

def build_annotation_lookup(coco):

    grouped = defaultdict(list)

    for ann in coco["annotations"]:

        grouped[ann["image_id"]].append(ann)

    return grouped

def build_category_lookup(coco):

    lookup = {}

    for cat in coco["categories"]:

        name = cat["name"].lower().strip()

        if name not in CLASS_MAPPING:
            continue

        merged = CLASS_MAPPING[name]

        lookup[cat["id"]] = YOLO_CLASSES[merged]

    return lookup

class DatasetStats:

    def __init__(self):

        self.images = 0
        self.labels = 0
        self.objects = 0
        self.skipped = 0
        self.empty = 0

    def print(self):

        print("\n==========================")

        print("AGV Conversion Summary")

        print("==========================")

        print(f"Images    : {self.images}")
        print(f"Labels    : {self.labels}")
        print(f"Objects   : {self.objects}")
        print(f"Skipped   : {self.skipped}")
        print(f"Negative  : {self.empty}")

        print("==========================\n")
def convert_task(annotation_json, mapping_json, stats):

    print(f"\nProcessing {annotation_json.stem}")

    coco = load_coco(annotation_json)

    filename_mapping = load_mapping(mapping_json)

    image_lookup = build_image_lookup(coco)

    annotation_lookup = build_annotation_lookup(coco)

    category_lookup = build_category_lookup(coco)

    for image in coco["images"]:

        original_name = Path(image["file_name"]).name

        if original_name not in filename_mapping:

            stats.skipped += 1
            continue

        new_image_name = filename_mapping[original_name]

        label_file = LABEL_DIR / (
            Path(new_image_name).stem + ".txt"
        )

        width = image["width"]
        height = image["height"]

        valid_annotations = []

        for ann in annotation_lookup.get(image["id"], []):

            if ann["category_id"] not in category_lookup:
                continue

            valid_annotations.append(ann)

        # ----------------------------------------------------
        # Negative Image
        # ----------------------------------------------------

        if len(valid_annotations) == 0:

            label_file.touch()

            stats.images += 1
            stats.labels += 1
            stats.empty += 1

            continue

        # ----------------------------------------------------

        with open(label_file, "w") as out:

            for ann in valid_annotations:

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

def convert_all_tasks():

    stats = DatasetStats()

    annotation_files = sorted(
        ANNOTATION_DIR.glob("*.json")
    )

    if len(annotation_files) == 0:

        print("No annotation files found.")

        return

    for annotation in annotation_files:

        mapping = (
            MAPPING_DIR /
            annotation.name
        )

        if not mapping.exists():

            print(
                f"Missing mapping : {mapping.name}"
            )

            continue

        convert_task(
            annotation,
            mapping,
            stats
        )

    stats.print()

# -------------------------------------------------------------
# Save class mapping
# -------------------------------------------------------------

def save_class_mapping():

    reverse = {}

    for name, idx in YOLO_CLASSES.items():

        reverse[idx] = name

    with open(
        AGV_OUTPUT / "class_mapping.json",
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

    print("\nValidating dataset...")

    images = {
        p.stem
        for p in IMAGE_DIR.iterdir()
    }

    labels = {
        p.stem
        for p in LABEL_DIR.iterdir()
    }

    missing = images - labels

    if len(missing):

        print(
            f"Missing labels : {len(missing)}"
        )

        for item in sorted(missing):

            print(item)

    else:

        print(
            "All images have labels."
        )
# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def convert_agv_dataset():

    print()

    print("=" * 60)

    print("Converting AGV COCO -> YOLO")

    print("=" * 60)

    ensure_dir(LABEL_DIR)

    convert_all_tasks()

    save_class_mapping()

    validate_dataset()

    print()

    print("=" * 60)

    print("AGV Conversion Completed")

    print("=" * 60)

# -------------------------------------------------------------

if __name__ == "__main__":

    convert_agv_dataset()
