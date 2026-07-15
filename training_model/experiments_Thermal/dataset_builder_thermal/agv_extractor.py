
"""
agv_extractor.py

Robust AGV thermal extractor.
- Case-insensitive matching between Base_scenes and Annotation folders.
- Natural numeric sorting of Thermal_task folders.
- Copies images to AGV/images.
- Copies COCO jsons to AGV/annotations.
- Writes filename mapping JSON for later conversion.
"""

import json
import re
import shutil
from pathlib import Path

from config import AGV_IMAGE_ROOT, AGV_ANNOTATION_ROOT, AGV_OUTPUT
from utils import ensure_dir, get_images

IMAGE_DIR = AGV_OUTPUT / "images"
ANNOTATION_DIR = AGV_OUTPUT / "annotations"
MAPPING_DIR = AGV_OUTPUT / "mappings"


def prepare():
    ensure_dir(IMAGE_DIR)
    ensure_dir(ANNOTATION_DIR)
    ensure_dir(MAPPING_DIR)


def build_annotation_lookup():
    lookup = {}
    for folder in AGV_ANNOTATION_ROOT.iterdir():
        if folder.is_dir():
            lookup[folder.name.lower()] = folder
    return lookup


ANNOTATION_LOOKUP = build_annotation_lookup()


def task_number(folder: Path):
    m = re.search(r'(\d+)', folder.name)
    return int(m.group(1)) if m else 9999


def thermal_tasks():
    tasks = []
    for folder in AGV_IMAGE_ROOT.iterdir():
        if folder.is_dir() and folder.name.lower().startswith("thermal_task"):
            tasks.append(folder)
    tasks.sort(key=task_number)
    return tasks


def annotation_json(task_name: str):
    folder = ANNOTATION_LOOKUP.get(task_name.lower())
    if folder is None:
        return None
    anno = folder / "annotations" / "instances_default.json"
    return anno if anno.exists() else None


def extract_agv_dataset():
    prepare()

    img_index = 1
    total = 0

    print("\nScanning Thermal Tasks...\n")

    for task in thermal_tasks():

        print(f"Processing {task.name}")

        anno = annotation_json(task.name)

        if anno is None:
            print(f"  [WARNING] Annotation missing for {task.name}")
            continue

        mapping = {}

        images = get_images(task)

        for img in images:
            new_name = f"agv_{img_index:06d}{img.suffix.lower()}"

            shutil.copy2(img, IMAGE_DIR / new_name)

            mapping[img.name] = new_name

            img_index += 1
            total += 1

        shutil.copy2(
            anno,
            ANNOTATION_DIR / f"{task.name}.json"
        )

        with open(MAPPING_DIR / f"{task.name}.json", "w") as f:
            json.dump(mapping, f, indent=2)

        print(f"  Images copied : {len(images)}")

    print("\n--------------------------------")
    print(f"Total images extracted : {total}")
    print("--------------------------------")


if __name__ == "__main__":
    extract_agv_dataset()
