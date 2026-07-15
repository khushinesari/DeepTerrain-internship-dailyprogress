"""
Generate dataset.yaml
"""

import json

from pathlib import Path

from config import MERGED_OUTPUT


def generate_yaml():

    class_file = Path(
        r"ThermalDataset/FLIR/class_mapping.json"
    )

    if not class_file.exists():

        class_file = Path(
            r"ThermalDataset/AGV/class_mapping.json"
        )

    with open(class_file) as f:

        classes = json.load(f)

    yaml_file = MERGED_OUTPUT / "dataset.yaml"

    with open(yaml_file, "w") as f:

        f.write(f"path: {MERGED_OUTPUT.resolve()}\n\n")

        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("test: test/images\n\n")

        f.write(f"nc: {len(classes)}\n\n")

        f.write("names:\n")

        for idx in sorted(classes.keys(), key=int):

            f.write(f"  {idx}: {classes[idx]}\n")

    print("dataset.yaml Generated")
