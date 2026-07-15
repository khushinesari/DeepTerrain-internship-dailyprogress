"""
Configuration for Thermal Dataset Builder
"""

from pathlib import Path
KEEP_NEGATIVE_IMAGES = True
# ------------------------------------------------------------------
# AGV DATASET
# ------------------------------------------------------------------

AGV_IMAGE_ROOT = Path(r"/home/deepterrain/Desktop/Data_March_labelled/DML/Base_scenes/AGV")
AGV_ANNOTATION_ROOT = Path(r"/home/deepterrain/Desktop/Data_March_labelled/DML/Annotation/AGV")

# ------------------------------------------------------------------
# FLIR DATASET
# ------------------------------------------------------------------

FLIR_ROOT = Path(r"/home/deepterrain/Desktop/flir_thermal_dataset")

FLIR_SPLITS = [
    "images_thermal_train",
    "images_thermal_val",
    "video_thermal_test"
]

# ------------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------------

OUTPUT_ROOT = Path(r"/home/deepterrain/Desktop/dataset_builder_thermal/ThermalDataset")

AGV_OUTPUT = OUTPUT_ROOT / "AGV"
FLIR_OUTPUT = OUTPUT_ROOT / "FLIR"

MERGED_OUTPUT = OUTPUT_ROOT / "Merged"

TRAIN_DIR = MERGED_OUTPUT / "train"
VAL_DIR = MERGED_OUTPUT / "val"
TEST_DIR = MERGED_OUTPUT / "test"

# ------------------------------------------------------------------
# DATASET SPLIT
# ------------------------------------------------------------------

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42

# ------------------------------------------------------------------
# IMAGE TYPES
# ------------------------------------------------------------------

VALID_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
]

