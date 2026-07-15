"""
Thermal Dataset Builder

Pipeline

1. Extract AGV
2. Convert AGV
3. Convert FLIR
4. Merge
5. Split
6. Generate dataset.yaml
7. Dataset Statistics

Run:

python builder.py
"""

from agv_extractor import extract_agv_dataset
from agv_converter import convert_agv_dataset

from flir_converter import convert_flir_dataset

from merge_builder import merge_datasets

from split_builder import split_dataset

from yaml_builder import generate_yaml

from dataset_stats import dataset_statistics


def banner(title):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)


def main():

    banner("STEP 1 : Extracting AGV Thermal Dataset")
    extract_agv_dataset()

    banner("STEP 2 : Converting AGV COCO -> YOLO")
    convert_agv_dataset()

    banner("STEP 3 : Converting FLIR COCO -> YOLO")
    convert_flir_dataset()

    banner("STEP 4 : Merging Datasets")
    merge_datasets()

    banner("STEP 5 : Splitting Dataset")
    split_dataset()

    banner("STEP 6 : Creating dataset.yaml")
    generate_yaml()

    banner("STEP 7 : Dataset Statistics")
    dataset_statistics("ThermalDataset/Merged")

    banner("Completed Successfully")


if __name__ == "__main__":
    main()
