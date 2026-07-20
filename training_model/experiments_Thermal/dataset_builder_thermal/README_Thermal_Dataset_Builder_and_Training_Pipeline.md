# Thermal Dataset Builder & YOLO Training Pipeline

## Project Overview

This document describes the complete workflow for building a unified
thermal object detection dataset from AGV Thermal and FLIR ADAS Thermal
data, followed by training, validation, and testing using YOLO.

## Dataset Builder Pipeline

``` text
AGV Thermal Images
      │
      ▼
AGV Extractor
      │
      ▼
AGV COCO → YOLO Converter
      │
      ├──────────────┐
      ▼              │
FLIR COCO → YOLO     │
Converter            │
      │              │
      └──────Merge───┘
             │
             ▼
      Merged Dataset
             │
             ▼
     Train / Val / Test Split
             │
             ▼
        dataset.yaml
             │
             ▼
      Dataset Statistics
```

## Directory Structure

``` text
dataset_builder_thermal/
├── builder.py
├── config.py
├── utils.py
├── class_mapping.py
├── agv_extractor.py
├── agv_converter.py
├── flir_converter.py
├── merge_builder.py
├── split_builder.py
├── yaml_builder.py
├── dataset_stats.py
├── train_thermal_yolo.py
└── ThermalDataset/
    ├── AGV/
    ├── FLIR/
    └── Merged/
```

## Dataset Builder Steps

1.  Extract AGV thermal images and annotations.
2.  Convert AGV COCO annotations to YOLO format.
3.  Convert FLIR COCO annotations to YOLO format.
4.  Merge AGV and FLIR datasets.
5.  Split merged dataset (default 80/10/10).
6.  Generate `dataset.yaml`.
7.  Validate dataset and generate statistics.

## Final Dataset Layout

``` text
Merged/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Training Pipeline

``` text
dataset.yaml
      │
      ▼
Dataset Validation
      │
      ▼
Load YOLO Model
      │
      ▼
Create Manual Experiment Folder
      │
      ▼
Training
      │
      ▼
Validation
      │
      ▼
Testing
      │
      ▼
Save Best Weights
      │
      ▼
Export Metrics and Plots
```

## Experiment Structure

``` text
experiments/
└── THERMAL_EXP001/
    ├── config.json
    ├── summary.txt
    ├── metrics.csv
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── plots/
    ├── predictions/
    └── logs/
```

## Evaluation Metrics

-   Precision
-   Recall
-   mAP@0.5
-   mAP@0.5:0.95
-   F1 Score
-   Confusion Matrix
-   Precision-Recall Curve

## Common Issues

-   Empty labels
-   Duplicate images


