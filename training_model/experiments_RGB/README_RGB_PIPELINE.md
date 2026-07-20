# RGB Dataset Builder Pipeline

## Overview

The RGB Dataset Builder is a modular pipeline developed to generate a YOLO-compatible object detection dataset by combining multiple datasets into a unified format.

The current implementation supports:

- AGV RGB Dataset (COCO)
- MEI Dataset (Pascal VOC XML)

The builder performs automatic:

- Dataset discovery
- Class extraction
- Class mapping
- Class ID generation
- Annotation conversion
- Dataset merging
- Train/Validation/Test split
- YOLO dataset generation
- Dataset validation

The final output is directly compatible with Ultralytics YOLO.

---

# Pipeline Architecture

```
                  config.py
                      │
                      ▼
              build_dataset.py
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
  AGV Parser                 MEI Parser
        │                           │
        └─────────────┬─────────────┘
                      ▼
              Dataset Discovery
                      │
                      ▼
              Class Mapping
                      │
                      ▼
           Generate YOLO Class IDs
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
      COCO → YOLO             VOC → YOLO
        │                           │
        └─────────────┬─────────────┘
                      ▼
               Merge Datasets
                      │
                      ▼
            Train / Val / Test Split
                      │
                      ▼
              Generate dataset.yaml
                      │
                      ▼
              Dataset Validation
                      │
                      ▼
              Ready for YOLO Training
```

---

# Execution Flow

The complete pipeline is executed using a single command.

```bash
python3 build_dataset.py
```

Internally, the following modules execute sequentially.

---

## Step 1 — Dataset Discovery

File

```
discover/discover.py
```

Purpose

- Detect all available datasets.
- Load the corresponding parser.
- Generate dataset summaries.

Parsers used

```
discover/parsers/agv_parser.py
discover/parsers/mei_parser.py
```

Output

```
Running AGV Parser...

Running MEI Parser...

Discovery Summary
```

---

## Step 2 — Class Mapping

File

```
mapping/class_mapper.py
```

Purpose

Merge equivalent class names into canonical YOLO classes.

Example

```
Intruder
Person
Pedestrian

↓

Person
```

Output

```
Class Mapping Complete

Canonical Classes : 14
```

---

## Step 3 — Generate YOLO Class IDs

File

```
mapping/generate_class_ids.py
```

Purpose

Assign continuous YOLO IDs.

Example

```
0 Person
1 Car
2 Truck
...
```

Output

```
YOLO CLASS IDS
```

---

## Step 4 — Annotation Conversion

Files

```
converters/coco_to_yolo.py
converters/voc_to_yolo.py
```

Purpose

Convert

AGV

```
COCO JSON
```

↓

YOLO TXT

MEI

```
VOC XML
```

↓

YOLO TXT

Output

```
Converted XXXX images.
```

---

## Step 5 — Dataset Merge

File

```
merger/merge_dataset.py
```

Purpose

Merge converted datasets into a common temporary directory.

Output structure

```
temp/

    agv/

    mei/

↓

temp/

    merged/

        images/

        labels/
```

Output

```
MERGE SUMMARY

Images Copied

Labels Copied

Renamed

Skipped
```

---

## Step 6 — Dataset Split

File

```
splitter/split_dataset.py
```

Purpose

Shuffle and split the merged dataset.

Default split

```
70%

20%

10%
```

Result

```
output/

    YOLO_DATASET/

        images/

            train/

            val/

            test/

        labels/

            train/

            val/

            test/
```

Output

```
DATASET SPLIT COMPLETE

Train

Val

Test
```

---

## Step 7 — Generate dataset.yaml

File

```
yaml_generator.py
```

Creates

```
dataset.yaml
```

Example

```yaml
path: output/YOLO_DATASET

train: images/train
val: images/val
test: images/test

nc: 14

names:
  0: Animal
  1: Auto_Rickshaw
  ...
```

---

## Step 8 — Dataset Validation

File

```
validator/verify_dataset.py
```

Validation checks

- Missing images
- Missing labels
- Corrupt images
- Invalid class IDs
- Bounding boxes outside [0,1]
- Dataset structure

Output

```
DATASET VALIDATION

Images

Labels

Objects

Warnings

Errors
```

If

```
Errors = 0
```

the dataset is ready for training.

---

# Final Dataset Structure

```
output/

└── YOLO_DATASET

    ├── images

    │     ├── train

    │     ├── val

    │     └── test

    │

    ├── labels

    │     ├── train

    │     ├── val

    │     └── test

    │

    └── dataset.yaml
```

---

# Training Workflow

After dataset generation, train YOLO using

```
train_yolo11.py
```

Pipeline

```
dataset.yaml

        │

        ▼

Load pretrained YOLO

        │

        ▼

Training

        │

        ▼

Validation after every epoch

        │

        ▼

Best Model Saved

        │

        ▼

Final Evaluation
```

Typical command

```bash
python3 train_yolo11.py
```

---

# Train / Validation / Test Procedure

## Training Set

Purpose

- Learn object features.
- Update model weights using backpropagation.

Typical size

```
70%
```

Example

```
42,879 images
```

---

## Validation Set

Purpose

- Evaluate after every epoch.
- Select the best model.
- Perform early stopping.

Typical size

```
20%
```

Example

```
12,251 images
```

Validation metrics

- Precision
- Recall
- mAP@50
- mAP@50-95

The validation set is **never used for weight updates**.

---

## Test Set

Purpose

Evaluate the final trained model on unseen data.

Typical size

```
10%
```

Example

```
6,127 images
```

The test set is used only after training is complete to estimate real-world performance.

---

# Output Generated During Training

Each experiment creates a dedicated directory.

```
experiments/

    exp01_yolo11n_50e_bs16/

        weights/

            best.pt

            last.pt

        results.png

        results.csv

        confusion_matrix.png

        F1_curve.png

        P_curve.png

        PR_curve.png

        R_curve.png

        args.yaml
```

# Notes

- Always validate the dataset before training.
- Preserve each experiment in a separate directory.
- Record model configuration, epochs, batch size and final metrics for reproducibility.
- Never modify the test set during experimentation.
