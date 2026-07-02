# YOLO11 Dataset Builder -- 

## Overview

This pipeline builds a unified YOLO11 dataset from: - AGV (COCO JSON) -
MEI (CVAT XML)

Pipeline:
```
Discovery → Reports → Class Mapping → Class IDs → COCO/CVAT Conversion → Merge → Split → dataset.yaml → Validation
````
## Repository Structure

``` text
dataset_builder/
├── build_dataset.py
├── config.py
├── utils.py
├── yaml_generator.py
├── discover/
├── mapping/
├── converters/
├── merger/
├── splitter/
├── validator/
├── reports/
├── temp/
└── output/
```

## Execution

Run:

``` bash
python3 build_dataset.py
```

Expected stages:

1.  AGV Discovery
2.  MEI Discovery
3.  Report Generation
4.  Automatic Class Mapping
5.  YOLO Class ID Generation
6.  AGV Conversion
7.  MEI Conversion
8.  Dataset Merge
9.  Dataset Split
10. dataset.yaml Generation
11. Validation

## Modules

### discover/

Parses datasets into a common DatasetInfo representation.

### mapping/

Maps dataset-specific classes to canonical classes and assigns YOLO IDs.

### converters/

Converts COCO JSON and CVAT XML annotations into YOLO format.

### merger/

Combines datasets while preserving unique filenames.

### splitter/

Creates train/val/test folders.

### validator/

Checks dataset consistency before training.

## Output

``` text
output/
└── YOLO_DATASET/
    ├── dataset.yaml
    ├── images/
    │   ├── train
    │   ├── val
    │   └── test
    └── labels/
        ├── train
        ├── val
        └── test
```

## Troubleshooting

-   Images = 0 → Check dataset root and parser.
-   Missing Images → Verify image cache/discovery.
-   No Canonical Classes → Verify discovery populated classes.
-   No Space Left on Device → Prefer symbolic links.

## Extending the Pipeline

To add a new dataset:
1. Create a parser under discover/parsers.
2. Register it.
3. Populate DatasetInfo.
4. Reuse the remaining pipeline.

## Note
Always validate Discovery before Conversion. The rest of the pipeline
assumes DatasetInfo is correct.
