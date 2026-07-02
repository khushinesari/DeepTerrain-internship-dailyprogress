# useful clases from coco dataset 
1. person
2. bicycle
3. car
4. motorcycle
5. bus
6. truck
7. cat
8. dog
9. cow
# dataset mei
- Pedestrian
- Attributes:
    - Age_Group: Child, Adult
    - Gender: Male, Female
- Bicycle
- Bike
- Car
- Truck
- Bus
- Auto_Rickshaw
- Person
- Attributes:
   - Age_Group: Child, Adult
   - Gender: Male,Female
- Tractor
- Van
- Proclainer
- Crane
- Road_Roller
- Road_paver
- Animal (Phase 2)
    - Attributes: Cow, Dog, Monkey
# deepterrain_data
- intruder(person only)
# thermal camera data 
- person 
--------------
## extracting and merging data from AGV-RGB and MEI dataset 
```
Data_March_labelled (COCO JSON)
                +
MEI_DATASET (Pascal VOC XML)
                │
                ▼
Standardize Classes
                │
                ▼
Convert Both to YOLO Format
                │
                ▼
Merge
                │
                ▼
Balance Dataset
                │
                ▼
Train YOLO11
```
### Dataset 1 (DML)
DML dataset contains COCO-format annotations with categories such as:
```
Intruder
Frame_Metadata
```
along with bounding boxes and attributes.
### Dataset 2 (MEI)
MEI dataset uses Pascal VOC XML annotations.

Its structure is:
```
MEI_DATASET/

Annotation_XML/
    MEI_AEBS_LDWS_Batch1/
        annotations.xml
    MEI_AEBS_LDWS_Batch2/
        annotations.xml
    ...

Preselection image/
    MEI_AEBS_LDWS_Batch1/
        *.jpg
```

### folder structure 
```
dataset_builder/
│
├── build_dataset.py
├── config.py
├── utils.py
│
├── discover/
│   ├── __init__.py
│   ├── dataset_info.py
│   ├── base_parser.py
│   ├── parser_registry.py
│   ├── discover.py
│   ├── report_generator.py
│   │
│   └── parsers/
│       ├── __init__.py
│       ├── agv_parser.py
│       └── mei_parser.py
│
├── mapping/
├── converters/
├── merger/
├── splitter/
├── validator/
│
├── reports/
├── logs/
├── temp/
└── output/
```
