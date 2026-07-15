
"""
Shared class mapping for AGV + FLIR thermal datasets.
Import this module from both agv_converter.py and flir_converter.py.
"""

# Merge original dataset classes into three semantic classes.
CLASS_MAPPING = {

    # Humans
    "person":"person",
    "pedestrian":"person",
    "human":"person",
    "cyclist":"person",
    "rider":"person",

    # Vehicles
    "car":"vehicle",
    "truck":"vehicle",
    "bus":"vehicle",
    "van":"vehicle",
    "pickup":"vehicle",
    "motorcycle":"vehicle",
    "bike":"vehicle",
    "bicycle":"vehicle",
    "tractor":"vehicle",

#     # Animals
#     "dog":"animal",
#     "cat":"animal",
#     "horse":"animal",
#     "cow":"animal",
#     "goat":"animal",
#     "sheep":"animal",
#     "bird":"animal",
#     "deer":"animal"
}

YOLO_CLASSES = {
    "person":0,
    "vehicle":1,
    # "animal":2
}

YOLO_NAMES = {
    0: "person",
    1: "vehicle",
    # 2: "animal",
}
