"""
==============================================================
Thermal Object Detection using YOLO11

Dataset:
    AGV Thermal + FLIR Thermal

Classes:
    0 -> Person
    1 -> Vehicle
    2 -> Animal

Author:
    Khushi

==============================================================
"""

from ultralytics import YOLO

from pathlib import Path

from datetime import datetime

import json
import shutil
import os
import time

import torch

# ==============================================================
#                     EXPERIMENT CONFIGURATION
# ==============================================================

EXPERIMENT = {

    # ----------------------------------------------------------
    # Experiment Information
    # ----------------------------------------------------------

    "id": "THERMAL_EXP003",

    "name": "YOLO11M_BASELINE",

    "description": "Baseline training on AGV + FLIR thermal dataset",

    # ----------------------------------------------------------
    # Dataset
    # ----------------------------------------------------------

    "dataset_yaml": "/home/deepterrain/Desktop/dataset_builder_thermal/ThermalDataset/Merged/dataset.yaml",

    "num_classes": 2,

    "classes": [

        "person",

        "vehicle"

    ],

    # ----------------------------------------------------------
    # Model
    # ----------------------------------------------------------

    "model": "yolo11m.pt",

    "pretrained": True,

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    "epochs": 50,

    "batch": 32,

    "imgsz": 1280,

    "optimizer": "AdamW",

    "lr": 0.001,

    "weight_decay": 0.0005,

    "workers": 8,

    "patience": 50,

    # ----------------------------------------------------------
    # Hardware
    # ----------------------------------------------------------

    "device": 1,

    "amp": True,

    "seed": 42,

    # ----------------------------------------------------------
    # Validation
    # ----------------------------------------------------------

    "conf": 0.25,

    "iou": 0.70

}



# ==============================================================
#             CREATE EXPERIMENT DIRECTORY
# ==============================================================

ROOT = Path.cwd()

EXP_DIR = ROOT / "experiments" / EXPERIMENT["id"]

TRAIN_DIR = EXP_DIR / "training"

TEST_DIR = EXP_DIR / "testing"

PRED_DIR = EXP_DIR / "predictions"

REPORT_DIR = EXP_DIR / "reports"

WEIGHT_DIR = TRAIN_DIR / "weights"

for folder in [

    EXP_DIR,

    TRAIN_DIR,

    TEST_DIR,

    PRED_DIR,

    REPORT_DIR,

    WEIGHT_DIR

]:

    folder.mkdir(

        parents=True,

        exist_ok=True

    )

# ==============================================================
# Save Configuration
# ==============================================================

config_file = EXP_DIR / "config.json"

with open(config_file, "w") as f:

    json.dump(

        EXPERIMENT,

        f,

        indent=4

    )

# ==============================================================
# Experiment Summary
# ==============================================================

print("\n")

print("="*70)

print("THERMAL OBJECT DETECTION")

print("="*70)

print("Experiment :", EXPERIMENT["id"])

print("Model      :", EXPERIMENT["model"])

print("Dataset    :", EXPERIMENT["dataset_yaml"])

print("Epochs     :", EXPERIMENT["epochs"])

print("Batch      :", EXPERIMENT["batch"])

print("Image Size :", EXPERIMENT["imgsz"])

print("="*70)
# ==============================================================
# Validate Dataset
# ==============================================================

dataset_yaml = Path(EXPERIMENT["dataset_yaml"])

if not dataset_yaml.exists():

    raise FileNotFoundError(

        dataset_yaml

    )

print("\nDataset found.")
# ==============================================================
#                  GPU INFORMATION
# ==============================================================

print("\nChecking hardware...\n")

if torch.cuda.is_available():

    DEVICE = f"cuda:{EXPERIMENT['device']}"

    gpu_name = torch.cuda.get_device_name(EXPERIMENT["device"])

    print(f"GPU       : {gpu_name}")

    print(f"CUDA      : {torch.version.cuda}")

    print(f"Device    : {DEVICE}")

else:

    DEVICE = "cpu"

    print("GPU not available.")
    print("Using CPU.")

print("\n")
# ==============================================================
#                    LOAD MODEL
# ==============================================================

print("=" * 70)
print("Loading YOLO Model")
print("=" * 70)

model = YOLO(EXPERIMENT["model"])

print(model.model)
# ==============================================================
#                  TRAINING TIMER
# ==============================================================

training_start = time.time()
# ==============================================================
#                      TRAIN MODEL
# ==============================================================

print("\n")
print("=" * 70)
print("Training Started")
print("=" * 70)

results = model.train(

    data=EXPERIMENT["dataset_yaml"],

    epochs=EXPERIMENT["epochs"],

    imgsz=EXPERIMENT["imgsz"],

    batch=EXPERIMENT["batch"],

    workers=EXPERIMENT["workers"],

    optimizer=EXPERIMENT["optimizer"],

    lr0=EXPERIMENT["lr"],

    weight_decay=EXPERIMENT["weight_decay"],

    patience=EXPERIMENT["patience"],

    device=DEVICE,

    amp=EXPERIMENT["amp"],

    seed=EXPERIMENT["seed"],

    project=str(EXP_DIR),

    name="training",

    exist_ok=True,

    verbose=True,

    pretrained=EXPERIMENT["pretrained"],

    save=True,

    save_period=10
)
# ==============================================================
#                  TRAINING COMPLETED
# ==============================================================

training_end = time.time()

training_time = training_end - training_start

hours = int(training_time // 3600)

minutes = int((training_time % 3600) // 60)

seconds = int(training_time % 60)

print("\n")
print("=" * 70)
print("Training Completed")
print("=" * 70)

print(

    f"Training Time : "

    f"{hours}h "

    f"{minutes}m "

    f"{seconds}s"

)
# ==============================================================
#               SAVE BEST MODEL LOCATION
# ==============================================================

best_weight = (

    EXP_DIR /

    "training" /

    "weights" /

    "best.pt"

)

last_weight = (

    EXP_DIR /

    "training" /

    "weights" /

    "last.pt"

)

print()

print("Best Model :")

print(best_weight)

print()

print("Last Model :")

print(last_weight)
# ==============================================================
#                  TRAINING SUMMARY
# ==============================================================

summary = {

    "experiment":

        EXPERIMENT["id"],

    "model":

        EXPERIMENT["model"],

    "epochs":

        EXPERIMENT["epochs"],

    "batch":

        EXPERIMENT["batch"],

    "imgsz":

        EXPERIMENT["imgsz"],

    "training_time_sec":

        training_time,

    "training_time":

        f"{hours}h {minutes}m {seconds}s"

}

with open(

    REPORT_DIR /

    "training_summary.json",

    "w"

) as f:

    json.dump(

        summary,

        f,

        indent=4

    )

# ==============================================================
#               LOAD BEST MODEL
# ==============================================================

print("\n")
print("=" * 70)
print("Loading Best Model")
print("=" * 70)

best_model = YOLO(best_weight)

print("Best model loaded successfully.")
# ==============================================================
#               VALIDATION
# ==============================================================

print("\n")
print("=" * 70)
print("VALIDATION")
print("=" * 70)

val_results = best_model.val(
    data=EXPERIMENT["dataset_yaml"],
    split="val",
    imgsz=EXPERIMENT["imgsz"],
    batch=EXPERIMENT["batch"],
    conf=EXPERIMENT["conf"],
    iou=EXPERIMENT["iou"],
    device=DEVICE,
    project=str(EXP_DIR),
    name="validation",
    exist_ok=True
)
# ==============================================================
#                 TEST SET
# ==============================================================

print("\n")
print("=" * 70)
print("TEST SET")
print("=" * 70)

test_results = best_model.val(
    data=EXPERIMENT["dataset_yaml"],
    split="test",
    imgsz=EXPERIMENT["imgsz"],
    batch=EXPERIMENT["batch"],
    conf=EXPERIMENT["conf"],
    iou=EXPERIMENT["iou"],
    device=DEVICE,
    project=str(EXP_DIR),
    name="test",
    exist_ok=True
)
# ==============================================================
#                 METRICS
# ==============================================================

metrics = {

    "Experiment": EXPERIMENT["id"],

    "Model": EXPERIMENT["model"],

    "Precision": float(test_results.box.mp),

    "Recall": float(test_results.box.mr),

    "mAP50": float(test_results.box.map50),

    "mAP50_95": float(test_results.box.map),

    "Fitness": float(test_results.fitness),

    "Training_Time": training_time

}
# ==============================================================
#              SAVE METRICS
# ==============================================================

metrics_file = REPORT_DIR / "metrics.json"

with open(metrics_file, "w") as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# ==============================================================
#              RESULTS
# ==============================================================

print("\n")
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for k, v in metrics.items():

    print(f"{k:20}: {v}")

print("=" * 70)
# ==============================================================
#            COPY IMPORTANT PLOTS
# ==============================================================

plots = [

    "results.png",

    "confusion_matrix.png",

    "confusion_matrix_normalized.png",

    "PR_curve.png",

    "P_curve.png",

    "R_curve.png",

    "F1_curve.png"

]

training_folder = EXP_DIR / "training"

for plot in plots:

    src = training_folder / plot

    if src.exists():

        shutil.copy2(
            src,
            REPORT_DIR / plot
        )

# ==============================================================
#               RUN INFERENCE ON TEST IMAGES
# ==============================================================

from pathlib import Path

print("\n")
print("=" * 70)
print("RUNNING SAMPLE INFERENCE")
print("=" * 70)

TEST_IMAGE_DIR = Path("ThermalDataset/Merged/test/images")

prediction_dir = PRED_DIR

prediction_dir.mkdir(exist_ok=True)

images = sorted(TEST_IMAGE_DIR.glob("*.*"))[:25]

for img in images:

    best_model.predict(

        source=str(img),

        save=True,

        project=str(prediction_dir),

        name="results",

        exist_ok=True,

        conf=EXPERIMENT["conf"]

    )

print(f"\nInference completed on {len(images)} images.")
# ==============================================================
#                 EXPORT MODEL
# ==============================================================

print("\n")
print("=" * 70)
print("EXPORTING MODEL")
print("=" * 70)

try:

    best_model.export(format="onnx")

    print("ONNX export successful.")

except Exception as e:

    print("ONNX export failed.")

    print(e)

# ==============================================================
#               GENERATE REPORT
# ==============================================================

report_file = REPORT_DIR / "experiment_report.txt"

with open(report_file, "w") as report:

    report.write("=" * 70 + "\n")

    report.write("THERMAL OBJECT DETECTION REPORT\n")

    report.write("=" * 70 + "\n\n")

    report.write(f"Experiment ID : {EXPERIMENT['id']}\n")

    report.write(f"Experiment    : {EXPERIMENT['name']}\n")

    report.write(f"Description   : {EXPERIMENT['description']}\n\n")

    report.write("MODEL\n")

    report.write(f"    {EXPERIMENT['model']}\n\n")

    report.write("DATASET\n")

    report.write(f"    {EXPERIMENT['dataset_yaml']}\n\n")

    report.write("TRAINING\n")

    report.write(f"Epochs      : {EXPERIMENT['epochs']}\n")

    report.write(f"Batch       : {EXPERIMENT['batch']}\n")

    report.write(f"Image Size  : {EXPERIMENT['imgsz']}\n")

    report.write(f"Optimizer   : {EXPERIMENT['optimizer']}\n")

    report.write(f"Learning Rt : {EXPERIMENT['lr']}\n")

    report.write(f"Training Time : {hours}h {minutes}m {seconds}s\n\n")

    report.write("RESULTS\n")

    report.write(f"Precision : {metrics['Precision']:.4f}\n")

    report.write(f"Recall    : {metrics['Recall']:.4f}\n")

    report.write(f"mAP50     : {metrics['mAP50']:.4f}\n")

    report.write(f"mAP50-95  : {metrics['mAP50_95']:.4f}\n")

    report.write(f"Fitness   : {metrics['Fitness']:.4f}\n")

    report.write("\n")

    report.write("=" * 70 + "\n")

# ==============================================================
#               EXPERIMENT COMPLETE
# ==============================================================

print("\n")
print("=" * 70)

print("EXPERIMENT COMPLETED SUCCESSFULLY")

print("=" * 70)

print(f"Experiment : {EXPERIMENT['id']}")

print(f"Model      : {EXPERIMENT['model']}")

print(f"Best Model : {best_weight}")

print(f"Reports    : {REPORT_DIR}")

print(f"Predictions: {PRED_DIR}")

print("=" * 70)

