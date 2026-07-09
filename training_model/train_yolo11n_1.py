#!/usr/bin/env python3
"""
=============================================================
YOLO11 Training Script
=============================================================

Trains a YOLO11 model on the merged AGV + MEI dataset.

Author : DeepTerrain AI
"""

from pathlib import Path
from ultralytics import YOLO
import torch
import time

# ==========================================================
# CONFIGURATION
# ==========================================================

# Dataset YAML
DATASET = "/home/deepterrain/Desktop/dataset_builder/output/YOLO_DATASET/dataset.yaml"

# Model
# Options:
# yolo11n.pt
# yolo11s.pt
# yolo11m.pt
# yolo11l.pt
# yolo11x.pt
MODEL = "yolo11n.pt"

# Experiment name
PROJECT = "runs/train"
NAME = "AGV_MEI_YOLO11"

# Training
EPOCHS = 50
IMGSZ = 1280
BATCH = 32
DEVICE = 0          # GPU ID
WORKERS = 8
PATIENCE = 20

# Optimizer
OPTIMIZER = "AdamW"
LR = 0.001
WEIGHT_DECAY = 5e-4

# Augmentation
MOSAIC = 1.0
MIXUP = 0.15
COPY_PASTE = 0.0

FLIPLR = 0.5
FLIPUD = 0.0

DEGREES = 5
TRANSLATE = 0.1
SCALE = 0.5
SHEAR = 2

CACHE = True
AMP = True

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("YOLO11 TRAINING")
    print("=" * 60)

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    print("=" * 60)

    if not Path(DATASET).exists():
        raise FileNotFoundError(DATASET)

    model = YOLO(MODEL)

    start = time.time()

    results = model.train(

        data=DATASET,

        epochs=EPOCHS,

        imgsz=IMGSZ,

        batch=BATCH,

        device=DEVICE,

        workers=WORKERS,

        optimizer=OPTIMIZER,

        lr0=LR,

        weight_decay=WEIGHT_DECAY,

        patience=PATIENCE,

        cache=CACHE,

        amp=AMP,

        cos_lr=True,

        pretrained=True,

        mosaic=MOSAIC,

        mixup=MIXUP,

        copy_paste=COPY_PASTE,

        fliplr=FLIPLR,

        flipud=FLIPUD,

        degrees=DEGREES,

        translate=TRANSLATE,

        scale=SCALE,

        shear=SHEAR,

        project=PROJECT,

        name=NAME,

        exist_ok=True,

        save=True,

        save_period=10,

        plots=True,

        verbose=True
    )

    end = time.time()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"Training Time : {(end-start)/3600:.2f} hours")

    print("\nEvaluating best model...\n")

    # Path where Ultralytics actually saved the experiment
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"

    print(f"Loading model from:\n{best_model_path}\n")

    best_model = YOLO(best_model_path)

    metrics = best_model.val()
    print("\n================ FINAL RESULTS ================\n")

    print(f"Precision : {metrics.box.mp:.4f}")
    print(f"Recall    : {metrics.box.mr:.4f}")
    print(f"mAP@50    : {metrics.box.map50:.4f}")
    print(f"mAP50-95  : {metrics.box.map:.4f}")

    print("\nBest Model Saved At:")
    print(best_model_path)

    print("\nResults Folder:")
    print(results.save_dir)

    print("=" * 60)


if __name__ == "__main__":
    main()
