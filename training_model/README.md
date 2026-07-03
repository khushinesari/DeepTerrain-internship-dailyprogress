## 📊 Recommended YOLO11 Training Experiments

The following experiments are recommended to evaluate the performance of different YOLO11 variants on the merged **AGV + MEI** dataset.

| Experiment | Model | Image Size | Epochs | Batch Size* | Purpose |
|------------|-------|-----------:|--------:|------------:|---------|
| E1 | `yolo11n.pt` | 640 | 50 | 16 | Fast baseline with lowest computational cost |
| E2 | `yolo11s.pt` | 640 | 100 | 16 | Lightweight model with improved accuracy |
| E3 | `yolo11m.pt` | 640 | 100 | 16 | Balanced trade-off between speed and accuracy |
| E4 ⭐ | `yolo11l.pt` | 640 | 100 | 16 | **Recommended model for the AGV + MEI dataset** |
| E5 | `yolo11x.pt` | 640 | 100 | 8–16 | Highest accuracy (requires a high-memory GPU) |
| E6 | `yolo11l.pt` | 1024 | 100 | 8 | High-resolution experiment for detecting smaller objects |

> **Note:** Batch size depends on available GPU memory. Adjust it according to your hardware specifications.

### Suggested Evaluation Metrics

For every experiment, record the following metrics:

| Metric | Description |
|--------|-------------|
| Precision | Fraction of predicted detections that are correct |
| Recall | Fraction of ground-truth objects correctly detected |
| mAP@50 | Mean Average Precision at IoU = 0.50 |
| mAP@50-95 | COCO-style Mean Average Precision across IoU thresholds |
| Training Time | Total training duration |
| Inference Speed | Average inference time per image |
| Model Size | Size of the exported `.pt` model |

### Recommended Final Model

For the merged **AGV + MEI** dataset (~67k images, 14 classes), the recommended configuration is:

| Parameter | Recommended Value |
|-----------|------------------|
| Model | `yolo11l.pt` |
| Epochs | `100` |
| Image Size | `640 × 640` |
| Optimizer | `AdamW` |
| Initial Learning Rate | `0.001` |
| Batch Size | `16` (adjust according to GPU memory) |
| Mixed Precision | Enabled (`AMP`) |
| Cache | Enabled |
| Learning Rate Scheduler | Cosine Annealing |
| Early Stopping Patience | `20` epochs |

### experimental models (YOLO11)

The experiments can be performed using different YOLO11 model variants depending on the available hardware and the desired balance between inference speed and detection accuracy.

| Model | Model Size | Performance | Typical Use Case |
|--------|------------|-------------|------------------|
| `yolo11n.pt` | Nano | Fastest inference with the lowest computational cost | Initial experiments, debugging, edge devices |
| `yolo11s.pt` | Small | Improved accuracy with a modest increase in computation | Lightweight deployment and rapid prototyping |
| `yolo11m.pt` | Medium | Good balance between speed and detection performance | General-purpose object detection |
| `yolo11l.pt` | Large | Higher accuracy with increased training and inference time | **Recommended for the merged AGV + MEI dataset** |
| `yolo11x.pt` | Extra Large | Highest detection accuracy with the largest computational requirements | Offline evaluation and high-end GPU systems |

> **Recommended configuration:** `yolo11l.pt` was selected for the final experiments as it provides a good balance between detection accuracy and computational cost for the merged AGV + MEI dataset.
