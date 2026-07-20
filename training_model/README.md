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

### experimental models (YOLO11)

The experiments can be performed using different YOLO11 model variants depending on the available hardware and the desired balance between inference speed and detection accuracy.

| Model | Model Size | Performance | Typical Use Case |
|--------|------------|-------------|------------------|
| `yolo11n.pt` | Nano | Fastest inference with the lowest computational cost | Initial experiments, debugging, edge devices |
| `yolo11s.pt` | Small | Improved accuracy with a modest increase in computation | Lightweight deployment and rapid prototyping |
| `yolo11m.pt` | Medium | Good balance between speed and detection performance | General-purpose object detection |
| `yolo11l.pt` | Large | Higher accuracy with increased training and inference time | **Recommended for the merged AGV + MEI dataset** |
| `yolo11x.pt` | Extra Large | Highest detection accuracy with the largest computational requirements | Offline evaluation and high-end GPU systems |

> **Note:** Batch size depends on available GPU memory. Adjust it according to your hardware specifications.

> **Recommended configuration:** `yolo11l.pt` was selected for the final experiments as it provides a good balance between detection accuracy and computational cost for the merged AGV + MEI dataset.
