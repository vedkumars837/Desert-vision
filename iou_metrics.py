# iou_metrics.py
import torch
import numpy as np


def calculate_iou(predictions, targets, num_classes):
    iou_per_class = []

    for cls in range(num_classes):
        pred_mask   = (predictions == cls)
        target_mask = (targets == cls)

        intersection = (pred_mask & target_mask).sum().float()
        union        = (pred_mask | target_mask).sum().float()

        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou = (intersection / union).item()
            iou_per_class.append(iou)

    valid_ious = [x for x in iou_per_class if not np.isnan(x)]
    mean_iou   = np.mean(valid_ious) if valid_ious else 0.0

    return iou_per_class, mean_iou


def print_iou_results(iou_per_class, mean_iou, class_names):
    print("\n" + "="*50)
    print("📊 ACCURACY RESULTS (IoU per class):")
    print("="*50)
    for i, (iou, name) in enumerate(zip(iou_per_class, class_names.values())):
        if np.isnan(iou):
            print(f"  {name:20s}: N/A")
        else:
            bar = "█" * int(iou * 20)
            print(f"  {name:20s}: {iou:.4f} |{bar:<20}|")
    print(f"\n  {'Mean IoU':20s}: {mean_iou:.4f}")
    print("="*50)