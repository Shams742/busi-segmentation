import csv
import os
from collections import defaultdict

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import train as T

MANIFEST = "split_manifest.csv"
CHECKPOINT = "best_grouped_split.pt"
THRESHOLD = 0.30
OUTPUT = "day8_validation_class_metrics.csv"

# Use only validation data; the held-out test set is not loaded or evaluated.
items, labels = T.get_samples()
image_ids = [os.path.relpath(img, T.ROOT).replace("\\", "/") for img, _ in items]
with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    split_by_image = {row["image"]: row["split"] for row in csv.DictReader(f)}
if set(image_ids) != set(split_by_image):
    raise ValueError("Dataset and split_manifest.csv do not contain the same image paths")
validation = [(item, label) for item, label, image_id in zip(items, labels, image_ids)
              if split_by_image[image_id] == "validation"]
if not validation:
    raise ValueError("No validation images found in split_manifest.csv")

model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(T.DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=T.DEVICE))
model.eval()
dataset = T.BUSI([item for item, _ in validation], A.Compose([]))
loader = DataLoader(dataset, batch_size=T.BATCH, shuffle=False, num_workers=0)
rows = []
position = 0
with torch.no_grad():
    for x, masks in loader:
        probabilities = torch.sigmoid(model(x.to(T.DEVICE))).cpu().numpy()
        masks = masks.numpy()
        for j in range(len(x)):
            pred = probabilities[j, 0] > THRESHOLD
            truth = masks[j, 0] > 0.5
            intersection = int(np.logical_and(pred, truth).sum())
            pred_count = int(pred.sum())
            true_count = int(truth.sum())
            dice = (2 * intersection + 1e-6) / (pred_count + true_count + 1e-6)
            iou = (intersection + 1e-6) / (pred_count + true_count - intersection + 1e-6)
            precision = intersection / pred_count if pred_count else (1.0 if not true_count else 0.0)
            recall = intersection / true_count if true_count else (1.0 if not pred_count else 0.0)
            item, label = validation[position]
            rows.append({
                "class": label,
                "image": item[0],
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall),
                "lesion_pixels": true_count,
                "predicted_pixels": pred_count,
            })
            position += 1

fields = ["class", "image", "dice", "iou", "precision", "recall",
          "lesion_pixels", "predicted_pixels"]
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, **{k: f"{row[k]:.6f}" for k in ("dice", "iou", "precision", "recall")}})

print(f"Validation metrics by class at threshold {THRESHOLD:.2f} (mean per-image):")
print("class      count  Dice   IoU    Precision  Recall  blank_predictions")
for label in ("benign", "malignant"):
    subset = [row for row in rows if row["class"] == label]
    if not subset:
        print(f"{label}: no validation images found")
        continue
    mean = lambda key: float(np.mean([row[key] for row in subset]))
    blank = sum(row["predicted_pixels"] == 0 for row in subset)
    print(f"{label:10} {len(subset):5d}  {mean('dice'):.3f}  {mean('iou'):.3f}"
          f"    {mean('precision'):.3f}      {mean('recall'):.3f}      {blank}")
print(f"saved {OUTPUT}")
