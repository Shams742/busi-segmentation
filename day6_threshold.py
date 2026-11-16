import csv
import os

import numpy as np
import torch
import albumentations as A
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import train as T

MANIFEST = "split_manifest.csv"
CHECKPOINT = "best_grouped_split.pt"
RESULTS = "day6_threshold_results.csv"
THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.801, 0.05)]

# Use the exact train/validation/test assignment from the grouped run.
items, _ = T.get_samples()
image_ids = [os.path.relpath(img, T.ROOT).replace("\\", "/") for img, _ in items]
with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    split_by_image = {row["image"]: row["split"] for row in csv.DictReader(f)}
if set(image_ids) != set(split_by_image):
    raise ValueError("Dataset and split_manifest.csv do not contain the same image paths")

by_split = {
    name: [item for item, image_id in zip(items, image_ids) if split_by_image[image_id] == name]
    for name in ("validation", "test")
}
if not by_split["validation"] or not by_split["test"]:
    raise ValueError("Validation or test split is empty in split_manifest.csv")

model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(T.DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=T.DEVICE))
model.eval()

def get_predictions(samples):
    dataset = T.BUSI(samples, A.Compose([]))
    loader = DataLoader(dataset, batch_size=T.BATCH, shuffle=False, num_workers=0)
    probabilities, masks = [], []
    with torch.no_grad():
        for x, y in loader:
            probabilities.extend(torch.sigmoid(model(x.to(T.DEVICE))).cpu().numpy())
            masks.extend(y.numpy())
    return probabilities, masks

def metrics(probabilities, masks, threshold):
    dice_scores, iou_scores = [], []
    for probability, mask in zip(probabilities, masks):
        pred = probability[0] > threshold
        truth = mask[0] > 0.5
        intersection = np.logical_and(pred, truth).sum()
        pred_count, truth_count = pred.sum(), truth.sum()
        dice_scores.append((2 * intersection + 1e-6) / (pred_count + truth_count + 1e-6))
        iou_scores.append((intersection + 1e-6) / (pred_count + truth_count - intersection + 1e-6))
    return float(np.mean(dice_scores)), float(np.mean(iou_scores))

# Choose a threshold from validation only.
val_probabilities, val_masks = get_predictions(by_split["validation"])
val_results = []
for threshold in THRESHOLDS:
    dice, iou = metrics(val_probabilities, val_masks, threshold)
    val_results.append((threshold, dice, iou))
best_threshold, best_val_dice, best_val_iou = max(val_results, key=lambda row: row[1])
print("validation threshold results (threshold, Dice, IoU):")
for threshold, dice, iou in val_results:
    print(f"{threshold:.2f}  {dice:.3f}  {iou:.3f}")
print(f"selected threshold: {best_threshold:.2f} | validation Dice: {best_val_dice:.3f} | IoU: {best_val_iou:.3f}")

# Evaluate the test set once, using the threshold selected on validation.
test_probabilities, test_masks = get_predictions(by_split["test"])
test_dice, test_iou = metrics(test_probabilities, test_masks, best_threshold)
print(f"TEST at selected threshold {best_threshold:.2f}: Dice {test_dice:.3f} | IoU {test_iou:.3f}")

with open(RESULTS, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["threshold", "validation_dice", "validation_iou", "test_dice", "test_iou"])
    for threshold, dice, iou in val_results:
        if threshold == best_threshold:
            writer.writerow([f"{threshold:.2f}", f"{dice:.6f}", f"{iou:.6f}", f"{test_dice:.6f}", f"{test_iou:.6f}"])
        else:
            writer.writerow([f"{threshold:.2f}", f"{dice:.6f}", f"{iou:.6f}", "", ""])
print(f"saved {RESULTS}")
