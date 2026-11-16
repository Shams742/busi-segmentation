import csv
import os

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import train as T

MANIFEST = "split_manifest.csv"
BASELINE_CHECKPOINT = "best_grouped_split.pt"
BALANCED_CHECKPOINT = "best_balanced.pt"
BASELINE_THRESHOLD = 0.30
BALANCED_THRESHOLD = 0.35
OUTPUT = "day12_paired_validation_comparison.csv"

items, labels = T.get_samples()
image_ids = [os.path.relpath(img, T.ROOT).replace("\\", "/") for img, _ in items]
label_by_image = dict(zip(image_ids, labels))
with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    split_by_image = {row["image"]: row["split"] for row in csv.DictReader(f)}
if set(image_ids) != set(split_by_image):
    raise ValueError("Dataset and split_manifest.csv do not contain the same image paths")

validation = [(item, image_id, label_by_image[image_id])
              for item, image_id in zip(items, image_ids)
              if split_by_image[image_id] == "validation"]
if not validation:
    raise ValueError("No validation images found in split_manifest.csv")

dataset = T.BUSI([item for item, _, _ in validation], A.Compose([]))
loader = DataLoader(dataset, batch_size=T.BATCH, shuffle=False, num_workers=0)

def evaluate_checkpoint(checkpoint, threshold):
    model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(T.DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=T.DEVICE))
    model.eval()
    values = []
    position = 0
    with torch.no_grad():
        for x, masks in loader:
            probs = torch.sigmoid(model(x.to(T.DEVICE))).cpu().numpy()
            masks = masks.numpy()
            for j in range(len(x)):
                pred = probs[j, 0] > threshold
                truth = masks[j, 0] > 0.5
                intersection = np.logical_and(pred, truth).sum()
                denominator = pred.sum() + truth.sum()
                dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
                values.append(float(dice))
                position += 1
    del model
    if position != len(validation):
        raise RuntimeError("Evaluation count did not match validation image count")
    return values

baseline = evaluate_checkpoint(BASELINE_CHECKPOINT, BASELINE_THRESHOLD)
balanced = evaluate_checkpoint(BALANCED_CHECKPOINT, BALANCED_THRESHOLD)
rows = []
for i, (_, image_id, label) in enumerate(validation):
    rows.append({
        "class": label,
        "image": image_id,
        "baseline_threshold": f"{BASELINE_THRESHOLD:.2f}",
        "baseline_dice": baseline[i],
        "balanced_threshold": f"{BALANCED_THRESHOLD:.2f}",
        "balanced_dice": balanced[i],
        "dice_change_balanced_minus_baseline": balanced[i] - baseline[i],
    })

with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    fields = ["class", "image", "baseline_threshold", "baseline_dice",
              "balanced_threshold", "balanced_dice", "dice_change_balanced_minus_baseline"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, **{key: f"{row[key]:.6f}" for key in
                                   ("baseline_dice", "balanced_dice",
                                    "dice_change_balanced_minus_baseline")}})

print(f"paired comparison uses the same {len(validation)} validation images")
print(f"baseline: threshold {BASELINE_THRESHOLD:.2f}, mean Dice {np.mean(baseline):.3f}")
print(f"balanced: threshold {BALANCED_THRESHOLD:.2f}, mean Dice {np.mean(balanced):.3f}")
print(f"balanced model: {sum(b > a + 1e-6 for a, b in zip(baseline, balanced))} images improved, "
      f"{sum(abs(b - a) <= 1e-6 for a, b in zip(baseline, balanced))} tied, "
      f"{sum(b < a - 1e-6 for a, b in zip(baseline, balanced))} lower")
print("mean Dice by class:")
for label in ("benign", "malignant"):
    indexes = [i for i, (_, _, y) in enumerate(validation) if y == label]
    if indexes:
        print(f"{label}: baseline {np.mean([baseline[i] for i in indexes]):.3f} | "
              f"balanced {np.mean([balanced[i] for i in indexes]):.3f}")
print(f"saved {OUTPUT}")
