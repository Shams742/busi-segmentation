import csv
import os
from collections import Counter

import albumentations as A
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import train as T

EPOCHS = T.EPOCHS
BATCH = T.BATCH
SEED = T.SEED
CHECKPOINT = "best_balanced.pt"
THRESHOLD = 0.30
METRICS_OUTPUT = "day9_balanced_validation_metrics.csv"

# Reuse the duplicate-aware split; this experiment does not evaluate the test set.
items, labels = T.get_samples()
tr, va, _ = T.make_grouped_split(items, labels)
image_ids = [os.path.relpath(img, T.ROOT).replace("\\", "/") for img, _ in items]
label_by_image = dict(zip(image_ids, labels))
train_labels = [label_by_image[os.path.relpath(img, T.ROOT).replace("\\", "/")]
                for img, _ in tr]
class_counts = Counter(train_labels)
weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(weights),
    num_samples=len(weights),
    replacement=True,
    generator=torch.Generator().manual_seed(SEED),
)
print(f"train images: {len(tr)} | validation images: {len(va)}")
print(f"training class counts: {dict(class_counts)} | balanced sampling: on")

train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])
no_aug = A.Compose([])
train_loader = DataLoader(T.BUSI(tr, train_aug), batch_size=BATCH,
                          sampler=sampler, num_workers=0)
val_loader = DataLoader(T.BUSI(va, no_aug), batch_size=BATCH,
                        shuffle=False, num_workers=0)

model = smp.Unet("resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=1).to(T.DEVICE)
dice_loss = smp.losses.DiceLoss("binary")
bce = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
best_val_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(T.DEVICE), y.to(T.DEVICE)
        logits = model(x)
        loss = dice_loss(logits, y) + bce(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    val_dice, val_iou = T.evaluate(model, val_loader)
    print(f"epoch {epoch}/{EPOCHS}  loss {total_loss/len(train_loader):.3f}  "
          f"val dice {val_dice:.3f}  val iou {val_iou:.3f}")
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), CHECKPOINT)

# Evaluate the selected checkpoint on validation only, using the fixed 0.30 threshold.
model.load_state_dict(torch.load(CHECKPOINT, map_location=T.DEVICE))
model.eval()
val_ids = [os.path.relpath(img, T.ROOT).replace("\\", "/") for img, _ in va]
val_labels = [label_by_image[image_id] for image_id in val_ids]
val_dataset = T.BUSI(va, no_aug)
val_loader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, num_workers=0)
rows = []
position = 0
with torch.no_grad():
    for x, masks in val_loader:
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
            rows.append({
                "class": val_labels[position],
                "image": va[position][0],
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
with open(METRICS_OUTPUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, **{k: f"{row[k]:.6f}" for k in ("dice", "iou", "precision", "recall")}})

print(f"class-wise validation metrics at threshold {THRESHOLD:.2f} (mean per-image):")
print("class      count  Dice   IoU    Precision  Recall  blank_predictions")
for label in ("benign", "malignant"):
    subset = [row for row in rows if row["class"] == label]
    if subset:
        mean = lambda key: float(np.mean([row[key] for row in subset]))
        blank = sum(row["predicted_pixels"] == 0 for row in subset)
        print(f"{label:10} {len(subset):5d}  {mean('dice'):.3f}  {mean('iou'):.3f}"
              f"    {mean('precision'):.3f}      {mean('recall'):.3f}      {blank}")
print(f"saved checkpoint {CHECKPOINT}")
print(f"saved {METRICS_OUTPUT}")
