import csv
import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
import train as T

MANIFEST = "split_manifest.csv"
CHECKPOINT = "best_grouped_split.pt"
THRESHOLD = 0.30
CSV_OUTPUT = "day7_validation_failures.csv"
IMAGE_OUTPUT = "worst8_validation_tuned.png"

# Select validation images only; this script does not evaluate the test set.
items, _ = T.get_samples()
image_ids = [os.path.relpath(img, T.ROOT).replace("\\", "/") for img, _ in items]
with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    split_by_image = {row["image"]: row["split"] for row in csv.DictReader(f)}
if set(image_ids) != set(split_by_image):
    raise ValueError("Dataset and split_manifest.csv do not contain the same image paths")
validation_items = [item for item, image_id in zip(items, image_ids)
                    if split_by_image[image_id] == "validation"]
if not validation_items:
    raise ValueError("No validation images found in split_manifest.csv")
print(f"validation images: {len(validation_items)} | threshold: {THRESHOLD:.2f}")

model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(T.DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=T.DEVICE))
model.eval()
dataset = T.BUSI(validation_items, A.Compose([]))
loader = DataLoader(dataset, batch_size=T.BATCH, shuffle=False, num_workers=0)

results = []
position = 0
with torch.no_grad():
    for x, masks in loader:
        probabilities = torch.sigmoid(model(x.to(T.DEVICE))).cpu().numpy()
        masks = masks.numpy()
        for j in range(len(x)):
            pred = probabilities[j, 0] > THRESHOLD
            truth = masks[j, 0] > 0.5
            intersection = np.logical_and(pred, truth).sum()
            pred_pixels = int(pred.sum())
            true_pixels = int(truth.sum())
            dice = (2 * intersection + 1e-6) / (pred_pixels + true_pixels + 1e-6)
            false_negative = int(np.logical_and(truth, ~pred).sum())
            false_positive = int(np.logical_and(pred, ~truth).sum())
            results.append({
                "dice": float(dice),
                "index": position,
                "lesion_pixels": true_pixels,
                "predicted_pixels": pred_pixels,
                "false_negative_pixels": false_negative,
                "false_positive_pixels": false_positive,
                "image": dataset.items[position][0],
            })
            position += 1

results.sort(key=lambda row: row["dice"])
with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
    fields = ["dice", "index", "lesion_pixels", "predicted_pixels",
              "false_negative_pixels", "false_positive_pixels", "image"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for row in results:
        writer.writerow({**row, "dice": f"{row['dice']:.6f}"})

print("worst 8 validation cases (dice, index, lesion_pixels, predicted_pixels):")
for row in results[:8]:
    print(f"{row['dice']:.3f} | index={row['index']} | lesion={row['lesion_pixels']} "
          f"| predicted={row['predicted_pixels']} | {row['image']}")

count = min(8, len(results))
fig, axes = plt.subplots(3, count, figsize=(3 * count, 9), squeeze=False)
for col, row in enumerate(results[:count]):
    i = row["index"]
    x, y = dataset[i]
    image = (x.numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225]
             + [0.485, 0.456, 0.406]).clip(0, 1)
    with torch.no_grad():
        pred = (torch.sigmoid(model(x.unsqueeze(0).to(T.DEVICE))) > THRESHOLD).cpu().numpy()[0, 0]
    axes[0, col].imshow(image)
    axes[0, col].set_title(f"Dice {row['dice']:.2f}")
    axes[1, col].imshow(y[0], cmap="gray", vmin=0, vmax=1)
    axes[2, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
    for r in range(3):
        axes[r, col].axis("off")
axes[0, 0].set_ylabel("Image")
axes[1, 0].set_ylabel("Reference mask")
axes[2, 0].set_ylabel("Prediction")
plt.tight_layout()
plt.savefig(IMAGE_OUTPUT, dpi=120)
print(f"saved {IMAGE_OUTPUT}")
print(f"saved {CSV_OUTPUT}")
