import csv
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import albumentations as A
import segmentation_models_pytorch as smp
import train as T

MANIFEST = "split_manifest.csv"
CHECKPOINT = "best_grouped_split.pt"
OUTPUT = "worst8_grouped_tuned.png"

# Load the exact test set created by train.py.
items, _ = T.get_samples()
item_ids = [os.path.relpath(img, T.ROOT).replace("\\", "/") for img, _ in items]
with open(MANIFEST, newline="", encoding="utf-8-sig") as f:
    split_by_image = {row["image"]: row["split"] for row in csv.DictReader(f)}
if set(item_ids) != set(split_by_image):
    raise ValueError("Dataset and split_manifest.csv do not contain the same image paths")
test_items = [item for item, image_id in zip(items, item_ids) if split_by_image[image_id] == "test"]
if not test_items:
    raise ValueError("No test images found in split_manifest.csv")
print(f"grouped test images: {len(test_items)}")

device = T.DEVICE
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
dataset = T.BUSI(test_items, A.Compose([]))

results = []
with torch.no_grad():
    for i in range(len(dataset)):
        x, y = dataset[i]
        pred = (torch.sigmoid(model(x.unsqueeze(0).to(device))) > 0.30).cpu().numpy()[0, 0]
        truth = y.numpy()[0] > 0.30
        intersection = np.logical_and(pred, truth).sum()
        denominator = pred.sum() + truth.sum()
        dice = (2 * intersection / denominator) if denominator else 1.0
        results.append((float(dice), i, int(truth.sum())))

results.sort()
print("worst 8 (dice, index, lesion_pixels):")
for dice, i, pixels in results[:8]:
    print(f"{dice:.3f} | index={i} | lesion_pixels={pixels} | {dataset.items[i][0]}")

count = min(8, len(results))
fig, axes = plt.subplots(3, count, figsize=(3 * count, 9), squeeze=False)
for col, (dice, i, _) in enumerate(results[:count]):
    x, y = dataset[i]
    image = (x.numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
    with torch.no_grad():
        pred = (torch.sigmoid(model(x.unsqueeze(0).to(device))) > 0.30).cpu().numpy()[0, 0]
    axes[0, col].imshow(image)
    axes[0, col].set_title(f"Dice {dice:.2f}")
    axes[1, col].imshow(y[0], cmap="gray", vmin=0, vmax=1)
    axes[2, col].imshow(pred, cmap="gray", vmin=0, vmax=1)
    for row in range(3):
        axes[row, col].axis("off")
axes[0, 0].set_ylabel("Image")
axes[1, 0].set_ylabel("Mask")
axes[2, 0].set_ylabel("Prediction")
plt.tight_layout()
plt.savefig(OUTPUT, dpi=120)
print(f"saved {OUTPUT}")
