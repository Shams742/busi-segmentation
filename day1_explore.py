import glob, os, random
import cv2
import numpy as np
import matplotlib.pyplot as plt

ROOT = "Dataset_BUSI_with_GT"
classes = ["benign", "malignant", "normal"]

def get_images(c):
    return [f for f in glob.glob(f"{ROOT}/{c}/*.png") if "_mask" not in f]

def get_mask(img_path):
    masks = glob.glob(img_path[:-4] + "_mask*.png")
    m = None
    for mp in masks:
        x = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        m = x if m is None else np.maximum(m, cv2.resize(x, (m.shape[1], m.shape[0])))
    return m

for c in classes:
    print(c, len(get_images(c)), "images")

pool = get_images("benign") + get_images("malignant")
sample = random.sample(pool, 5)

fig, ax = plt.subplots(3, 5, figsize=(16, 9))
for i, p in enumerate(sample):
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    mask = get_mask(p)
    ax[0, i].imshow(img, cmap="gray")
    ax[0, i].set_title(os.path.basename(os.path.dirname(p)))
    ax[1, i].imshow(mask, cmap="gray")
    ax[2, i].imshow(img, cmap="gray")
    ax[2, i].imshow(mask, alpha=0.4, cmap="Reds")
    for r in range(3):
        ax[r, i].axis("off")
    print(os.path.basename(p), img.shape, "mask pixels:", int((mask > 0).sum()))
plt.tight_layout()
plt.savefig("day1_samples.png")
plt.show()
