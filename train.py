import glob, os
import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ---------- Settings ----------
EPOCHS = 30          # 1 for smoke test, then change to 30
BATCH = 16
SIZE = 256
SEED = 42
ROOT = "Dataset_BUSI_with_GT"
DEVICE = "cuda"

# ---------- 1. Collect images + merge masks ----------
def get_samples():
    items, labels = [], []
    for cls in ["benign", "malignant"]:
        for img in glob.glob(f"{ROOT}/{cls}/*.png"):
            if "_mask" in os.path.basename(img):
                continue
            base = img[:-4]
            masks = glob.glob(base + "_mask*.png")   # finds _mask, _mask_1, ...
            items.append((img, masks))
            labels.append(cls)
    return items, labels

# ---------- 2. Dataset ----------
class BUSI(Dataset):
    def __init__(self, items, aug):
        self.items, self.aug = items, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, mask_paths = self.items[i]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (SIZE, SIZE))
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        for mp in mask_paths:
            m = cv2.imread(mp, 0)
            m = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            mask = np.maximum(mask, (m > 0).astype(np.uint8))   # merge
        out = self.aug(image=img, mask=mask)
        img, mask = out["image"], out["mask"]
        img = (img / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return img, mask

# ---------- 3. Metrics ----------
def dice_iou(logits, mask):
    p = (torch.sigmoid(logits) > 0.5).float()
    inter = (p * mask).sum((1, 2, 3))
    union = p.sum((1, 2, 3)) + mask.sum((1, 2, 3))
    dice = ((2 * inter + 1e-6) / (union + 1e-6)).mean().item()
    iou = ((inter + 1e-6) / (union - inter + 1e-6)).mean().item()
    return dice, iou

def evaluate(model, loader):
    model.eval()
    d, j, n = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            di, io = dice_iou(model(x), y)
            d += di * len(x); j += io * len(x); n += len(x)
    return d / n, j / n

# ---------- 4. Main ----------
if __name__ == "__main__":
    items, labels = get_samples()
    print("total images:", len(items))

    # 70 / 15 / 15 stratified split
    tr, rest, _, rest_l = train_test_split(items, labels, test_size=0.30,
                                           stratify=labels, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest_l, random_state=SEED)
    print("train/val/test:", len(tr), len(va), len(te))

    train_aug = A.Compose([A.HorizontalFlip(p=0.5),
                           A.Rotate(limit=15, p=0.5),
                           A.RandomBrightnessContrast(p=0.3)])
    no_aug = A.Compose([])

    tl = DataLoader(BUSI(tr, train_aug), batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(BUSI(va, no_aug), batch_size=BATCH, num_workers=0)
    tel = DataLoader(BUSI(te, no_aug), batch_size=BATCH, num_workers=0)

    model = smp.Unet("resnet34", encoder_weights="imagenet",
                     in_channels=3, classes=1).to(DEVICE)
    dice_loss = smp.losses.DiceLoss("binary")
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    best = 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = dice_loss(out, y) + bce(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        vd, vi = evaluate(model, vl)
        print(f"epoch {ep}/{EPOCHS}  loss {total/len(tl):.3f}  val dice {vd:.3f}  val iou {vi:.3f}")
        if vd > best:
            best = vd
            torch.save(model.state_dict(), "best.pt")

    model.load_state_dict(torch.load("best.pt"))
    td, ti = evaluate(model, tel)
    print(f"TEST dice {td:.3f}  iou {ti:.3f}")