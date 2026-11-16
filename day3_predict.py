import numpy as np, cv2, torch, matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
import train as T

items, labels = T.get_samples()
tr, rest, _, rest_l = train_test_split(items, labels, test_size=0.30, stratify=labels, random_state=T.SEED)
va, te = train_test_split(rest, test_size=0.50, stratify=rest_l, random_state=T.SEED)

ds = T.BUSI(te, T.A.Compose([]))
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1).to("cuda")
model.load_state_dict(torch.load("best.pt"))
model.eval()

rows = []
with torch.no_grad():
    for i in range(len(ds)):
        x, y = ds[i]
        p = (torch.sigmoid(model(x[None].cuda())) > 0.5).float().cpu()
        d, _ = T.dice_iou(p.logit(eps=1e-3) if False else (p*20-10), y[None])
        rows.append((d, i, y.sum().item()))

rows.sort()
print("worst 8 (dice, index, tumour_pixels):")
for r in rows[:8]:
    print(f"{r[0]:.3f}  {r[1]}  {int(r[2])}")

fig, ax = plt.subplots(3, 8, figsize=(24, 9))
for k, (d, i, _) in enumerate(rows[:8]):
    x, y = ds[i]
    img = (x.numpy().transpose(1, 2, 0) * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
    with torch.no_grad():
        p = (torch.sigmoid(model(x[None].cuda())) > 0.5).float().cpu()[0, 0].numpy()
    ax[0, k].imshow(img); ax[0, k].set_title(f"dice {d:.2f}")
    ax[1, k].imshow(y[0], cmap="gray"); ax[2, k].imshow(p, cmap="gray")
    for r in range(3): ax[r, k].axis("off")
plt.tight_layout(); plt.savefig("worst8.png", dpi=100)
print("saved worst8.png")