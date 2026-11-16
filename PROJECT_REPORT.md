\# BUSI Ultrasound Lesion Segmentation



\## Goal

Train a U-Net with a ResNet34 encoder to segment lesion masks in BUSI breast ultrasound images. This is a learning project, not a clinical tool.



\## Data and split

\- 647 images

\- Similar-image groups were kept within one split to reduce data leakage.

\- Train: 449 images; validation: 100; test: 98.

\- Dataset source and license: verify and cite the source used.



\## Experiments

Both models used the same grouped split.



| Model | Threshold | Validation Dice / IoU | Test Dice / IoU |

|---|---:|---:|---:|

| Original | 0.30 | 0.819 / 0.729 | 0.772 / 0.688 |

| Balanced sampling | 0.35 | 0.835 / 0.745 | 0.768 / 0.681 |



Balanced sampling improved validation results but did not improve the test result on this split.



\## Limitations and next work

The model still makes large segmentation errors on some images. Results come from one split and should not be treated as clinical performance. Future work could test more grouped splits and improve the handling of difficult cases.


## Paired validation comparison

Both models were compared on the same 100 validation images. The original model used threshold 0.30; the balanced model used 0.35.

- Mean validation Dice: original 0.819; balanced 0.835.
- The balanced model scored higher on 52 images and lower on 48.
- Benign mean Dice: original 0.861; balanced 0.870.
- Malignant mean Dice: original 0.730; balanced 0.761.

On the held-out test set, the original scored Dice 0.772 and IoU 0.688. The balanced model scored Dice 0.768 and IoU 0.681. Balanced sampling improved validation results, but not the test result on this split. These are research-project results, not clinical performance.
