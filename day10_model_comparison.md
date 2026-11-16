Both models used the same duplicate-aware split: 449 train, 100 validation, 98 test images.



Original model:

\- Selected threshold: 0.30

\- Validation Dice / IoU: 0.819 / 0.729

\- Test Dice / IoU: 0.772 / 0.688



Balanced-sampling model:

\- Selected threshold: 0.35

\- Validation Dice / IoU: 0.835 / 0.745

\- Test Dice / IoU: 0.768 / 0.681



Conclusion: balanced sampling improved validation results but did not improve the held-out test result on this split. Keep both checkpoints and report both results. Use validation data to guide any future experiments.

