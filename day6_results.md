# Grouped Split Results — Day 6

Date: 2026-11-17

## Data split

- Images: 647
- Duplicate groups kept together: 589
- Training / validation / test: 449 / 100 / 98 images
- Split assignments: `split_manifest.csv`

## Threshold selection

Thresholds were compared using the validation set only. The selected threshold was **0.30** (validation Dice **0.819**, IoU **0.729**).

## Held-out test results

- At threshold 0.50: Dice **0.768**, IoU **0.682**
- At selected threshold 0.30: Dice **0.772**, IoU **0.688**

The test set was evaluated at 0.30 after the threshold was selected on validation. The improvement was small.

## Failure-case review

The eight lowest-Dice test cases at threshold 0.30 had Dice scores:

- Six cases: **0.000**
- One case: **0.062**
- One case: **0.090**

The lower threshold barely changed the hardest cases. The visualization is saved as `worst8_grouped_tuned.png`.

## Current setting

Keep `best_grouped_split.pt` and threshold **0.30** as the current grouped-split result. Use the validation set for any further model experiments; do not tune the threshold against the test set.
