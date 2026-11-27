# Saved Stage 2 Model Artifacts (`data_v2/models/`)

This directory stores serialized model artifacts for the leak-free Stage 2 Earthquake Aftershock Forecasting pipeline without modifying any existing reports or original prototype files.

## Artifact: `ml_calibrated_chrono.json`

- **Model**: L2-regularized logistic regression (`20` strictly $T_0$-available features) trained on the `89` chronological training events (`2000-03-28` to `2016-12-08`) with 3-fold event-grouped out-of-fold Platt scaling.
- **Contents**:
  - `base_t0_cols` (14 base columns from `data_v2/processed/*.csv`) and `feature_names` (all 20 input features in exact order)
  - `hyperparameters` (`l2`, optimizer, learning rate, batch size, epochs, random seed)
  - `standardizer` (`means` and `stds`, length 20)
  - `weights` (`bias` and `coef`, total 21 fitted parameters)
  - `platt_calibration` (`a` and `b`)
  - `metadata` (split name, event counts, inner folds)

## Loading and Predicting Without Retraining (Python 3.9.6+)

```python
import csv
import sys
sys.path.insert(0, "src")

from train_ml import load_calibrated_ml, predict_calibrated_ml

# 1. Reconstruct model from JSON
model = load_calibrated_ml("data_v2/models/ml_calibrated_chrono.json")

# 2. Load T0-only grid rows for any event in data_v2/processed/
with open("data_v2/processed/us6000m0xl.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# 3. Generate calibrated probabilities in (0, 1)
probs = predict_calibrated_ml(model, rows)
```
