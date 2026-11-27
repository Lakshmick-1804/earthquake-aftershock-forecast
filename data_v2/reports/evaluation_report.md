# Stage 2 Evaluation Report

## 1. Chronological Holdout (89 Train [2000-2016] / 29 Test [2017-2024])
- Test cells: 46,400 across 29 held-out events (positive prevalence: 5.4655%)

### Cell-Level Pooled Metrics (with 95% Event-Cluster Bootstrap CIs)

| Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Brier Score (95% CI) | Log Loss (95% CI) |
|---|---|---|---|---|
| `radial_baseline` | 0.3341 [0.2687, 0.3926] | 0.8497 [0.8115, 0.8815] | 0.04300 [0.03271, 0.05562] | 0.1615 [0.1263, 0.2062] |
| `etas_spatial_benchmark` | 0.3326 [0.2647, 0.3944] | 0.8459 [0.7990, 0.8840] | 0.04368 [0.03312, 0.05674] | 0.1648 [0.1277, 0.2116] |
| `ml_calibrated` | 0.2988 [0.2256, 0.3725] | 0.8155 [0.7425, 0.8702] | 0.04441 [0.03373, 0.05855] | 0.1699 [0.1314, 0.2241] |

### Event-Level Mean Metrics (across 29 Test Events)

| Model | Event PR-AUC | Event ROC-AUC | Event Brier | Hit@5% | Hit@10% | Hit@20% |
|---|---:|---:|---:|---:|---:|---:|
| `radial_baseline` | 0.3986 | 0.8701 | 0.04300 | 53.97% | 68.98% | 81.11% |
| `etas_spatial_benchmark` | 0.3986 | 0.8701 | 0.04368 | 53.97% | 68.99% | 81.05% |
| `ml_calibrated` | 0.3963 | 0.8677 | 0.04441 | 53.98% | 68.92% | 81.12% |

## 2. Secondary 5-Fold Event-Grouped CV (All 118 Events)

| Model | Pooled PR-AUC | Pooled ROC-AUC | Pooled Brier | Pooled Log Loss | Event Hit@10% |
|---|---:|---:|---:|---:|---:|
| `radial_baseline` | 0.3329 | 0.8254 | 0.04135 | 0.1595 | 69.75% |
| `etas_spatial_benchmark` | 0.3309 | 0.8205 | 0.04154 | 0.1608 | 69.75% |
| `ml_calibrated` | 0.3221 | 0.8133 | 0.04183 | 0.1626 | 69.59% |
