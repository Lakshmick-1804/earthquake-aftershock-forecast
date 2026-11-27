# Earthquake Aftershock Forecasting: A Leak-Free Machine Learning and Spatial Benchmark Study

---

## 1. Title Page Information

- **Project Title:** Earthquake Aftershock Forecasting — Leak-Free Spatial Probability Modeling on a Fixed 118-Event Catalog
- **Student Name(s):** `[Student Name 1]`, `[Student Name 2]`, `[Student Name 3]`
- **USN(s):** `[USN 1]`, `[USN 2]`, `[USN 3]`
- **Department / Institution:** `[Department Name]`, `[College / University Name]`
- **Project Guide:** `[Guide Name, Designation]`
- **Academic Year:** `[2025–2026 / 2026–2027]`

---

## 2. Abstract

Machine learning models for spatial aftershock forecasting can appear artificially accurate when post-mainshock information leaks into prediction-time features or when evaluation splits fail to respect earthquake-level and temporal boundaries. In this project, we audit and redesign an existing 118-earthquake aftershock forecasting prototype while strictly retaining the original 118-event dataset (`data/events.csv` and `data/raw/`). We eliminate future-strike leakage, isolate strictly prediction-time ($T_0$) features, separate 30-day future aftershock records exclusively for target labeling, and implement a reproducible pipeline in Python featuring a chronological holdout (`89` training events `2000–2016` / `29` test events `2016–2024`), a secondary 5-fold event-grouped cross-validation, an Isotropic Radial Baseline, an ETAS-inspired spatial benchmark, and an L2-regularized logistic regression classifier calibrated via event-aware out-of-fold Platt scaling. On the chronological test set (`46,400` grid cells), the Isotropic Radial Baseline achieves `0.3341` PR-AUC (`0.8497` ROC-AUC, `0.04300` Brier), the ETAS-inspired spatial benchmark achieves `0.3326` PR-AUC (`0.8459` ROC-AUC, `0.04368` Brier), and the calibrated ML model achieves `0.2988` PR-AUC (`0.8155` ROC-AUC, `0.04441` Brier), with all three models capturing ~69% of in-grid aftershocks in the top 10% highest-probability cells. We serialize the fitted ML model to JSON, verify exact numerical reconstruction, and build a decoupled Streamlit dashboard (`src/app_v2.py`) while preserving all original prototype files.

---

## 3. Introduction

Large earthquakes ($M \ge 7.0$) frequently trigger damaging aftershock sequences over the days and weeks following the mainshock. Forecasting *where* aftershocks are most likely to occur within a local spatial grid around the epicenter is a central problem in statistical seismology and hazard communication. While classical geophysical models rely on empirical scaling laws and spatial triggering kernels, machine learning (ML) approaches attempt to learn spatial probability maps from historical catalogs. However, ML pipelines must obey strict causality: only information available at the mainshock origin time ($T_0$) may be used as model inputs, and evaluation must test generalization to unseen future earthquakes.

---

## 4. Problem Statement

The original project prototype reported an average ROC-AUC of `0.8806` across 118 earthquakes, but a scientific audit revealed critical methodological flaws:
1. **Future-strike leakage:** Fault strike was estimated via PCA on the *future 30-day aftershock cloud* and used to orient input stress-proxy features.
2. **Selection bias:** Mainshocks were retained only if they produced at least `20` recorded M4.0+ aftershocks (`12` `M7.0+` candidates with `2–18` aftershocks were excluded).
3. **Uncalibrated scores & overwritten features:** `train.py` downsampled training negatives (`1:3`) without probability calibration, compared against a signed baseline score (`[-2, 2]`), and overwrote `data/processed/*.csv` with prediction columns.
4. **Non-chronological evaluation:** Only 5-fold `GroupKFold` was used, without testing forward-in-time generalization.

The problem addressed in this project is to **redesign and validate a scientifically sound, reproducible, leak-free forecasting system using the exact same 118 earthquakes**.

---

## 5. Objectives and Scope

### Objectives
1. Audit and preserve the fixed 118-earthquake dataset (`data/events.csv`, `data/raw/`) without downloading new events or modifying original files.
2. Construct strictly $T_0$-available input features and separate future 30-day aftershock counts/labels.
3. Design an event-grouped chronological validation split (`89` train / `29` test) plus secondary 5-fold event-grouped CV.
4. Implement and fairly compare an Isotropic Radial Baseline, an ETAS-inspired spatial benchmark, and a calibrated ML classifier using proper scoring rules (PR-AUC, Brier score, Log Loss), ROC-AUC, and spatial hit rates.
5. Serialize the fitted ML model for zero-retraining reproducibility and build a modular Streamlit dashboard (`src/app_v2.py`).

### Scope
- **In scope:** Spatial binary occurrence forecasting ($P(\text{aftershock count} \ge 1)$) over 30 days on a $40 \times 40$ grid around each of the 118 mainshocks.
- **Out of scope:** Operational real-time alerting, downloading external focal-mechanism/depth catalogs, or full space-time ETAS background modeling.

---

## 6. Existing Prototype and Its Limitations

Inspection of `config.py`, `data.py`, `features.py`, `train.py`, `data/meta.json`, and `data/summary.json` established:
- `data/raw/` contains `130` CSV files, of which `118` are in `data/events.csv` and `12` (`2–18` aftershocks) were excluded by `MIN_AFTERSHOCKS = 20`.
- `features.py` called `estimate_strike(ax, ay)` on future 30-day aftershocks to compute `f1, f2, f3, u, v`.
- `train.py` overwrote `data/processed/<id>.csv` with `['x', 'y', 'label', 'nn', 'base']`, erasing input features and storing raw `f1` scores (`[-2, 2]`, `94,400` negative values) in `base`.
- `summary.json` reported `auc_nn = 0.8806` and `auc_base = 0.5073`, which reflected future-strike leakage and uncalibrated scores rather than true $T_0$ forecasting skill.

---

## 7. Proposed System and Improvements

The improved system (`src/` and `data_v2/`) isolates all new code and outputs from the original prototype:
- **Immutable provenance:** `data_v2/raw/` holds SHA-256-verified copies of `events.csv` and the 118 selected raw CSVs.
- **Leak-free features:** Eliminates future strike and uses 20 strictly $T_0$-available isotropic, magnitude, and coordinate features.
- **Clean data separation:** `data_v2/processed/` stores features and labels immutably; predictions and reports are written separately to `data_v2/reports/`.
- **Proper calibration & metrics:** Trains on unmodified class distributions (with training-only balanced weights and 3-fold OOF Platt scaling) and evaluates using PR-AUC, ROC-AUC, Brier score, Log Loss, and spatial Hit@5%/10%/20%.

---

## 8. Dataset and Data Provenance

| Property | Verified Value |
|---|---|
| Source | USGS FDSNWS catalog query (`config.py`, `data.py`) |
| Candidate Query | `M7.0+` (`MIN_MAINSHOCK_MAG = 7.0`), `2000-01-01` to `2024-12-31` |
| Selected Mainshocks | `118` unique events (`2000-03-28T11:00:22.510Z` to `2024-01-01T07:10:09.476Z`) |
| Realized Magnitudes | `M7.5` to `M9.1` |
| Excluded Candidates | `12` `M7.0+` raw files (`2–18` aftershocks `< 20`) |
| Raw Aftershock Records | `37,289` rows across the 118 selected events (`20–3,129` per event; `mag` `4.00–8.20`) |
| Mainshock Duplicates in Raw | `0` rows |
| Grid Coverage | `35,555` (`95.35%`) inside `[-E, E] × [-E, E]`; `1,734` (`4.65%`) outside |
| Total Grid Cells | `188,800` (`118 × 1,600`), with `9,905` (`5.2463%`) positive cells (`13–658` per event, median `68.0`) |

---

## 9. Data Preprocessing and Spatial Grid Construction

Implemented in `src/prepare_data.py` and `src/features_t0.py`:
1. **Rupture-length & grid scaling:** Computes an empirical Wells & Coppersmith (1994) rupture-length estimate $L = 10^{-2.44 + 0.59 M}$ (not a measured fault trace) and grid half-width $E = \max(80.0, 2.0 \cdot L)$ km.
2. **Local projection:** Projects $(\text{lat}, \text{lon})$ to local East/North km offsets $(x, y)$ relative to $(\text{lat}_0, \text{lon}_0)$ over a $40 \times 40$ grid (`GRID_N = 40`, `1,600` cells/event).
3. **Target labeling:** Filters `data_v2/raw/<id>.csv` to `0 < dt_days <= 30.0` (excluding any mainshock row), bins aftershocks onto the $40 \times 40$ grid to record `aftershock_count`, and sets `label = 1` if `aftershock_count > 0`, else `0`.

---

## 10. $T_0$-Only Feature Engineering

All 20 features in `ML_FEATURE_NAMES` (`src/train_ml.py`) depend strictly on mainshock parameters known at $T_0$:
- **14 Base $T_0$ Columns (`BASE_T0_COLS`):** `x`, `y`, `r_km`, `r_norm`, `log_r_norm`, `inv_r2`, `inv_r3`, `mag`, `length_km`, `extent_km`, `cell_area_km2`, `lat0`, `lon0`, `abs_lat`.
- **6 Derived $T_0$ Interactions (`INTERACTION_FEATURE_NAMES`):** `r_norm_sq`, `log_r_norm_sq`, `mag_x_log_r_norm`, `mag_x_inv_r2`, `abs_lat_x_log_r_norm`, `cell_area_scaled_x_inv_r2`.

Identifiers (`event`, `cell_id`, `row_idx`, `col_idx`) and future targets (`aftershock_count`, `label`) are strictly excluded.

---

## 11. Mathematical Formulation of All Three Models

1. **Isotropic Radial Baseline (`src/baselines.py`):**
   L2-regularized logistic regression (`l2 = 1.0`, Newton-IRLS) on standardized $[\text{log\_r\_norm}, \text{mag}]$:
   $$P(Y_{i,c}=1) = \sigma\!\left(w_0 + w_1 \tilde{z}_{\log r} + w_2 \tilde{z}_{M}\right)$$

2. **ETAS-Inspired Spatial Benchmark (`src/baselines.py`):**
   *Not a complete space-time ETAS model.* Defines expected 30-day cell intensity $\lambda_{i,c}$ using Utsu-style magnitude scaling, power-law spatial decay, and scaled cell area $A_{i,c} = \text{cell\_area\_km2}/1000$:
   $$\lambda_{i,c} = \exp\!\big(\beta_0 + \alpha (M_i - 7.5)\big) \cdot (r_{\text{norm}}^2 + d^2)^{-q} \cdot A_{i,c}, \qquad P(Y_{i,c}=1) = 1 - \exp(-\lambda_{i,c})$$
   Fitted parameters on the 89 training events: $\beta_0 = 0.4772$, $\alpha = -1.9221$, $d^2 = 0.0446$, $q = 0.9291$.

3. **Calibrated ML Classifier (`src/train_ml.py`):**
   L2-regularized logistic regression (`l2 = 10.0`) on 20 standardized $T_0$ features, trained via mini-batch Adam (`lr = 0.08`, `beta1 = 0.9`, `beta2 = 0.999`, `batch_size = 8192`, `epochs = 6`, `seed = 42`) with training-only balanced weights (`w_pos = n_neg / n_pos`). Outputs are calibrated via Platt scaling:
   $$P(Y_{i,c}=1) = \sigma(a \cdot z_{i,c} + b), \qquad a = 0.8250,\; b = -2.8907$$

---

## 12. Training and Calibration Methodology

- **Zero test-set leakage:** Standardization (`fit_standardizer`), class weighting, model fitting, and Platt calibration are performed strictly inside the training set.
- **Event-aware OOF Platt calibration:** Within the 89 training events, events are partitioned into 3 event-grouped inner folds (`n_inner_folds = 3`). Out-of-fold logits are collected on held-out inner-fold events and used to fit `(a, b)` via 1D Newton-IRLS (`l2 = 1e-4`). The final classifier is refit on all 89 training events, and `(a, b)` is applied to test logits.

---

## 13. Experimental Design and Validation

- **Primary Chronological Split (`data_v2/reports/splits_manifest.json`):**
  - Train: `89` events (`2000-03-28` to `2016-12-08`, `142,400` cells, `5.1749%` positive prevalence).
  - Test: `29` events (`2016-12-17` to `2024-01-01`, `46,400` cells, `5.4655%` positive prevalence).
- **Secondary 5-Fold Event-Grouped CV:** Interleaved chronological folds across all `118` events (`24, 24, 24, 23, 23` test events; `0` event overlap).
- **Uncertainty Estimation:** 95% confidence intervals computed via `100`-resample event-cluster bootstrap (`seed = 42`).

---

## 14. Results with Tables and Interpretation

### Table 1: Chronological Holdout — Cell-Level Pooled Metrics (29 Test Events, 46,400 Cells)

| Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Brier Score (95% CI) | Log Loss (95% CI) |
|---|---|---|---|---|
| `radial_baseline` | **0.3341** [0.2687, 0.3926] | **0.8497** [0.8115, 0.8815] | **0.04300** [0.03271, 0.05562] | **0.1615** [0.1263, 0.2062] |
| `etas_spatial_benchmark` | 0.3326 [0.2647, 0.3944] | 0.8459 [0.7990, 0.8840] | 0.04368 [0.03312, 0.05674] | 0.1648 [0.1277, 0.2116] |
| `ml_calibrated` | 0.2988 [0.2256, 0.3725] | 0.8155 [0.7425, 0.8702] | 0.04441 [0.03373, 0.05855] | 0.1699 [0.1314, 0.2241] |

*(Note: ROC-AUC measures ranking discrimination across thresholds and is not classification accuracy.)*

### Table 2: Chronological Holdout — Event-Level Mean Metrics (29 Test Events)

| Model | Event PR-AUC | Event ROC-AUC | Event Brier | Hit@5% | Hit@10% | Hit@20% |
|---|---:|---:|---:|---:|---:|---:|
| `radial_baseline` | **0.3986** | **0.8701** | **0.04300** | 53.97% | 68.98% | 81.11% |
| `etas_spatial_benchmark` | **0.3986** | **0.8701** | 0.04368 | 53.97% | **68.99%** | 81.05% |
| `ml_calibrated` | 0.3963 | 0.8677 | 0.04441 | **53.98%** | 68.92% | **81.12%** |

### Table 3: Secondary 5-Fold Event-Grouped CV (All 118 Events, 188,800 Cells)

| Model | Pooled PR-AUC | Pooled ROC-AUC | Pooled Brier | Pooled Log Loss | Event Hit@10% |
|---|---:|---:|---:|---:|---:|
| `radial_baseline` | **0.3329** | **0.8254** | **0.04135** | **0.1595** | **69.75%** |
| `etas_spatial_benchmark` | 0.3309 | 0.8205 | 0.04154 | 0.1608 | **69.75%** |
| `ml_calibrated` | 0.3221 | 0.8133 | 0.04183 | 0.1626 | 69.59% |

*(95% bootstrap CIs were not computed for the 5-fold CV table in `evaluation_report.json` and are therefore not reported.)*

### Interpretation
On the chronological test set, the radial baseline and ETAS-inspired spatial benchmark achieve slightly better pooled PR-AUC and calibration scores than the calibrated ML model. This indicates that the additional interaction features did not improve held-out performance in this experiment. The absence of depth and fault-plane geometry may limit the additional predictive information available to the ML model.

---

## 15. Dashboard Architecture and Implementation

- **Backend (`src/dashboard_service.py`):** Pure-Python stdlib service handling catalog/split loading, `load_calibrated_ml()` reconstruction, ETAS-inspired parameter loading, deterministic `radial_baseline` fitting, per-event probability/rank calculation, and defensive cell lookup (`get_cell()`).
- **Frontend (`src/app_v2.py`):** 5-tab Streamlit application (*Overview*, *Explorer & Forecast Map*, *Model Comparison*, *Prediction Analysis*, *Methodology & Limitations*) using `streamlit`, `matplotlib`, and `pandas` without altering `app.py`.

---

## 16. Testing and Verification

- **Model serialization (`src/export_model.py`):** Reconstructed `ml_calibrated_chrono.json` predictions match the in-memory model within `0.000e+00` (`<= 1e-8`) and `predictions_chrono.csv` within `5.000e-07` across `46,400` rows.
- **Stage 4 automated tests:** Verified `118` events (`89`/`29` split), `0` NaN/Inf/out-of-range probabilities across `188,800` cells, `<= 1e-6` prediction consistency for all 3 models, boundary cells `(0,0)`, `(0,39)`, `(39,0)`, `(39,39)`, and `ValueError` handling in `get_cell()`.
- **Live UI verification:** Syntax compilation (`py_compile`) passed, and the live Streamlit dashboard (`src/app_v2.py`) launched successfully at `http://localhost:8501/` and was manually tested across all five tabs (*Overview*, *Explorer & Forecast Map*, *Model Comparison*, *Prediction Analysis*, and *Methodology & Limitations*). Backend automated testing passed all seven categories. Fresh-machine installation and reproducibility have not been independently verified.

---

## 17. Limitations

1. **Selection Bias:** All 118 events had $\ge 20$ M4.0+ aftershocks (`12` `M7.0+` candidates excluded).
2. **Grid Boundary:** Forecasts cover only `[-E, E] × [-E, E]` (`95.35%` of recorded aftershocks; `4.65%` outside).
3. **Missing Source Parameters:** `depth`, `strike`, `dip`, and `rake` are absent from `events.csv` and not fabricated.
4. **Observation-Window Coverage:** Original USGS query logs/headers were not saved; complete 30-day catalog coverage has not been established for every event (`44` events have last recorded aftershock `< 29.0` days, minimum `12.292` days for `usp000c2ka`).
5. **ETAS-Inspired Scope:** Not a full space-time ETAS model (no pre-$T_0$ background seismicity rate $\mu$).
6. **Non-Operational Disclaimer:** Research prototype only; not suitable for operational earthquake forecasting or emergency decisions.

---

## 18. Future Enhancements

- Integrate real-time $T_0$ focal mechanisms (strike, dip, rake) and hypocentral depth to break radial symmetry legitimately.
- Incorporate pre-$T_0$ regional seismicity catalogs to support a complete space-time ETAS model with local background rate $\mu(x, y)$.
- Retain raw API query metadata to verify 30-day observation completeness across all events.

---

## 19. Conclusion

By auditing and removing future-strike leakage, row-level splitting, and uncalibrated scoring from the original prototype while preserving the exact 118-earthquake dataset, this project establishes a scientifically rigorous benchmark for spatial aftershock forecasting. The results demonstrate that without fault-plane geometry or depth at $T_0$, simple isotropic radial and ETAS-inspired spatial benchmarks remain competitive with or slightly stronger than higher-dimensional ML interaction models, underscoring the importance of physical features and leak-free evaluation.

---

## 20. References

1. Wells, D. L., & Coppersmith, K. J. (1994). New empirical relationships among magnitude, rupture length, rupture width, rupture area, and surface displacement. *Bulletin of the Seismological Society of America*, 84(4), 974–1002.
2. Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*, 61–74.
3. U.S. Geological Survey (USGS) Earthquake Hazards Program — FDSN Event Web Service (`https://earthquake.usgs.gov/fdsnws/event/1/query`).
4. Project Artifacts: `data_v2/reports/data_validation_report.json`, `data_v2/reports/splits_manifest.json`, `data_v2/reports/evaluation_report.json`, and `data_v2/models/ml_calibrated_chrono.json`.
