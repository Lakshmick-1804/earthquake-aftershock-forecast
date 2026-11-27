"""Stage 2C: Regularized ML classifier with event-aware out-of-fold Platt calibration.

Uses strictly T0-available features (no future strike, no downsampled test evaluation,
no test-set tuning, and no test-set calibration).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from math_utils import (
    apply_platt_scaler,
    apply_standardizer,
    fit_logistic_l2,
    fit_platt_scaler,
    fit_standardizer,
    predict_logits,
)

BASE_T0_COLS = [
    "x",
    "y",
    "r_km",
    "r_norm",
    "log_r_norm",
    "inv_r2",
    "inv_r3",
    "mag",
    "length_km",
    "extent_km",
    "cell_area_km2",
    "lat0",
    "lon0",
    "abs_lat",
]

INTERACTION_FEATURE_NAMES = [
    "r_norm_sq",
    "log_r_norm_sq",
    "mag_x_log_r_norm",
    "mag_x_inv_r2",
    "abs_lat_x_log_r_norm",
    "cell_area_scaled_x_inv_r2",
]

ML_FEATURE_NAMES = BASE_T0_COLS + INTERACTION_FEATURE_NAMES


def extract_ml_features(rows: list[dict]) -> list[list[float]]:
    """Build T0-only feature vector with smooth radial/magnitude interactions."""
    feats = []
    for r in rows:
        base = [float(r[c]) for c in BASE_T0_COLS]
        r_norm = base[3]
        log_r = base[4]
        inv_r2 = base[5]
        mag = base[7]
        area = base[10]
        abs_lat = base[13]
        interactions = [
            r_norm * r_norm,
            log_r * log_r,
            mag * log_r,
            mag * inv_r2,
            abs_lat * log_r,
            (area / 1000.0) * inv_r2,
        ]
        feats.append(base + interactions)
    return feats


def _balanced_weights(y: list[int]) -> list[float]:
    n_pos = sum(y)
    n_neg = len(y) - n_pos
    w_pos = n_neg / max(n_pos, 1)
    return [w_pos if yi == 1 else 1.0 for yi in y]


def fit_calibrated_ml(train_rows: list[dict], l2: float = 10.0, n_inner_folds: int = 5) -> dict:
    """Fit regularized classifier with event-aware out-of-fold Platt calibration."""
    # Group training rows by event_id
    by_event: dict[str, list[dict]] = {}
    for r in train_rows:
        by_event.setdefault(r["event"], []).append(r)
    train_event_ids = list(by_event.keys())

    # Assign events to inner event-grouped folds (zero row-level splitting)
    inner_folds = [train_event_ids[i::n_inner_folds] for i in range(n_inner_folds)]

    oof_logits: list[float] = []
    oof_labels: list[int] = []

    for k in range(n_inner_folds):
        val_set = set(inner_folds[k])
        tr_sub = [r for eid in train_event_ids if eid not in val_set for r in by_event[eid]]
        va_sub = [r for eid in inner_folds[k] for r in by_event[eid]]

        X_tr_raw = extract_ml_features(tr_sub)
        y_tr = [int(r["label"]) for r in tr_sub]
        means, stds = fit_standardizer(X_tr_raw)
        X_tr = apply_standardizer(X_tr_raw, means, stds)

        w_tr = fit_logistic_l2(X_tr, y_tr, l2=l2, max_iter=25, weights=_balanced_weights(y_tr))

        X_va = apply_standardizer(extract_ml_features(va_sub), means, stds)
        oof_logits.extend(predict_logits(X_va, w_tr))
        oof_labels.extend(int(r["label"]) for r in va_sub)

    # Fit Platt calibrator strictly on OOF training logits
    platt_a, platt_b = fit_platt_scaler(oof_logits, oof_labels)

    # Refit final classifier on all training events
    X_all_raw = extract_ml_features(train_rows)
    y_all = [int(r["label"]) for r in train_rows]
    means_all, stds_all = fit_standardizer(X_all_raw)
    X_all = apply_standardizer(X_all_raw, means_all, stds_all)
    w_all = fit_logistic_l2(X_all, y_all, l2=l2, max_iter=25, weights=_balanced_weights(y_all))

    return {
        "means": means_all,
        "stds": stds_all,
        "weights": w_all,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "l2": l2,
    }


def predict_calibrated_ml(model: dict, rows: list[dict]) -> list[float]:
    X = apply_standardizer(extract_ml_features(rows), model["means"], model["stds"])
    logits = predict_logits(X, model["weights"])
    return apply_platt_scaler(logits, model["platt_a"], model["platt_b"])


def save_calibrated_ml(model: dict, path: str | Path, metadata: dict | None = None) -> Path:
    """Serialize fitted ML weights, standardizer stats, Platt params, and config to JSON."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_type": "l2_logistic_adam_platt_calibrated",
        "python_version": sys.version.split()[0],
        "base_t0_cols": BASE_T0_COLS,
        "feature_names": ML_FEATURE_NAMES,
        "n_features": len(ML_FEATURE_NAMES),
        "hyperparameters": {
            "l2": model["l2"],
            "optimizer": "mini_batch_adam",
            "lr": 0.08,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "batch_size": 8192,
            "epochs": 6,
            "random_seed": 42,
        },
        "standardizer": {
            "means": [float(v) for v in model["means"]],
            "stds": [float(v) for v in model["stds"]],
        },
        "weights": {
            "bias": float(model["weights"][0]),
            "coef": [float(v) for v in model["weights"][1:]],
        },
        "platt_calibration": {
            "a": float(model["platt_a"]),
            "b": float(model["platt_b"]),
        },
        "metadata": metadata or {},
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_calibrated_ml(path: str | Path) -> dict:
    """Reconstruct a fitted calibrated ML model dict from a JSON artifact without retraining."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("feature_names") != ML_FEATURE_NAMES:
        raise ValueError("Feature schema mismatch in saved model artifact")
    weights = [float(payload["weights"]["bias"])] + [float(v) for v in payload["weights"]["coef"]]
    return {
        "means": [float(v) for v in payload["standardizer"]["means"]],
        "stds": [float(v) for v in payload["standardizer"]["stds"]],
        "weights": weights,
        "platt_a": float(payload["platt_calibration"]["a"]),
        "platt_b": float(payload["platt_calibration"]["b"]),
        "l2": float(payload["hyperparameters"]["l2"]),
        "feature_names": payload["feature_names"],
    }
