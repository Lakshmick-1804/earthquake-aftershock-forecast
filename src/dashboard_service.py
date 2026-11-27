"""Backend service for the Stage 3 Streamlit dashboard (pure Python stdlib).

Reuses existing Stage 1 & Stage 2 modules without duplicating model logic or retraining.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from baselines import (
    fit_radial_baseline,
    predict_etas_spatial_benchmark,
    predict_radial_baseline,
)
from config_v2 import (
    GRID_N,
    V2_DATA_DIR,
    V2_INTERIM_DIR,
    V2_PROCESSED_DIR,
    V2_RAW_DIR,
    V2_REPORTS_DIR,
)
from evaluate import load_events_rows
from math_utils import spatial_hit_rate
from train_ml import load_calibrated_ml, predict_calibrated_ml

MODEL_ARTIFACT_PATH = V2_DATA_DIR / "models" / "ml_calibrated_chrono.json"


def load_catalog_and_splits() -> dict:
    """Load the 118-event catalog, split assignments, summaries, and Stage 2 evaluation report."""
    events_path = V2_RAW_DIR / "events.csv"
    splits_path = V2_REPORTS_DIR / "splits_manifest.json"
    summaries_path = V2_INTERIM_DIR / "event_summaries.json"
    eval_path = V2_REPORTS_DIR / "evaluation_report.json"

    for p in (events_path, splits_path, summaries_path, eval_path):
        if not p.exists():
            raise FileNotFoundError(f"Required Stage 1/2 artifact not found: {p}")

    with events_path.open("r", newline="", encoding="utf-8") as f:
        raw_events = {r["id"]: r for r in csv.DictReader(f)}

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    summaries = {s["id"]: s for s in json.loads(summaries_path.read_text(encoding="utf-8"))}
    eval_report = json.loads(eval_path.read_text(encoding="utf-8"))

    train_ids = set(splits["chronological_split"]["train"]["event_ids"])
    test_ids = set(splits["chronological_split"]["test"]["event_ids"])

    catalog = {}
    for eid, ev in raw_events.items():
        s = summaries.get(eid, {})
        role = "test" if eid in test_ids else ("train" if eid in train_ids else "unknown")
        catalog[eid] = {
            "id": eid,
            "time": ev["time"],
            "latitude": float(ev["latitude"]),
            "longitude": float(ev["longitude"]),
            "mag": float(ev["mag"]),
            "place": ev["place"],
            "split_role": role,
            "length_km": s.get("length_km"),
            "extent_km": s.get("extent_km"),
            "raw_total": s.get("raw_total"),
            "inside_grid_aftershocks": s.get("inside_grid_aftershocks"),
            "outside_grid_aftershocks": s.get("outside_grid_aftershocks"),
            "positive_cells": s.get("positive_cells"),
            "positive_rate": s.get("positive_rate"),
        }

    return {
        "catalog": catalog,
        "splits": splits,
        "eval_report": eval_report,
    }


def load_models(splits: dict, eval_report: dict) -> dict:
    """Load serialized ML model and ETAS-inspired params; fit radial baseline on train split."""
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Serialized ML artifact not found: {MODEL_ARTIFACT_PATH}")

    ml_model = load_calibrated_ml(MODEL_ARTIFACT_PATH)

    etas_p = eval_report["chronological_holdout"]["fitted_params"]["etas_spatial_benchmark"]
    etas_model = {
        "b0": float(etas_p["b0"]),
        "alpha": float(etas_p["alpha"]),
        "d2": float(etas_p["d2"]),
        "q": float(etas_p["q"]),
        "raw_params": [
            float(etas_p["b0"]),
            float(etas_p["alpha"]),
            math.log(float(etas_p["d2"])),
            math.log(float(etas_p["q"])),
        ],
    }

    train_ids = splits["chronological_split"]["train"]["event_ids"]
    tr_rows = load_events_rows(train_ids)
    radial_model = fit_radial_baseline(tr_rows)

    return {
        "radial_baseline": radial_model,
        "etas_spatial_benchmark": etas_model,
        "ml_calibrated": ml_model,
    }


def get_event_grid_predictions(event_id: str, models: dict) -> dict:
    """Load T0-processed grid for event_id and compute all three model probability maps."""
    proc_path = V2_PROCESSED_DIR / f"{event_id}.csv"
    if not proc_path.exists():
        raise FileNotFoundError(f"Processed grid file not found for event '{event_id}': {proc_path}")

    with proc_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    p_rad = predict_radial_baseline(models["radial_baseline"], rows)
    p_etas = predict_etas_spatial_benchmark(models["etas_spatial_benchmark"], rows)
    p_ml = predict_calibrated_ml(models["ml_calibrated"], rows)

    counts = [int(r["aftershock_count"]) for r in rows]

    # Compute 1-based descending probability ranks per model
    def _ranks(probs: list[float]) -> list[int]:
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        rks = [0] * len(probs)
        for rk, idx in enumerate(order, start=1):
            rks[idx] = rk
        return rks

    rk_rad = _ranks(p_rad)
    rk_etas = _ranks(p_etas)
    rk_ml = _ranks(p_ml)

    enriched = []
    for i, r in enumerate(rows):
        enriched.append({
            "cell_id": int(r["cell_id"]),
            "row_idx": int(r["row_idx"]),
            "col_idx": int(r["col_idx"]),
            "x": float(r["x"]),
            "y": float(r["y"]),
            "r_km": float(r["r_km"]),
            "aftershock_count": counts[i],
            "label": int(r["label"]),
            "p_radial": p_rad[i],
            "p_etas": p_etas[i],
            "p_ml": p_ml[i],
            "rank_radial": rk_rad[i],
            "rank_etas": rk_etas[i],
            "rank_ml": rk_ml[i],
        })

    hit_rates = {}
    for key, plist in (("radial_baseline", p_rad), ("etas_spatial_benchmark", p_etas), ("ml_calibrated", p_ml)):
        hit_rates[key] = {
            "hit_5": spatial_hit_rate(counts, plist, 0.05),
            "hit_10": spatial_hit_rate(counts, plist, 0.10),
            "hit_20": spatial_hit_rate(counts, plist, 0.20),
        }

    return {
        "event_id": event_id,
        "grid_n": GRID_N,
        "rows": enriched,
        "hit_rates": hit_rates,
    }


def verify_prediction_consistency(models: dict, splits: dict) -> dict:
    """Verify dashboard model predictions match saved data_v2/reports/predictions_chrono.csv."""
    pred_csv = V2_REPORTS_DIR / "predictions_chrono.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(f"Saved predictions CSV not found: {pred_csv}")

    with pred_csv.open("r", newline="", encoding="utf-8") as f:
        saved_rows = list(csv.DictReader(f))

    te_ids = splits["chronological_split"]["test"]["event_ids"]
    te_rows = load_events_rows(te_ids)

    p_rad = predict_radial_baseline(models["radial_baseline"], te_rows)
    p_etas = predict_etas_spatial_benchmark(models["etas_spatial_benchmark"], te_rows)
    p_ml = predict_calibrated_ml(models["ml_calibrated"], te_rows)

    return {
        "n_test_rows": len(te_rows),
        "max_diff_radial": max(abs(a - float(r["p_radial"])) for a, r in zip(p_rad, saved_rows)),
        "max_diff_etas": max(abs(a - float(r["p_etas"])) for a, r in zip(p_etas, saved_rows)),
        "max_diff_ml": max(abs(a - float(r["p_ml"])) for a, r in zip(p_ml, saved_rows)),
    }


def get_cell(rows: list[dict], row_idx: int, col_idx: int) -> dict:
    """Return the grid cell at (row_idx, col_idx) after validating 0..GRID_N-1 bounds."""
    if (
        isinstance(row_idx, bool)
        or isinstance(col_idx, bool)
        or not isinstance(row_idx, int)
        or not isinstance(col_idx, int)
    ):
        raise ValueError(f"row_idx and col_idx must be integers, got ({row_idx!r}, {col_idx!r})")
    if not (0 <= row_idx < GRID_N and 0 <= col_idx < GRID_N):
        raise ValueError(f"Grid indices out of range [0, {GRID_N - 1}]: ({row_idx}, {col_idx})")
    for r in rows:
        if r["row_idx"] == row_idx and r["col_idx"] == col_idx:
            return r
    raise ValueError(f"Grid cell ({row_idx}, {col_idx}) not found in grid rows")
