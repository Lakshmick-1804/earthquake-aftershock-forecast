"""Train the bundled local aftershock estimator from a USGS CSV catalog.

The exported model is a small regression random forest. It predicts the number
of M>=3 aftershocks in a forecast window, using only catalog-derived features
that are also available to the browser app. No external service is used at
runtime; the browser downloads assets/ml-model.json and evaluates the trees.

This is a teaching prototype. The held-out metrics are reported honestly and
are not an operational validation or an accuracy claim.
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MODEL_PATH = ASSETS / "ml-model.json"
TRAINING_CSV_PATH = ASSETS / "usgs-all-month-training.csv"
FEED_URL = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"

OMORI_C = 0.05
OMORI_P = 1.1

START_HOURS = (0.0, 6.0, 24.0, 72.0)
WINDOW_HOURS = (24.0, 168.0, 720.0)
MIN_MAIN_MAG = 4.5
TARGET_MAG = 3.0
TEST_FRACTION = 0.2
RANDOM_STATE = 42


def omori_integral(start_days: float, end_days: float) -> float:
    """Same Omori–Utsu integral used by the browser baseline."""
    return (
        math.pow(start_days + OMORI_C, 1 - OMORI_P)
        - math.pow(end_days + OMORI_C, 1 - OMORI_P)
    ) / (1 - OMORI_P)


OMORI_NORM = omori_integral(0.0, 30.0)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def parse_time(value: str) -> float:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def local_km(lat: float, lon: float, main_lat: float, main_lon: float) -> tuple[float, float]:
    lat_scale = 111.32
    lon_scale = 111.32 * max(0.05, math.cos(math.radians(main_lat)))
    return (lon - main_lon) * lon_scale, (lat - main_lat) * lat_scale


def infer_trend(events: list[dict], main: dict) -> tuple[float, float, float]:
    """Return angle in degrees, fault length in km, and anisotropy strength."""
    if len(events) < 3:
        fallback = 12.0 if main["mag"] >= 7 else -6.0
        length = clamp(40.0 + (main["mag"] - 5.0) * 10.0, 40.0, 140.0)
        return fallback, length, 0.0

    sample = events[:80]
    mean_x = sum(event["x"] for event in sample) / len(sample)
    mean_y = sum(event["y"] for event in sample) / len(sample)

    cov_xx = sum((event["x"] - mean_x) ** 2 for event in sample) / len(sample)
    cov_yy = sum((event["y"] - mean_y) ** 2 for event in sample) / len(sample)
    cov_xy = sum((event["x"] - mean_x) * (event["y"] - mean_y) for event in sample) / len(sample)

    angle = math.degrees(0.5 * math.atan2(2 * cov_xy, cov_xx - cov_yy))
    half = (cov_xx + cov_yy) / 2
    radius = math.sqrt(max(0.0, ((cov_xx - cov_yy) / 2) ** 2 + cov_xy**2))
    eigen_max = half + radius
    eigen_min = max(0.0, half - radius)
    length = clamp(math.sqrt(max(0.0, eigen_max)) * 4, 40.0, 150.0)
    strength = 0.0 if eigen_max + eigen_min == 0 else (eigen_max - eigen_min) / (eigen_max + eigen_min)
    return angle, length, strength


def read_catalog(text: str) -> list[dict]:
    rows: list[dict] = []
    for raw in csv.DictReader(io.StringIO(text.lstrip("\ufeff"))):
        if (raw.get("type") or "").strip().lower() != "earthquake":
            continue
        try:
            time = parse_time(raw["time"])
            lat = float(raw["latitude"])
            lon = float(raw["longitude"])
            depth = float(raw["depth"])
            mag = float(raw["mag"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (time, lat, lon, depth, mag)):
            continue
        rows.append(
            {
                "time": time,
                "lat": lat,
                "lon": lon,
                "depth": depth,
                "mag": mag,
                "id": raw.get("id") or "",
            }
        )
    rows.sort(key=lambda row: row["time"])
    return rows


def events_near(rows: list[dict], main: dict, radius: float, after_hours: float, before_hours: float) -> list[dict]:
    """Events after `after_hours` and up to `before_hours` relative to mainshock."""
    selected: list[dict] = []
    lower = main["time"] + after_hours * 3600
    upper = main["time"] + before_hours * 3600
    for row in rows:
        if row is main or row["time"] <= main["time"] or row["time"] > upper:
            continue
        if row["time"] < lower:
            continue
        x, y = local_km(row["lat"], row["lon"], main["lat"], main["lon"])
        distance = math.hypot(x, y)
        if distance <= radius:
            selected.append({"x": x, "y": y, "mag": row["mag"], "time": row["time"]})
    return selected


def build_samples(rows: list[dict]) -> tuple[list[list[float]], list[float], list[str], dict]:
    if not rows:
        raise ValueError("catalog is empty")

    latest = rows[-1]["time"]
    mainshocks = [row for row in rows if row["mag"] >= MIN_MAIN_MAG]

    features: list[list[float]] = []
    targets: list[float] = []
    groups: list[str] = []

    for index, main in enumerate(mainshocks):
        available_hours = (latest - main["time"]) / 3600
        if available_hours <= 0:
            continue

        radius = clamp(60.0 + (main["mag"] - 5.0) * 45.0, 80.0, 220.0)
        mag_factor = math.pow(10, 0.8 * (main["mag"] - 5.0))
        depth_factor = 1.0 + clamp(main["depth"], 0.0, 700.0) / 500.0

        for start in START_HOURS:
            if start > available_hours:
                continue

            early = events_near(rows, main, radius, 0.0, start)
            early.sort(key=lambda event: event["time"])
            early_m3 = sum(1 for event in early if event["mag"] >= TARGET_MAG)
            early_all = len(early)
            early_max_mag = max((event["mag"] for event in early), default=main["mag"])
            angle, fault_length, trend_strength = infer_trend(early, main)

            baseline_early = (
                12.0
                * mag_factor
                * depth_factor
                * (omori_integral(0.0, start / 24) / OMORI_NORM)
                if start > 0
                else 0.0
            )
            productivity_scale = (
                clamp(early_m3 / max(0.25, baseline_early), 0.2, 5.0)
                if start > 0 and early_m3 > 0
                else 1.0
            )
            early_m3_per_hour = early_m3 / max(start, 1.0)

            for window in WINDOW_HOURS:
                if start + window > available_hours:
                    continue
                target_events = events_near(rows, main, radius, start, start + window)
                target = sum(1 for event in target_events if event["mag"] >= TARGET_MAG)

                omori_fraction = omori_integral(start / 24, (start + window) / 24) / OMORI_NORM
                baseline_expected = 3.0 + 12.0 * mag_factor * depth_factor * omori_fraction
                baseline_prediction = baseline_expected * productivity_scale

                theta = math.radians(angle)
                features.append(
                    [
                        main["mag"],
                        clamp(main["depth"], 0.0, 700.0),
                        start,
                        window,
                        radius,
                        math.log10(mag_factor),
                        depth_factor,
                        omori_fraction,
                        math.log1p(early_m3),
                        math.log1p(early_all),
                        early_m3_per_hour,
                        max(0.0, early_max_mag - main["mag"]),
                        productivity_scale,
                        fault_length / 100.0,
                        math.cos(theta),
                        math.sin(theta),
                        trend_strength,
                        math.log1p(baseline_expected),
                        math.log1p(baseline_prediction),
                    ]
                )
                targets.append(float(target))
                groups.append(main["id"] or f"main-{index}")

        if (index + 1) % 100 == 0:
            print(f"prepared {index + 1}/{len(mainshocks)} mainshocks", flush=True)

    return features, targets, groups, {"mainshocks": len(mainshocks), "latest": latest}


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.clip(np.asarray(predicted, dtype=float), 0.0, 600.0)
    probability = 1.0 - np.exp(-predicted)
    binary = (actual > 0).astype(float)
    probability = np.clip(probability, 1e-6, 1 - 1e-6)
    return {
        "mae": round(float(np.mean(np.abs(actual - predicted))), 4),
        "rmse": round(float(np.sqrt(np.mean((actual - predicted) ** 2))), 4),
        "brier": round(float(np.mean((probability - binary) ** 2)), 4),
        "logLoss": round(
            float(
                -np.mean(
                    binary * np.log(probability) + (1 - binary) * np.log(1 - probability)
                )
            ),
            4,
        ),
    }


def export_tree(estimator) -> dict:
    tree = estimator.tree_
    values = tree.value
    return {
        "l": [int(node) for node in tree.children_left],
        "r": [int(node) for node in tree.children_right],
        "f": [int(node) for node in tree.feature],
        "t": [round(float(node), 6) for node in tree.threshold],
        "v": [round(float(values[i, 0, 0]), 6) for i in range(tree.node_count)],
    }


def fetch_training_csv() -> str:
    if TRAINING_CSV_PATH.exists():
        return TRAINING_CSV_PATH.read_text(encoding="utf-8")
    print(f"downloading {FEED_URL}", flush=True)
    with urllib.request.urlopen(FEED_URL, timeout=60) as response:
        text = response.read().decode("utf-8")
    TRAINING_CSV_PATH.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    text = fetch_training_csv()
    rows = read_catalog(text)
    features, targets, groups, info = build_samples(rows)
    if len(features) < 100:
        raise ValueError(f"not enough training samples: {len(features)}")

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    y_log = np.log1p(y)
    groups_array = np.asarray(groups)

    first_seen: dict[str, int] = {}
    for position, group in enumerate(groups_array):
        if group not in first_seen:
            first_seen[group] = position
    # Groups are ordered by first appearance, which is catalog (time) order.
    unique_groups = sorted(first_seen, key=first_seen.get)
    split_index = max(1, int(len(unique_groups) * (1 - TEST_FRACTION)))
    train_groups = set(unique_groups[:split_index])
    test_groups = set(unique_groups[split_index:])
    train_mask = np.array([group in train_groups for group in groups_array])
    test_mask = np.array([group in test_groups for group in groups_array])

    if train_mask.sum() < 50 or test_mask.sum() < 20:
        raise ValueError("train/test split produced too few samples")

    model = RandomForestRegressor(
        n_estimators=90,
        max_depth=8,
        min_samples_leaf=3,
        max_features=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x[train_mask], y_log[train_mask])

    ml_test_log = model.predict(x[test_mask])
    ml_test_count = np.expm1(np.clip(ml_test_log, 0.0, 20.0))
    baseline_test_log = x[test_mask, 18]
    baseline_test_count = np.expm1(np.clip(baseline_test_log, 0.0, 20.0))

    validation = {
        "split": "time-ordered 80/20 holdout by mainshock",
        "target": "count of M>=3 events in the window, per event-window sample",
        "samples": {
            "total": int(len(targets)),
            "train": int(train_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "mainshocks": {
            "total": info["mainshocks"],
            "train": len(train_groups),
            "test": len(test_groups),
        },
        "ml": metrics(y[test_mask], ml_test_count),
        "baseline": metrics(y[test_mask], baseline_test_count),
        "note": (
            "In-sample prototype comparison on one USGS monthly catalog. This is not "
            "out-of-sample validation across regions, not an accuracy claim, and not "
            "suitable for operational aftershock forecasting."
        ),
    }

    tree_export = [export_tree(estimator) for estimator in model.estimators_]
    node_count = sum(len(tree["l"]) for tree in tree_export)

    model_json = {
        "name": "Earthquake Aftershock Forecasting local aftershock estimator",
        "version": "1.0",
        "trainedAt": dt.date.today().isoformat(),
        "algorithm": f"RandomForestRegressor ({len(tree_export)} trees, max depth 8)",
        "predictionTarget": "log1p(M>=3 count in forecast window)",
        "features": [
            "magnitude",
            "depthKm",
            "startHours",
            "windowHours",
            "radiusKm",
            "log10MagnitudeFactor",
            "depthFactor",
            "omoriFraction",
            "log1pEarlyM3",
            "log1pEarlyAll",
            "earlyM3PerHour",
            "earlyMaxMagRelative",
            "productivityScale",
            "faultLengthScaled",
            "trendCos",
            "trendSin",
            "trendStrength",
            "log1pBaselineExpected",
            "log1pBaselinePrediction",
        ],
        "trees": tree_export,
        "validation": validation,
        "catalog": {
            "source": FEED_URL,
            "rows": len(rows),
            "latestTime": dt.datetime.fromtimestamp(info["latest"], dt.timezone.utc).isoformat(),
            "minMainshockMagnitude": MIN_MAIN_MAG,
            "startHours": list(START_HOURS),
            "windowHours": list(WINDOW_HOURS),
        },
        "nodeCount": node_count,
    }

    ASSETS.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model_json, separators=(",", ":")), encoding="utf-8")
    print(json.dumps(validation, indent=2))
    print(f"wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size:,} bytes)")
    print(f"wrote {TRAINING_CSV_PATH} ({TRAINING_CSV_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
