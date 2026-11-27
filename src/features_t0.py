"""Stage 1B: Generate strictly T0-only prediction features and separate 30-day evaluation labels.

Reads ONLY from data_v2/raw/ (never modifies data/).
Writes clean feature+label CSVs to data_v2/processed/<event_id>.csv
and per-event grid/label diagnostics to data_v2/interim/event_summaries.json.
"""
import csv
import json
import math
import statistics
from datetime import datetime, timezone

from config_v2 import (
    CELLS_PER_EVENT,
    FORECAST_DAYS,
    GRID_N,
    R_SOFTENING,
    V2_INTERIM_DIR,
    V2_PROCESSED_DIR,
    V2_RAW_DIR,
    V2_REPORTS_DIR,
    extent_km,
    fault_length_km,
    to_local_km,
)
from prepare_data import is_mainshock_row, parse_iso_utc

T0_FEATURES = [
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

OUTPUT_COLUMNS = [
    "event",
    "cell_id",
    "row_idx",
    "col_idx",
    *T0_FEATURES,
    "aftershock_count",
    "label",
]


def build_event_table(ev: dict) -> tuple[list[dict], dict]:
    eid = ev["id"]
    t0 = ev["t0"]
    lat0 = ev["lat0"]
    lon0 = ev["lon0"]
    mag0 = ev["mag0"]

    L = fault_length_km(mag0)
    E = extent_km(L)
    half_L = L / 2.0
    step = (2.0 * E) / GRID_N
    cell_area = step * step
    centers = [-E + (i + 0.5) * step for i in range(GRID_N)]

    # Read future 30-day catalog STRICTLY for ground-truth labels
    raw_path = V2_RAW_DIR / f"{eid}.csv"
    with raw_path.open("r", newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    counts_grid = [[0 for _ in range(GRID_N)] for _ in range(GRID_N)]
    n_excluded_mainshock = 0
    n_excluded_window = 0
    n_outside_grid = 0
    n_inside_grid = 0

    for r in raw_rows:
        if is_mainshock_row(r, t0, lat0, lon0, mag0):
            n_excluded_mainshock += 1
            continue
        dt_days = (parse_iso_utc(r["time"]) - t0).total_seconds() / 86400.0
        if not (0.0 < dt_days <= float(FORECAST_DAYS)):
            n_excluded_window += 1
            continue

        ax, ay = to_local_km(float(r["latitude"]), float(r["longitude"]), lat0, lon0)
        # Match numpy.histogram2d(ay, ax, bins=[edges, edges]) binning: [-E, E]
        if ax < -E or ax > E or ay < -E or ay > E:
            n_outside_grid += 1
            continue
        col_idx = min(GRID_N - 1, max(0, int((ax + E) / step)))
        row_idx = min(GRID_N - 1, max(0, int((ay + E) / step)))
        counts_grid[row_idx][col_idx] += 1
        n_inside_grid += 1

    # Build T0-only feature rows (zero dependency on raw_rows)
    out_rows = []
    pos_cells = 0
    for row_idx in range(GRID_N):
        y = centers[row_idx]
        for col_idx in range(GRID_N):
            x = centers[col_idx]
            cell_id = row_idx * GRID_N + col_idx
            r_km = math.hypot(x, y)
            r_norm = r_km / half_L
            r2 = r_norm * r_norm + R_SOFTENING
            log_r_norm = 0.5 * math.log(r2)
            inv_r2 = 1.0 / r2
            inv_r3 = 1.0 / (r2 ** 1.5)

            cnt = counts_grid[row_idx][col_idx]
            lbl = 1 if cnt > 0 else 0
            pos_cells += lbl

            out_rows.append({
                "event": eid,
                "cell_id": cell_id,
                "row_idx": row_idx,
                "col_idx": col_idx,
                "x": x,
                "y": y,
                "r_km": r_km,
                "r_norm": r_norm,
                "log_r_norm": log_r_norm,
                "inv_r2": inv_r2,
                "inv_r3": inv_r3,
                "mag": mag0,
                "length_km": L,
                "extent_km": E,
                "cell_area_km2": cell_area,
                "lat0": lat0,
                "lon0": lon0,
                "abs_lat": abs(lat0),
                "aftershock_count": cnt,
                "label": lbl,
            })

    summary = {
        "id": eid,
        "time": ev["time"],
        "mag": mag0,
        "length_km": L,
        "extent_km": E,
        "raw_total": len(raw_rows),
        "excluded_mainshock": n_excluded_mainshock,
        "excluded_window": n_excluded_window,
        "inside_grid_aftershocks": n_inside_grid,
        "outside_grid_aftershocks": n_outside_grid,
        "positive_cells": pos_cells,
        "total_cells": CELLS_PER_EVENT,
        "positive_rate": pos_cells / CELLS_PER_EVENT,
    }
    return out_rows, summary


def main() -> None:
    V2_INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    V2_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    V2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with (V2_RAW_DIR / "events.csv").open("r", newline="", encoding="utf-8") as f:
        raw_events = list(csv.DictReader(f))

    parsed_events = []
    for ev in raw_events:
        parsed_events.append({
            "id": ev["id"],
            "time": ev["time"],
            "t0": parse_iso_utc(ev["time"]),
            "lat0": float(ev["latitude"]),
            "lon0": float(ev["longitude"]),
            "mag0": float(ev["mag"]),
            "place": ev["place"],
        })

    summaries = []
    for ev in parsed_events:
        rows, summary = build_event_table(ev)
        out_path = V2_PROCESSED_DIR / f"{ev['id']}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        summaries.append(summary)

    (V2_INTERIM_DIR / "event_summaries.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    pos_cells_list = [s["positive_cells"] for s in summaries]
    total_pos = sum(pos_cells_list)
    total_cells = sum(s["total_cells"] for s in summaries)
    print(
        f"Stage 1B complete: wrote {len(summaries)} leak-free feature+label CSVs to {V2_PROCESSED_DIR}. "
        f"Total positive cells: {total_pos}/{total_cells} ({total_pos/total_cells:.4%}), "
        f"per-event positive cells min={min(pos_cells_list)}, median={statistics.median(pos_cells_list)}, max={max(pos_cells_list)}"
    )


if __name__ == "__main__":
    main()
