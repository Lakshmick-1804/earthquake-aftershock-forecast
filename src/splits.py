"""Stage 2A: Create event-level chronological and grouped CV splits.

Reads data_v2/raw/events.csv and data_v2/processed/*.csv.
Saves split manifests and class-prevalence statistics to data_v2/reports/.
"""
import csv
import json
from pathlib import Path

from config_v2 import V2_PROCESSED_DIR, V2_RAW_DIR, V2_REPORTS_DIR


def load_event_stats() -> list[dict]:
    with (V2_RAW_DIR / "events.csv").open("r", newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))

    enriched = []
    for ev in events:
        eid = ev["id"]
        p = V2_PROCESSED_DIR / f"{eid}.csv"
        with p.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        n_cells = len(rows)
        n_pos = sum(1 for r in rows if int(r["label"]) == 1)
        n_after = sum(int(r["aftershock_count"]) for r in rows)
        enriched.append({
            "id": eid,
            "time": ev["time"],
            "mag": float(ev["mag"]),
            "place": ev["place"],
            "n_cells": n_cells,
            "n_pos_cells": n_pos,
            "n_aftershocks_in_grid": n_after,
            "pos_rate": n_pos / n_cells if n_cells else 0.0,
        })
    # Sort chronologically by mainshock origin time
    enriched.sort(key=lambda x: x["time"])
    return enriched


def summarize_group(items: list[dict]) -> dict:
    n_events = len(items)
    n_cells = sum(x["n_cells"] for x in items)
    n_pos = sum(x["n_pos_cells"] for x in items)
    n_after = sum(x["n_aftershocks_in_grid"] for x in items)
    return {
        "n_events": n_events,
        "start_time": items[0]["time"] if items else None,
        "end_time": items[-1]["time"] if items else None,
        "n_cells": n_cells,
        "n_pos_cells": n_pos,
        "n_aftershocks_in_grid": n_after,
        "pos_prevalence": n_pos / n_cells if n_cells else 0.0,
        "event_ids": [x["id"] for x in items],
        "events": [{"id": x["id"], "time": x["time"], "mag": x["mag"], "n_pos_cells": x["n_pos_cells"]} for x in items],
    }


def main() -> None:
    V2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    events = load_event_stats()

    # Primary chronological split: earliest 89 events (2000-2016) vs latest 29 events (2017-2024)
    n_train = 89
    chrono_train = events[:n_train]
    chrono_test = events[n_train:]

    # Secondary diagnostic: 5-fold event-grouped CV (interleaved chronologically for balanced folds)
    folds = [[] for _ in range(5)]
    for idx, ev in enumerate(events):
        folds[idx % 5].append(ev)

    manifest = {
        "total_events": len(events),
        "chronological_split": {
            "description": "Earliest 89 events (train) vs latest 29 events (test)",
            "train": summarize_group(chrono_train),
            "test": summarize_group(chrono_test),
        },
        "group_cv_5fold": [
            {
                "fold": k,
                "train": summarize_group([e for j, f in enumerate(folds) if j != k for e in f]),
                "test": summarize_group(folds[k]),
            }
            for k in range(5)
        ],
    }

    (V2_REPORTS_DIR / "splits_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    tr = manifest["chronological_split"]["train"]
    te = manifest["chronological_split"]["test"]
    lines = [
        "# Stage 2A Split Design & Class Prevalence",
        "",
        "## Primary Chronological Split (89 Train / 29 Test)",
        "",
        "| Split | Events | Date Range | Total Cells | Positive Cells | Prevalence | In-Grid Aftershocks |",
        "|---|---:|---|---:|---:|---:|---:|",
        f"| Train | {tr['n_events']} | {tr['start_time'][:10]} to {tr['end_time'][:10]} | {tr['n_cells']:,} | {tr['n_pos_cells']:,} | {tr['pos_prevalence']:.4%} | {tr['n_aftershocks_in_grid']:,} |",
        f"| Test | {te['n_events']} | {te['start_time'][:10]} to {te['end_time'][:10]} | {te['n_cells']:,} | {te['n_pos_cells']:,} | {te['pos_prevalence']:.4%} | {te['n_aftershocks_in_grid']:,} |",
        "",
        "## Secondary 5-Fold Event-Grouped CV",
        "",
        "| Fold | Train Events | Test Events | Test Cells | Test Positive Cells | Test Prevalence |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for f in manifest["group_cv_5fold"]:
        ft = f["test"]
        lines.append(
            f"| {f['fold'] + 1} | {f['train']['n_events']} | {ft['n_events']} | {ft['n_cells']:,} | {ft['n_pos_cells']:,} | {ft['pos_prevalence']:.4%} |"
        )
    (V2_REPORTS_DIR / "splits_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Splits saved: Chrono Train={tr['n_events']} ({tr['pos_prevalence']:.4%}), Test={te['n_events']} ({te['pos_prevalence']:.4%})")


if __name__ == "__main__":
    main()
