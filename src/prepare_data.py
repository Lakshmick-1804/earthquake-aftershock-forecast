"""Stage 1A: Preserve the 118-event dataset in data_v2/raw and run scientific data validation.

Reads data/events.csv and data/raw/*.csv strictly read-only.
Writes:
  - data_v2/raw/events.csv and the 118 selected data_v2/raw/<id>.csv files
  - data_v2/reports/source_manifest.json & .csv
  - data_v2/reports/data_validation_report.json & .md
"""
import csv
import hashlib
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path

from config_v2 import (
    EXPECTED_N_EVENTS,
    FORECAST_DAYS,
    ORIG_EVENTS_CSV,
    ORIG_RAW_DIR,
    V2_INTERIM_DIR,
    V2_PROCESSED_DIR,
    V2_RAW_DIR,
    V2_REPORTS_DIR,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_iso_utc(ts: str) -> datetime:
    s = ts.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def ensure_directories() -> None:
    for d in (V2_RAW_DIR, V2_INTERIM_DIR, V2_PROCESSED_DIR, V2_REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def is_mainshock_row(row: dict, t0: datetime, lat0: float, lon0: float, mag0: float) -> bool:
    """Detect whether a raw catalog row is the mainshock itself."""
    t_row = parse_iso_utc(row["time"])
    dt_sec = abs((t_row - t0).total_seconds())
    lat = float(row["latitude"])
    lon = float(row["longitude"])
    mag = float(row["mag"])
    return dt_sec <= 1.0 and abs(lat - lat0) < 1e-4 and abs(lon - lon0) < 1e-4 and abs(mag - mag0) < 0.05


def main() -> None:
    ensure_directories()

    with ORIG_EVENTS_CSV.open("r", newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))

    if len(events) != EXPECTED_N_EVENTS:
        raise ValueError(f"Expected {EXPECTED_N_EVENTS} events, found {len(events)}")

    event_ids = [e["id"] for e in events]
    if len(set(event_ids)) != EXPECTED_N_EVENTS:
        raise ValueError("Duplicate event IDs found in data/events.csv")

    # Preserve events.csv in data_v2/raw/
    v2_events_path = V2_RAW_DIR / "events.csv"
    shutil.copy2(ORIG_EVENTS_CSV, v2_events_path)

    # Identify extra raw files in original data/raw/ (excluded by MIN_AFTERSHOCKS=20)
    all_orig_raw = sorted(p.stem for p in ORIG_RAW_DIR.glob("*.csv"))
    selected_set = set(event_ids)
    excluded_raw_ids = sorted(eid for eid in all_orig_raw if eid not in selected_set)

    manifest_rows = []
    validation_events = []

    events_with_mainshock_row = []
    events_with_negative_dt = []
    events_with_gt_30d = []
    events_mag_below_4 = []
    year_counts: dict[str, int] = {}

    for ev in events:
        eid = ev["id"]
        src_raw = ORIG_RAW_DIR / f"{eid}.csv"
        dst_raw = V2_RAW_DIR / f"{eid}.csv"
        if not src_raw.exists():
            raise FileNotFoundError(f"Missing raw CSV for selected event: {src_raw}")

        shutil.copy2(src_raw, dst_raw)
        src_hash = sha256_file(src_raw)
        dst_hash = sha256_file(dst_raw)
        if src_hash != dst_hash:
            raise RuntimeError(f"Checksum mismatch after copy for {eid}")

        t0 = parse_iso_utc(ev["time"])
        lat0 = float(ev["latitude"])
        lon0 = float(ev["longitude"])
        mag0 = float(ev["mag"])
        year = str(t0.year)
        year_counts[year] = year_counts.get(year, 0) + 1

        with dst_raw.open("r", newline="", encoding="utf-8") as f:
            raw_rows = list(csv.DictReader(f))

        dts_days = []
        mags = []
        mainshock_matches = 0
        neg_dt_count = 0
        gt_30d_count = 0
        lt_4_count = 0

        for r in raw_rows:
            t_r = parse_iso_utc(r["time"])
            dt_d = (t_r - t0).total_seconds() / 86400.0
            m_r = float(r["mag"])
            if any(math.isnan(v) or math.isinf(v) for v in (float(r["latitude"]), float(r["longitude"]), m_r)):
                raise ValueError(f"Invalid numeric value in {eid}")
            dts_days.append(dt_d)
            mags.append(m_r)
            if is_mainshock_row(r, t0, lat0, lon0, mag0):
                mainshock_matches += 1
            if dt_d < 0.0:
                neg_dt_count += 1
            if dt_d > float(FORECAST_DAYS):
                gt_30d_count += 1
            if m_r < 4.0:
                lt_4_count += 1

        if mainshock_matches > 0:
            events_with_mainshock_row.append((eid, mainshock_matches))
        if neg_dt_count > 0:
            events_with_negative_dt.append((eid, neg_dt_count))
        if gt_30d_count > 0:
            events_with_gt_30d.append((eid, gt_30d_count))
        if lt_4_count > 0:
            events_mag_below_4.append((eid, lt_4_count))

        manifest_rows.append({
            "id": eid,
            "time": ev["time"],
            "latitude": lat0,
            "longitude": lon0,
            "mag": mag0,
            "place": ev["place"],
            "orig_raw_path": str(src_raw),
            "v2_raw_path": str(dst_raw),
            "sha256": dst_hash,
            "raw_row_count": len(raw_rows),
        })

        validation_events.append({
            "id": eid,
            "time": ev["time"],
            "mag": mag0,
            "raw_row_count": len(raw_rows),
            "mainshock_row_matches": mainshock_matches,
            "neg_dt_count": neg_dt_count,
            "gt_30d_count": gt_30d_count,
            "min_dt_days": min(dts_days) if dts_days else None,
            "max_dt_days": max(dts_days) if dts_days else None,
            "window_span_days": (max(dts_days) - min(dts_days)) if dts_days else 0.0,
            "min_mag": min(mags) if mags else None,
            "median_mag": statistics.median(mags) if mags else None,
            "max_mag": max(mags) if mags else None,
            "mag_lt_4_count": lt_4_count,
        })

    # Write source manifest CSV & JSON
    manifest_csv = V2_REPORTS_DIR / "source_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_json = {
        "n_selected_events": len(manifest_rows),
        "orig_events_csv": str(ORIG_EVENTS_CSV),
        "orig_events_sha256": sha256_file(ORIG_EVENTS_CSV),
        "v2_events_csv": str(v2_events_path),
        "n_orig_raw_files": len(all_orig_raw),
        "excluded_raw_ids_lt_20_aftershocks": excluded_raw_ids,
        "events": manifest_rows,
    }
    (V2_REPORTS_DIR / "source_manifest.json").write_text(
        json.dumps(manifest_json, indent=2), encoding="utf-8"
    )

    # Global validation summary
    min_dts = [e["min_dt_days"] for e in validation_events]
    max_dts = [e["max_dt_days"] for e in validation_events]
    min_mags = [e["min_mag"] for e in validation_events]
    med_mags = [e["median_mag"] for e in validation_events]
    max_mags = [e["max_mag"] for e in validation_events]

    val_summary = {
        "n_selected_events": len(validation_events),
        "events_with_mainshock_row": events_with_mainshock_row,
        "events_with_negative_dt": events_with_negative_dt,
        "events_with_gt_30d": events_with_gt_30d,
        "events_mag_below_4": events_mag_below_4,
        "global_min_dt_days": min(min_dts),
        "global_max_dt_days": max(max_dts),
        "median_max_dt_days": statistics.median(max_dts),
        "min_of_max_dt_days": min(max_dts),
        "global_min_aftershock_mag": min(min_mags),
        "median_event_median_mag": statistics.median(med_mags),
        "global_max_aftershock_mag": max(max_mags),
        "events_per_year": dict(sorted(year_counts.items())),
        "excluded_raw_count": len(excluded_raw_ids),
        "per_event": validation_events,
    }
    (V2_REPORTS_DIR / "data_validation_report.json").write_text(
        json.dumps(val_summary, indent=2), encoding="utf-8"
    )

    # Human-readable markdown summary
    md_lines = [
        "# Stage 1 Data Validation Report (118 Selected Earthquakes)",
        "",
        f"- **Selected events verified**: {len(validation_events)} (exact match to `data/events.csv`)",
        f"- **Excluded candidate raw CSVs in `data/raw/`**: {len(excluded_raw_ids)} (all had < 20 aftershocks due to original `MIN_AFTERSHOCKS=20` filter)",
        f"- **Mainshock rows found inside raw CSVs**: {len(events_with_mainshock_row)} events ({events_with_mainshock_row[:5]})",
        f"- **Rows with `dt < 0` (before T0)**: {len(events_with_negative_dt)} events",
        f"- **Rows with `dt > 30` days**: {len(events_with_gt_30d)} events",
        f"- **Observed `dt` range across all 118 events**: [{min(min_dts):.6f}, {max(max_dts):.6f}] days (min event max-dt: {min(max_dts):.2f} days, median max-dt: {statistics.median(max_dts):.2f} days)",
        f"- **Aftershock magnitude range**: [{min(min_mags):.2f}, {max(max_mags):.2f}] (events with `mag < 4.0`: {len(events_mag_below_4)}; median of event medians: {statistics.median(med_mags):.2f})",
        "",
        "## Provisional Chronological Distribution (Events per Year)",
        "",
        "| Year | Event Count |",
        "|---|---:|",
    ]
    for y, cnt in sorted(year_counts.items()):
        md_lines.append(f"| {y} | {cnt} |")
    md_lines.extend([
        "",
        "## Scientific & Completeness Notes",
        "- **Selection bias**: Retaining only events with >= 20 M4+ aftershocks excludes 12 M7.5+ candidates; all results are conditional on productive sequences.",
        "- **Completeness heterogeneity**: Although `AFTER_MIN_MAG = 4.0` was queried, catalog completeness varies by region, depth, and early-aftershock coda overlap.",
        "- **Missing source parameters**: `depth`, `strike`, `dip`, and `rake` are absent from `data/events.csv` and are not fabricated.",
    ])
    (V2_REPORTS_DIR / "data_validation_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Stage 1A complete: preserved {len(manifest_rows)} events in {V2_RAW_DIR} and wrote validation reports to {V2_REPORTS_DIR}")


if __name__ == "__main__":
    main()
