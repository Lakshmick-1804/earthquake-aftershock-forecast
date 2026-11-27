"""Stage 2D: Unified evaluation of all three models on identical held-out events.

Evaluates:
  1. Primary Chronological Split (89 train / 29 test)
  2. Secondary 5-Fold Event-Grouped CV (118 events)

Reports cell-level pooled metrics (with 95% event-cluster bootstrap CIs),
event-level metrics (PR-AUC, ROC-AUC, Brier, Log Loss, Spatial Hit Rate @ 5/10/20%),
and 10-bin calibration reliability.
"""
import csv
import json
import statistics
from pathlib import Path

from baselines import (
    fit_etas_spatial_benchmark,
    fit_radial_baseline,
    predict_etas_spatial_benchmark,
    predict_radial_baseline,
)
from config_v2 import V2_PROCESSED_DIR, V2_REPORTS_DIR
from math_utils import (
    average_precision_score,
    bootstrap_ci,
    brier_score,
    log_loss,
    roc_auc_score,
    spatial_hit_rate,
)
from train_ml import fit_calibrated_ml, predict_calibrated_ml


def load_events_rows(event_ids: list[str]) -> list[dict]:
    rows = []
    for eid in event_ids:
        with (V2_PROCESSED_DIR / f"{eid}.csv").open("r", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def evaluate_predictions(rows: list[dict], preds_by_model: dict[str, list[float]], include_ci: bool = True) -> dict:
    by_event: dict[str, dict] = {}
    for idx, r in enumerate(rows):
        eid = r["event"]
        if eid not in by_event:
            by_event[eid] = {"y": [], "counts": [], "preds": {m: [] for m in preds_by_model}}
        by_event[eid]["y"].append(int(r["label"]))
        by_event[eid]["counts"].append(int(r["aftershock_count"]))
        for m, plist in preds_by_model.items():
            by_event[eid]["preds"][m].append(plist[idx])

    event_ids = list(by_event.keys())
    y_pooled = [int(r["label"]) for r in rows]

    results = {"n_events": len(event_ids), "n_cells": len(rows), "prevalence": sum(y_pooled) / len(y_pooled), "models": {}}

    for m, p_pooled in preds_by_model.items():
        cell_metrics = {
            "pr_auc": average_precision_score(y_pooled, p_pooled),
            "roc_auc": roc_auc_score(y_pooled, p_pooled),
            "brier": brier_score(y_pooled, p_pooled),
            "log_loss": log_loss(y_pooled, p_pooled),
        }
        if include_ci:
            ev_map = {eid: {"y": by_event[eid]["y"], "p": by_event[eid]["preds"][m]} for eid in event_ids}
            cell_metrics["pr_auc_ci95"] = list(bootstrap_ci(event_ids, ev_map, average_precision_score, n_boot=100))
            cell_metrics["roc_auc_ci95"] = list(bootstrap_ci(event_ids, ev_map, roc_auc_score, n_boot=100))
            cell_metrics["brier_ci95"] = list(bootstrap_ci(event_ids, ev_map, brier_score, n_boot=100))
            cell_metrics["log_loss_ci95"] = list(bootstrap_ci(event_ids, ev_map, log_loss, n_boot=100))

        ev_pr, ev_roc, ev_brier, ev_ll = [], [], [], []
        hit5, hit10, hit20 = [], [], []
        for eid in event_ids:
            ye = by_event[eid]["y"]
            ce = by_event[eid]["counts"]
            pe = by_event[eid]["preds"][m]
            ev_pr.append(average_precision_score(ye, pe))
            ev_roc.append(roc_auc_score(ye, pe))
            ev_brier.append(brier_score(ye, pe))
            ev_ll.append(log_loss(ye, pe))
            hit5.append(spatial_hit_rate(ce, pe, 0.05))
            hit10.append(spatial_hit_rate(ce, pe, 0.10))
            hit20.append(spatial_hit_rate(ce, pe, 0.20))

        # 10-bin reliability table (equal-width in [0, 1] plus summary)
        bins = []
        for b in range(10):
            lo, hi = b * 0.1, (b + 1) * 0.1
            idxs = [i for i, pv in enumerate(p_pooled) if (lo <= pv < hi) or (b == 9 and pv == 1.0)]
            if idxs:
                bins.append({
                    "bin": f"[{lo:.1f}, {hi:.1f}]",
                    "count": len(idxs),
                    "mean_pred": sum(p_pooled[i] for i in idxs) / len(idxs),
                    "obs_rate": sum(y_pooled[i] for i in idxs) / len(idxs),
                })

        results["models"][m] = {
            "cell_pooled": cell_metrics,
            "event_level_mean": {
                "pr_auc": statistics.mean(ev_pr),
                "roc_auc": statistics.mean(ev_roc),
                "brier": statistics.mean(ev_brier),
                "log_loss": statistics.mean(ev_ll),
                "hit_rate_5pct": statistics.mean(hit5),
                "hit_rate_10pct": statistics.mean(hit10),
                "hit_rate_20pct": statistics.mean(hit20),
            },
            "event_level_per_event": {
                eid: {"pr_auc": ev_pr[i], "brier": ev_brier[i], "hit_rate_10pct": hit10[i]}
                for i, eid in enumerate(event_ids)
            },
            "calibration_bins": bins,
        }

    # Event-level win rates of ML vs baselines
    ml_pe = results["models"]["ml_calibrated"]["event_level_per_event"]
    win_rates = {}
    for base_name in ("radial_baseline", "etas_spatial_benchmark"):
        b_pe = results["models"][base_name]["event_level_per_event"]
        win_rates[f"ml_beats_{base_name}_pr_auc"] = sum(
            1 for eid in event_ids if ml_pe[eid]["pr_auc"] > b_pe[eid]["pr_auc"]
        ) / len(event_ids)
        win_rates[f"ml_beats_{base_name}_brier"] = sum(
            1 for eid in event_ids if ml_pe[eid]["brier"] < b_pe[eid]["brier"]
        ) / len(event_ids)
    results["event_win_rates"] = win_rates
    return results


def main() -> None:
    manifest = json.loads((V2_REPORTS_DIR / "splits_manifest.json").read_text(encoding="utf-8"))

    # 1. Primary Chronological Split (89 train / 29 test)
    tr_ids = manifest["chronological_split"]["train"]["event_ids"]
    te_ids = manifest["chronological_split"]["test"]["event_ids"]
    tr_rows = load_events_rows(tr_ids)
    te_rows = load_events_rows(te_ids)

    m_radial = fit_radial_baseline(tr_rows)
    m_etas = fit_etas_spatial_benchmark(tr_rows)
    m_ml = fit_calibrated_ml(tr_rows, l2=10.0, n_inner_folds=3)

    chrono_preds = {
        "radial_baseline": predict_radial_baseline(m_radial, te_rows),
        "etas_spatial_benchmark": predict_etas_spatial_benchmark(m_etas, te_rows),
        "ml_calibrated": predict_calibrated_ml(m_ml, te_rows),
    }
    chrono_eval = evaluate_predictions(te_rows, chrono_preds, include_ci=True)

    # Save chronological test predictions
    pred_csv = V2_REPORTS_DIR / "predictions_chrono.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event", "cell_id", "x", "y", "label", "aftershock_count", "p_radial", "p_etas", "p_ml"])
        for i, r in enumerate(te_rows):
            writer.writerow([
                r["event"], r["cell_id"], r["x"], r["y"], r["label"], r["aftershock_count"],
                f"{chrono_preds['radial_baseline'][i]:.6f}",
                f"{chrono_preds['etas_spatial_benchmark'][i]:.6f}",
                f"{chrono_preds['ml_calibrated'][i]:.6f}",
            ])

    # 2. Secondary 5-Fold Event-Grouped CV (all 118 events)
    cv_rows_all = []
    cv_preds_all = {"radial_baseline": [], "etas_spatial_benchmark": [], "ml_calibrated": []}
    for fold in manifest["group_cv_5fold"]:
        f_tr = load_events_rows(fold["train"]["event_ids"])
        f_te = load_events_rows(fold["test"]["event_ids"])
        fr = fit_radial_baseline(f_tr)
        fe = fit_etas_spatial_benchmark(f_tr)
        fm = fit_calibrated_ml(f_tr, l2=10.0, n_inner_folds=3)
        cv_rows_all.extend(f_te)
        cv_preds_all["radial_baseline"].extend(predict_radial_baseline(fr, f_te))
        cv_preds_all["etas_spatial_benchmark"].extend(predict_etas_spatial_benchmark(fe, f_te))
        cv_preds_all["ml_calibrated"].extend(predict_calibrated_ml(fm, f_te))

    cv_eval = evaluate_predictions(cv_rows_all, cv_preds_all, include_ci=False)

    report = {
        "chronological_holdout": {
            "fitted_params": {
                "etas_spatial_benchmark": {"b0": m_etas["b0"], "alpha": m_etas["alpha"], "d2": m_etas["d2"], "q": m_etas["q"]},
                "ml_platt_calibrator": {"a": m_ml["platt_a"], "b": m_ml["platt_b"], "l2": m_ml["l2"]},
            },
            "evaluation": chrono_eval,
        },
        "group_cv_5fold": cv_eval,
    }
    (V2_REPORTS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write Markdown summary
    lines = [
        "# Stage 2 Evaluation Report",
        "",
        "## 1. Chronological Holdout (89 Train [2000-2016] / 29 Test [2017-2024])",
        f"- Test cells: {chrono_eval['n_cells']:,} across {chrono_eval['n_events']} held-out events (positive prevalence: {chrono_eval['prevalence']:.4%})",
        "",
        "### Cell-Level Pooled Metrics (with 95% Event-Cluster Bootstrap CIs)",
        "",
        "| Model | PR-AUC (95% CI) | ROC-AUC (95% CI) | Brier Score (95% CI) | Log Loss (95% CI) |",
        "|---|---|---|---|---|",
    ]
    for name in ("radial_baseline", "etas_spatial_benchmark", "ml_calibrated"):
        cm = chrono_eval["models"][name]["cell_pooled"]
        lines.append(
            f"| `{name}` | {cm['pr_auc']:.4f} [{cm['pr_auc_ci95'][0]:.4f}, {cm['pr_auc_ci95'][1]:.4f}] | "
            f"{cm['roc_auc']:.4f} [{cm['roc_auc_ci95'][0]:.4f}, {cm['roc_auc_ci95'][1]:.4f}] | "
            f"{cm['brier']:.5f} [{cm['brier_ci95'][0]:.5f}, {cm['brier_ci95'][1]:.5f}] | "
            f"{cm['log_loss']:.4f} [{cm['log_loss_ci95'][0]:.4f}, {cm['log_loss_ci95'][1]:.4f}] |"
        )

    lines.extend([
        "",
        "### Event-Level Mean Metrics (across 29 Test Events)",
        "",
        "| Model | Event PR-AUC | Event ROC-AUC | Event Brier | Hit@5% | Hit@10% | Hit@20% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("radial_baseline", "etas_spatial_benchmark", "ml_calibrated"):
        em = chrono_eval["models"][name]["event_level_mean"]
        lines.append(
            f"| `{name}` | {em['pr_auc']:.4f} | {em['roc_auc']:.4f} | {em['brier']:.5f} | "
            f"{em['hit_rate_5pct']:.2%} | {em['hit_rate_10pct']:.2%} | {em['hit_rate_20pct']:.2%} |"
        )

    lines.extend([
        "",
        "## 2. Secondary 5-Fold Event-Grouped CV (All 118 Events)",
        "",
        "| Model | Pooled PR-AUC | Pooled ROC-AUC | Pooled Brier | Pooled Log Loss | Event Hit@10% |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in ("radial_baseline", "etas_spatial_benchmark", "ml_calibrated"):
        cm = cv_eval["models"][name]["cell_pooled"]
        em = cv_eval["models"][name]["event_level_mean"]
        lines.append(
            f"| `{name}` | {cm['pr_auc']:.4f} | {cm['roc_auc']:.4f} | {cm['brier']:.5f} | {cm['log_loss']:.4f} | {em['hit_rate_10pct']:.2%} |"
        )

    (V2_REPORTS_DIR / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Stage 2 evaluation complete. Reports written to", V2_REPORTS_DIR)


if __name__ == "__main__":
    main()
