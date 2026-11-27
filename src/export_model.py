"""Export the fitted Stage 2 chronological ML model to data_v2/models/ and verify reconstruction.

Does NOT modify any existing reports in data_v2/reports/ or original prototype files.
"""
import csv
import json
from pathlib import Path

from config_v2 import V2_DATA_DIR, V2_REPORTS_DIR
from evaluate import load_events_rows
from train_ml import fit_calibrated_ml, load_calibrated_ml, predict_calibrated_ml, save_calibrated_ml

MODEL_ARTIFACT_PATH = V2_DATA_DIR / "models" / "ml_calibrated_chrono.json"


def main() -> None:
    manifest = json.loads((V2_REPORTS_DIR / "splits_manifest.json").read_text(encoding="utf-8"))
    tr_ids = manifest["chronological_split"]["train"]["event_ids"]
    te_ids = manifest["chronological_split"]["test"]["event_ids"]

    tr_rows = load_events_rows(tr_ids)
    te_rows = load_events_rows(te_ids)

    # Fit with exact Stage 2 settings
    model = fit_calibrated_ml(tr_rows, l2=10.0, n_inner_folds=3)
    save_calibrated_ml(
        model,
        MODEL_ARTIFACT_PATH,
        metadata={
            "split": "chronological_holdout",
            "n_train_events": len(tr_ids),
            "n_test_events": len(te_ids),
            "n_inner_folds": 3,
        },
    )

    # Reconstruct from JSON and verify exact numerical match
    reconstructed = load_calibrated_ml(MODEL_ARTIFACT_PATH)
    preds_mem = predict_calibrated_ml(model, te_rows)
    preds_loaded = predict_calibrated_ml(reconstructed, te_rows)
    max_diff_exact = max(abs(a - b) for a, b in zip(preds_mem, preds_loaded))

    # Also compare against saved predictions_chrono.csv (stored at 6 decimal places)
    with (V2_REPORTS_DIR / "predictions_chrono.csv").open("r", newline="", encoding="utf-8") as f:
        saved_p_ml = [float(r["p_ml"]) for r in csv.DictReader(f)]
    max_diff_csv = max(abs(a - b) for a, b in zip(preds_loaded, saved_p_ml))

    print(f"Saved model artifact: {MODEL_ARTIFACT_PATH}")
    print(f"Weights count: {len(reconstructed['weights'])} (1 bias + {len(reconstructed['weights']) - 1} coefficients)")
    print(f"Platt params: a={reconstructed['platt_a']:.10f}, b={reconstructed['platt_b']:.10f}")
    print(f"Max abs diff (reconstructed vs in-memory): {max_diff_exact:.3e} (tolerance 1e-8)")
    print(f"Max abs diff (reconstructed vs 6-dp predictions_chrono.csv): {max_diff_csv:.3e}")
    assert max_diff_exact <= 1e-8, f"Reconstruction exceeded 1e-8 tolerance: {max_diff_exact}"
    assert max_diff_csv <= 1e-6, f"CSV mismatch exceeded 6-dp rounding: {max_diff_csv}"


if __name__ == "__main__":
    main()
