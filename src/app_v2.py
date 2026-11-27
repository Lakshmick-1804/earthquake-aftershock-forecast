"""Stage 3: Interactive Earthquake Aftershock Forecasting Dashboard (v2).

Run with:
    streamlit run src/app_v2.py

Preserves the original app.py and data/ untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from dashboard_service import (
    get_cell,
    get_event_grid_predictions,
    load_catalog_and_splits,
    load_models,
)

st.set_page_config(
    page_title="Earthquake Aftershock Forecasting (v2)",
    page_icon="🌍",
    layout="wide",
)

MODEL_LABELS = {
    "radial_baseline": "Isotropic Radial Baseline",
    "etas_spatial_benchmark": "ETAS-Inspired Spatial Benchmark",
    "ml_calibrated": "Calibrated ML (L2 Logistic + Platt)",
}
MODEL_PROB_COL = {
    "radial_baseline": "p_radial",
    "etas_spatial_benchmark": "p_etas",
    "ml_calibrated": "p_ml",
}
MODEL_RANK_COL = {
    "radial_baseline": "rank_radial",
    "etas_spatial_benchmark": "rank_etas",
    "ml_calibrated": "rank_ml",
}


@st.cache_resource
def _cached_bundle():
    bundle = load_catalog_and_splits()
    models = load_models(bundle["splits"], bundle["eval_report"])
    return bundle, models


@st.cache_data
def _cached_event_grid(event_id: str):
    _, models = _cached_bundle()
    return get_event_grid_predictions(event_id, models)


def _plot_probability_heatmap(grid_rows: list[dict], prob_col: str, extent_km: float, title: str):
    n = 40
    z = [[0.0] * n for _ in range(n)]
    for r in grid_rows:
        z[r["row_idx"]][r["col_idx"]] = r[prob_col]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(z, extent=[-extent_km, extent_km, -extent_km, extent_km], origin="lower", cmap="inferno", vmin=0.0)
    ax.plot(0, 0, marker="*", color="cyan", markersize=10, label="Mainshock epicenter")
    ax.set_title(title)
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.legend(loc="upper right", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("Predicted P(aftershock >= 1)")
    fig.tight_layout()
    return fig


def _plot_observed_grid(grid_rows: list[dict], extent_km: float):
    n = 40
    z = [[0] * n for _ in range(n)]
    for r in grid_rows:
        z[r["row_idx"]][r["col_idx"]] = r["label"]
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    cmap = matplotlib.colors.ListedColormap(["#0B1F3A", "#FFD60A"])
    ax.imshow(z, extent=[-extent_km, extent_km, -extent_km, extent_km], origin="lower", cmap=cmap, vmin=0, vmax=1)
    ax.plot(0, 0, marker="*", color="cyan", markersize=10, label="Mainshock epicenter")
    ax.set_title("Observed 30-Day Aftershock Cells (Ground Truth)")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def main():
    st.title("Earthquake Aftershock Forecasting — Leak-Free Research Prototype (v2)")
    st.caption(
        "College Research Project | 118 selected earthquakes from M7.0+ USGS candidates (2000–2024; realized selected magnitudes M7.5–M9.1) | "
        "Strictly T0-Available Features | Not for operational emergency decision-making"
    )

    try:
        bundle, _ = _cached_bundle()
    except Exception as exc:
        st.error(f"Failed to load Stage 2 data or model artifacts: {exc}")
        st.stop()

    catalog = bundle["catalog"]
    splits = bundle["splits"]
    eval_report = bundle["eval_report"]

    tabs = st.tabs([
        "1. Overview",
        "2. Explorer & Forecast Map",
        "3. Model Comparison",
        "4. Prediction Analysis",
        "5. Methodology & Limitations",
    ])

    # 1. OVERVIEW
    with tabs[0]:
        st.subheader("Project Purpose & Dataset Overview")
        st.write(
            "This dashboard forecasts the spatial probability of M4.0+ aftershocks occurring within 30 days "
            "after a major earthquake (118 selected earthquakes from M7.0+ USGS candidates, 2000–2024; realized selected magnitudes M7.5–M9.1) "
            "on a 40 x 40 local spatial grid centered on the mainshock epicenter. "
            "All prediction-time inputs use only information available at the mainshock origin time (T0)."
        )
        tr = splits["chronological_split"]["train"]
        te = splits["chronological_split"]["test"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Earthquakes", splits["total_events"])
        c2.metric("Chronological Train (2000–2016)", tr["n_events"])
        c3.metric("Chronological Test (2016–2024)", te["n_events"])
        c4.metric("Test Positive Cell Prevalence", f"{te['pos_prevalence']:.2%}")

        st.markdown("### Three Forecasting Approaches")
        st.markdown(
            "- **Isotropic Radial Baseline**: L2-regularized logistic regression using normalized radial distance "
            "`log_r_norm` and mainshock magnitude `mag`.\n"
            "- **ETAS-Inspired Spatial Benchmark**: Parametric Utsu productivity and power-law spatial decay kernel "
            "converted to cell occurrence probability via $P(Y \\ge 1) = 1 - \\exp(-\\lambda)$. "
            "*(Not a complete space-time ETAS model.)*\n"
            "- **Calibrated ML Model**: L2-regularized logistic regression over 20 strictly T0-available spatial/magnitude "
            "features with event-aware 3-fold out-of-fold Platt scaling."
        )

    # 2. EARTHQUAKE EXPLORER & FORECAST MAP
    with tabs[1]:
        ordered_ids = sorted(catalog.keys(), key=lambda k: (-catalog[k]["mag"], catalog[k]["time"]))
        selected_id = st.selectbox(
            "Select Earthquake (118 events)",
            ordered_ids,
            format_func=lambda k: (
                f"[{catalog[k]['split_role'].upper()}] M{catalog[k]['mag']:.1f} — {catalog[k]['place']} "
                f"({catalog[k]['time'][:10]}) [{k}]"
            ),
        )
        ev = catalog[selected_id]

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Event ID", ev["id"])
        m2.metric("Magnitude", f"M{ev['mag']:.2f}")
        m3.metric("Split Role", ev["split_role"].upper())
        m4.metric("In-Grid Aftershocks", ev["inside_grid_aftershocks"])
        m5.metric("Positive Cells", f"{ev['positive_cells']} / 1,600")

        st.write(
            f"**Date (UTC):** `{ev['time']}` | **Location:** `{ev['place']}` "
            f"(`lat={ev['latitude']:.4f}`, `lon={ev['longitude']:.4f}`) | "
            f"**Est. Fault Length:** `{ev['length_km']:.1f} km` | **Grid Half-Width (E):** `±{ev['extent_km']:.1f} km`"
        )
        st.info(
            "Missing source parameters: `depth`, `strike`, `dip`, and `rake` are not present in `events.csv` "
            "and are not fabricated. Forecasts cover only the 40 x 40 grid [-E, E] x [-E, E]."
        )

        grid_data = _cached_event_grid(selected_id)
        model_choice = st.radio(
            "Select Forecast Model for Heatmap",
            list(MODEL_LABELS.keys()),
            format_func=lambda k: MODEL_LABELS[k],
            horizontal=True,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.pyplot(_plot_probability_heatmap(
                grid_data["rows"], MODEL_PROB_COL[model_choice], ev["extent_km"], MODEL_LABELS[model_choice]
            ))
        with col_r:
            st.pyplot(_plot_observed_grid(grid_data["rows"], ev["extent_km"]))

        st.markdown("### Inspect Individual Grid Cell")
        sc1, sc2 = st.columns(2)
        sel_row = sc1.slider("Grid Row Index (North-South)", 0, 39, 20)
        sel_col = sc2.slider("Grid Column Index (West-East)", 0, 39, 20)
        try:
            cell = get_cell(grid_data["rows"], sel_row, sel_col)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()
        cc1, cc2, cc3, cc4, cc5 = st.columns(5)
        cc1.metric("Offset (x, y) km", f"({cell['x']:.1f}, {cell['y']:.1f})")
        cc2.metric("Radial Baseline P", f"{cell['p_radial']:.4f}")
        cc3.metric("ETAS-Inspired P", f"{cell['p_etas']:.4f}")
        cc4.metric("Calibrated ML P", f"{cell['p_ml']:.4f}")
        cc5.metric("Observed Aftershocks", f"{cell['aftershock_count']} (label={cell['label']})")

    # 3. MODEL COMPARISON
    with tabs[2]:
        st.subheader("Chronological Test-Set Comparison (29 Held-Out Events, 46,400 Cells)")
        st.caption("ROC-AUC measures ranking discrimination across thresholds and is NOT classification accuracy.")
        chrono = eval_report["chronological_holdout"]["evaluation"]["models"]
        rows_cmp = []
        for mkey in ("radial_baseline", "etas_spatial_benchmark", "ml_calibrated"):
            cp = chrono[mkey]["cell_pooled"]
            em = chrono[mkey]["event_level_mean"]
            rows_cmp.append({
                "Model": MODEL_LABELS[mkey],
                "Pooled PR-AUC": round(cp["pr_auc"], 4),
                "Pooled ROC-AUC": round(cp["roc_auc"], 4),
                "Pooled Brier": round(cp["brier"], 5),
                "Pooled Log Loss": round(cp["log_loss"], 4),
                "Event Hit@5%": f"{em['hit_rate_5pct']:.2%}",
                "Event Hit@10%": f"{em['hit_rate_10pct']:.2%}",
                "Event Hit@20%": f"{em['hit_rate_20pct']:.2%}",
            })
        st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True, hide_index=True)
        st.info(
            "On the chronological test set, the radial baseline and ETAS-inspired spatial benchmark achieve slightly better "
            "pooled PR-AUC and calibration scores than the calibrated ML model. This indicates that the additional interaction "
            "features did not improve held-out performance in this experiment. The absence of depth and fault-plane geometry "
            "may limit the additional predictive information available to the ML model."
        )

        st.subheader("Secondary 5-Fold Event-Grouped CV (All 118 Events)")
        cv = eval_report["group_cv_5fold"]["models"]
        rows_cv = []
        for mkey in ("radial_baseline", "etas_spatial_benchmark", "ml_calibrated"):
            cp = cv[mkey]["cell_pooled"]
            em = cv[mkey]["event_level_mean"]
            rows_cv.append({
                "Model": MODEL_LABELS[mkey],
                "Pooled PR-AUC": round(cp["pr_auc"], 4),
                "Pooled ROC-AUC": round(cp["roc_auc"], 4),
                "Pooled Brier": round(cp["brier"], 5),
                "Pooled Log Loss": round(cp["log_loss"], 4),
                "Event Hit@10%": f"{em['hit_rate_10pct']:.2%}",
            })
        st.dataframe(pd.DataFrame(rows_cv), use_container_width=True, hide_index=True)

    # 4. PREDICTION ANALYSIS
    with tabs[3]:
        grid_data = _cached_event_grid(selected_id)
        st.subheader(f"Prediction Analysis for {selected_id}")
        st.write(
            "Predicted probabilities represent $P(\\text{aftershock count} \\ge 1)$ in each cell over 30 days, "
            "whereas `aftershock_count` is the observed integer count of M4.0+ aftershocks in that cell. "
            "Cell ranking orders all 1,600 cells from highest to lowest predicted probability."
        )
        analysis_model = st.selectbox(
            "Model for Distribution & Top-Ranked Cells",
            list(MODEL_LABELS.keys()),
            format_func=lambda k: MODEL_LABELS[k],
            key="analysis_model",
        )
        pcol = MODEL_PROB_COL[analysis_model]
        rcol = MODEL_RANK_COL[analysis_model]

        probs = [r[pcol] for r in grid_data["rows"]]
        fig, ax = plt.subplots(figsize=(7, 3.2))
        ax.hist(probs, bins=25, color="#1D3557", edgecolor="white")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Number of Grid Cells")
        ax.set_title(f"Predicted Probability Distribution ({MODEL_LABELS[analysis_model]})")
        fig.tight_layout()
        st.pyplot(fig)

        top_cells = sorted(grid_data["rows"], key=lambda r: r[rcol])[:15]
        df_top = pd.DataFrame([
            {
                "Rank": r[rcol],
                "Row": r["row_idx"],
                "Col": r["col_idx"],
                "x (km)": round(r["x"], 1),
                "y (km)": round(r["y"], 1),
                "r (km)": round(r["r_km"], 1),
                "Predicted Prob": round(r[pcol], 4),
                "Observed Count": r["aftershock_count"],
                "Observed Label": r["label"],
            }
            for r in top_cells
        ])
        st.markdown("### Top-15 Ranked Spatial Cells")
        st.dataframe(df_top, use_container_width=True, hide_index=True)

    # 5. METHODOLOGY & LIMITATIONS
    with tabs[4]:
        st.subheader("Methodology")
        st.markdown(
            "- **Data Source**: 118 selected earthquakes from M7.0+ USGS candidates (2000–2024; realized selected magnitudes M7.5–M9.1).\n"
            "- **T0-Only Features**: 14 base + 6 interaction features derived strictly from mainshock parameters "
            "available at $T_0$ (`mag`, `lat0`, `lon0`, grid offsets, and an empirical Wells & Coppersmith (1994) "
            "rupture-length estimate derived from mainshock magnitude; this is not a measured fault trace).\n"
            "- **30-Day Target Labels**: Future M4.0+ aftershocks in $(T_0, T_0 + 30\\text{ days}]$ binned onto the 40 x 40 grid.\n"
            "- **Chronological Split**: Earliest 89 events (`2000–2016`) for training; latest 29 events (`2016–2024`) for testing, "
            "with zero earthquake overlap across splits.\n"
            "- **Calibration & Evaluation**: 3-fold event-grouped out-of-fold Platt scaling inside training data; "
            "evaluated using PR-AUC, ROC-AUC, Brier score, Log Loss, and spatial Hit@5%/10%/20%."
        )
        st.subheader("Scientific Limitations & Disclaimers")
        st.warning(
            "1. **Selection Bias (>= 20 aftershocks)**: The original dataset excluded 12 M7.0+ candidates that did not satisfy "
            "the selection criterion of at least 20 recorded M4.0+ aftershocks; results apply to productive sequences only.\n"
            "2. **Grid Boundary**: 95.35% of recorded aftershocks fall inside the 40 x 40 grid [-E, E]; 4.65% fall outside.\n"
            "3. **Missing Source Parameters**: Mainshock `depth`, `strike`, `dip`, and `rake` are unavailable and not fabricated.\n"
            "4. **ETAS-Inspired Benchmark**: Uses a spatial triggering kernel without pre-T0 background rates; it is not a full space-time ETAS model.\n"
            "5. **Research Prototype**: Intended for academic study only—not suitable for operational earthquake forecasting or emergency decisions."
        )


if __name__ == "__main__":
    main()
