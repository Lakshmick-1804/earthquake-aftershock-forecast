# Aftershock Forecast Studio

An app-first prototype aftershock outlook workspace driven by the official **USGS Earthquake Catalog CSV** feeds. The page opens directly into a working dashboard: choose a mainshock from the catalog (or import a USGS-format CSV), review observed aftershock evidence, inspect a spatial estimate and priority zones, and export the results.

The default forecast engine is a **bundled Random Forest regressor** trained offline on USGS catalog data. The transparent catalog-calibrated baseline remains available from the engine selector for side-by-side comparison.

It also includes an **AI vs earlier manual method** benchmark section. It preserves the 118 historical earthquake comparison with pre-rendered heatmaps from the original study: AI likelihood, the earlier simplified stress-pattern baseline, the observed aftershock cloud, and ROC-AUC-style scores for both methods.

A **Relief & recovery** area provides official shelter, food, and safety links plus a needs checklist that is saved only in the browser. Nothing is submitted to an agency or external service.

The site is isolated from the unrelated React/Vite app in the repository root. There is no build step and no third-party JavaScript.

## Run it

Use a local server so the browser can fetch the live USGS CSV feed, the bundled CSV snapshot, and the model JSON:

```powershell
cd aftershock-site
python -m http.server 8765
```

Open <http://127.0.0.1:8765/>.

Opening `index.html` directly with `file://` may still render the UI, but browser CORS rules will block the live feed, the local CSV fallback, and the model JSON. Use the local server, or import a CSV file from your computer.

## Data sources

| Source | URL / file | Behaviour |
| --- | --- | --- |
| Live · past day | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.csv` | Alternative for very recent sequences |
| Live · past week | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.csv` | Default selection |
| Live · past month | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv` | Longer catalog window |
| Bundled snapshot | `assets/usgs-all-week.csv` | Used automatically if a live feed cannot be reached; captured 26 Nov 2026 |
| Training snapshot | `assets/usgs-all-month-training.csv` | USGS monthly CSV used to train the Random Forest |
| Bundled model | `assets/ml-model.json` | Exported Random Forest trees and validation metadata |
| Historical benchmark | `assets/comparison/meta.json` | 118-event index with names, magnitudes, and method scores |
| Historical summary | `assets/comparison/summary.json` | Average AI/manual scores and AI-win rate |
| Benchmark figures | `assets/comparison/*.png` | Pre-rendered AI, manual-method, and observed heatmaps |
| Import CSV | Local file picker | Accepts any USGS Earthquake Catalog CSV with the standard column headings |

The USGS feed sends `Access-Control-Allow-Origin: *`, so the browser can fetch it directly. The bundled snapshots are saved copies of the same official CSV format.

CSV rows with `type` other than `earthquake` are excluded. The parser handles quoted place names and accepts the standard USGS columns: `time`, `latitude`, `longitude`, `depth`, `mag`, `place`, `type`, `status`, `net`, `id`, and optional `url`.

## Forecast engines

Use **Forecast engine → Active model** in the left rail to switch engines.

### 1. Trained Random Forest v1.0 (default)

- Offline-trained `RandomForestRegressor`: 90 trees, maximum depth 8, exported to `assets/ml-model.json`.
- The browser evaluates the trees locally; there is no API, backend, or runtime network call for the model.
- The model predicts the expected count of **M≥3** events in the selected window.
- Inputs use only information available at the selected forecast start: mainshock magnitude and depth, elapsed time, forecast window, Omori fraction, early observed M≥3/all counts, early productivity scale, and an inferred trend from early events.
- The spatial distribution still uses the transparent anisotropic prior/observed kernel, so the map probability is `P = 1 − exp(−N̂_RF × w)`.

### 2. Catalog-calibrated baseline v0.2

- Transparent ETAS-inspired formula with Omori–Utsu decay, anisotropic spatial kernel, and a productivity scale from observed M≥3 counts capped to `0.2×–5×`.
- Probability per 3 km grid cell: `P = 1 − exp(−N × w)`.
- Useful as an interpretable comparison for the trained model.

For either engine, only observed catalog rows at or before the selected forecast start are used. Later rows are excluded so the outlook does not read its own future.

## How the catalog outlook works

1. The selected earthquake becomes the mainshock. Its magnitude, depth, coordinates, and hours since the mainshock are prefilled into the controls.
2. Earthquakes after the mainshock, within a magnitude-scaled search radius (80–220 km) and the available catalog time, are treated as **observed evidence**. They are shown in the Catalog evidence table and plotted on the map.
3. The active engine estimates the expected M≥3 count in the selected window.
4. That count is distributed over a 40 × 25 grid of 3 km cells using the spatial kernel, producing per-cell probabilities, priority zones, and the daily-rate chart.
5. The result is an **estimate**, not a validated prediction. The word “forecast” in the UI means a prototype decision-support output, not an official warning or public-safety product.

The three schematic presets (Japan Trench, Chile Margin, California) remain available for comparing the catalog-driven view with a controlled scenario. In that mode the early-event dots are explicitly labelled synthetic.

## Model validation

The bundled model was trained with `tools/train_ml.py` using the USGS monthly CSV snapshot captured for this build.

| Metric (time-ordered holdout) | Random Forest v1.0 | Baseline v0.2 |
| --- | ---: | ---: |
| Test samples | 418 | 418 |
| Mainshocks in test | 98 | 98 |
| MAE, M≥3 count | **0.3789** | 8.4350 |
| RMSE, M≥3 count | **0.6332** | 21.5667 |
| Brier score, “≥1 M≥3” | **0.1258** | 0.7641 |
| Log loss | **0.4055** | 4.8711 |

These numbers are an honest **same-split prototype comparison** on one monthly catalog. They are not out-of-sample validation across regions or time periods, not a calibration certificate, and not an accuracy claim. The baseline is intentionally simple and overpredicts many small sequences; the Random Forest learns a much more conservative count from the same features.

To retrain or inspect the process:

```powershell
python -m pip install numpy scikit-learn
python tools\train_ml.py
```

The script downloads the USGS monthly feed only if the training snapshot is missing, writes `assets/usgs-all-month-training.csv`, retrains the forest, and rewrites `assets/ml-model.json` with updated validation metadata.

## AI vs earlier manual method

The historical benchmark section is an additional static feature. It loads `assets/comparison/meta.json` and `summary.json`, then swaps three pre-rendered figures for the selected event:

- **AI prediction** — brighter cells are higher likelihood from the neural-network model used in the original benchmark.
- **Earlier manual method** — the simplified stress-pattern baseline from the same study.
- **What really happened** — yellow grid cells that contain observed aftershocks.

It also shows the per-event ROC-AUC-style score for both methods and the overall summary: AI average `0.88`, manual-method average `0.51`, and AI higher in `96%` of events. These figures are preserved historical benchmark outputs, not live forecasts and not a validation of operational prediction.

## Relief & recovery

The relief section provides:

- official shelter and disaster-assistance links (Red Cross, FEMA, 211);
- official food and water guidance (Feeding America, Red Cross, 211);
- official safety and preparedness guidance (USGS, CDC, Ready.gov);
- a personal needs checklist saved to `localStorage` under `earthquake-aftershock-forecast.needs.v1`;
- a CSV export of the checklist and a printable family contact card.

The app does **not** dispatch help, submit a request, collect contact details, or send data to any agency. The checklist intentionally avoids names, addresses, and phone numbers. A future backend can be added separately; this build deliberately has none.

## Validation of the web app

- Browser-tested with the live USGS feed, bundled fallback, simulated feed outage, CSV import, mainshock switching, schematic presets, map hover, layer toggles, engine switching, and the method dialog.
- Grid CSV, observed-event CSV, map PNG, and needs-checklist CSV exports were generated and inspected.
- Needs checklist save, restore after reload, autosave, and clear were verified in `localStorage`.
- Historical comparison selector verified with all 118 events; AI, manual, and observed figures swap correctly and the 118-event summary populates.
- Print contact card button verified; print stylesheet shows only the contact card.
- No page-level horizontal overflow at desktop, 390 px, or 320 px viewports. Tables scroll inside their own panels.
- Lighthouse: **Accessibility 100**, **Best Practices 100**, **SEO 100**.
- `node --check app.js` passes.

## Files

```text
aftershock-site/
├─ index.html                     app shell, catalog controls, engine selector, comparison, relief
├─ styles.css                     workspace, comparison, catalog, engine, relief, print and responsive rules
├─ app.js                         USGS CSV loader, Random Forest inference, map, benchmark selector, exports, checklist
├─ assets/
│  ├─ usgs-all-week.csv           bundled USGS fallback snapshot
│  ├─ usgs-all-month-training.csv USGS CSV used for offline training
│  ├─ ml-model.json               exported Random Forest trees and validation metadata
│  └─ comparison/                 118-event AI vs manual benchmark figures and metadata
├─ tools/
│  └─ train_ml.py                 offline Random Forest training/export script
└─ README.md                      this file
```

## Limitations

- The Random Forest is a teaching prototype, not an operationally validated aftershock prediction model.
- The holdout split comes from one monthly catalog; regional and temporal generalisation were not established.
- Observed events in the same CSV are used as a prototype calibration signal, not as out-of-sample validation against a separate catalog.
- Magnitudes use the USGS `mag` value and are labelled `M`, not `Mw`, because the feed mixes magnitude types.
- The inferred spatial trend is approximate, not authoritative fault geometry.
- Official relief links can change; availability of shelters and food services must be confirmed locally.
- This tool must not be used for emergency response, evacuation orders, or public-safety decisions.
