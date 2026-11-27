# Earthquake Aftershock Forecasting

An app-first prototype aftershock outlook workspace driven by the official **USGS Earthquake Catalog CSV** feeds. This repository contains the finished static website, its model assets, the offline training script, and the ready-to-share website ZIP.

## What is included

```text
aftershock-site/                         complete static website and model assets
docs/                                    GitHub Pages copy of the website
deliverables/Earthquake_Aftershock_Forecasting_Website.zip   packaged website
```

### Website

- Live USGS CSV feeds, bundled USGS fallback snapshot, and local CSV import
- Default **Random Forest v1.0** estimator trained offline on USGS catalog data
- Selectable transparent **catalog-calibrated baseline v0.2**
- Spatial probability map, priority zones, rate chart, catalog evidence table
- Additional **AI vs earlier manual method** benchmark with 118 historical events and pre-rendered prediction/observed heatmaps
- PNG/CSV exports and a local-only relief & recovery checklist
- No backend, no API key, no runtime network call for the model

## Run the website

```powershell
cd aftershock-site
python -m http.server 8765 --bind 127.0.0.1
```

Open <http://127.0.0.1:8765/>.

Do not open `index.html` with `file://`; browser CORS rules will block the live feed, bundled CSV, and model JSON.

## Model notes

The bundled Random Forest is a compact teaching prototype exported to `aftershock-site/assets/ml-model.json` and evaluated directly in the browser. It predicts expected M≥3 aftershock counts from catalog-derived features. The spatial kernel then distributes that count over a 3 km grid.

A time-ordered 80/20 holdout on one USGS monthly catalog is reported in the app:

| Metric | Random Forest | Baseline |
| --- | ---: | ---: |
| MAE, M≥3 count | 0.3789 | 8.4350 |
| Brier score | 0.1258 | 0.7641 |

This is a same-split prototype comparison only. It is **not** an accuracy claim, operational validation, or a public-safety product.

The historical benchmark section is separate: `aftershock-site/assets/comparison/` contains pre-rendered figures and metadata for the original 118-event AI vs manual-method comparison. It is a preserved static benchmark, not a live forecast.

To retrain:

```powershell
python -m pip install numpy scikit-learn
cd aftershock-site
python tools\train_ml.py
```

## Detailed documentation

See `aftershock-site/README.md` for data sources, method, validation, and limitations.

## Responsible use

This tool must not be used for emergency response, evacuation orders, or public-safety decisions. The relief section links to official guidance and stores a checklist only in the browser; it never submits data to an agency.
