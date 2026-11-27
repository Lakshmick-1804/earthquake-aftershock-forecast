# Stage 1 Data Validation Report (118 Selected Earthquakes)

- **Selected events verified**: 118 (exact match to `data/events.csv`)
- **Excluded candidate raw CSVs in `data/raw/`**: 12 (all had < 20 aftershocks due to original `MIN_AFTERSHOCKS=20` filter)
- **Mainshock rows found inside raw CSVs**: 0 events ([])
- **Rows with `dt < 0` (before T0)**: 0 events
- **Rows with `dt > 30` days**: 0 events
- **Observed `dt` range across all 118 events**: [0.001186, 29.993210] days (min event max-dt: 12.29 days, median max-dt: 29.43 days)
- **Aftershock magnitude range**: [4.00, 8.20] (events with `mag < 4.0`: 0; median of event medians: 4.50)

## Provisional Chronological Distribution (Events per Year)

| Year | Event Count |
|---|---:|
| 2000 | 6 |
| 2001 | 6 |
| 2002 | 6 |
| 2003 | 5 |
| 2004 | 3 |
| 2005 | 4 |
| 2006 | 5 |
| 2007 | 8 |
| 2008 | 1 |
| 2009 | 8 |
| 2010 | 6 |
| 2011 | 4 |
| 2012 | 5 |
| 2013 | 5 |
| 2014 | 5 |
| 2015 | 8 |
| 2016 | 6 |
| 2017 | 3 |
| 2018 | 6 |
| 2019 | 1 |
| 2020 | 4 |
| 2021 | 5 |
| 2022 | 1 |
| 2023 | 6 |
| 2024 | 1 |

## Scientific & Completeness Notes
- **Selection bias**: Retaining only events with >= 20 M4+ aftershocks excludes 12 M7.5+ candidates; all results are conditional on productive sequences.
- **Completeness heterogeneity**: Although `AFTER_MIN_MAG = 4.0` was queried, catalog completeness varies by region, depth, and early-aftershock coda overlap.
- **Missing source parameters**: `depth`, `strike`, `dip`, and `rake` are absent from `data/events.csv` and are not fabricated.
