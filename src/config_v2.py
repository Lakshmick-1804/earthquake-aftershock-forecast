"""Configuration for the v2 leak-free Earthquake Aftershock Forecasting pipeline.

Strictly preserves original data/ and prototype files. All outputs go to data_v2/.
"""
import math
from pathlib import Path

# Original read-only paths
ORIG_EVENTS_CSV = Path("data/events.csv")
ORIG_RAW_DIR = Path("data/raw")

# New isolated v2 paths
V2_DATA_DIR = Path("data_v2")
V2_RAW_DIR = V2_DATA_DIR / "raw"
V2_INTERIM_DIR = V2_DATA_DIR / "interim"
V2_PROCESSED_DIR = V2_DATA_DIR / "processed"
V2_REPORTS_DIR = V2_DATA_DIR / "reports"

# Fixed dataset parameters matching the college-provided 118-event dataset
EXPECTED_N_EVENTS = 118
FORECAST_DAYS = 30
GRID_N = 40
CELLS_PER_EVENT = GRID_N * GRID_N  # 1,600
R_SOFTENING = 0.15


def fault_length_km(mag: float) -> float:
    """Wells & Coppersmith (1994) style rupture length approximation from magnitude."""
    return 10.0 ** (-2.44 + 0.59 * mag)


def extent_km(length_km: float) -> float:
    """Half-width of the square forecast grid around the mainshock epicenter (km)."""
    return max(80.0, 2.0 * length_km)


def to_local_km(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Convert (lat, lon) to local East/North km offsets relative to (lat0, lon0)."""
    x = (lon - lon0) * 111.32 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110.57
    return x, y


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(min(1.0, math.sqrt(a)))
