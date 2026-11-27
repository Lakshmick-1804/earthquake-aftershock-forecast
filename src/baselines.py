"""Stage 2B: Statistical benchmarks fit strictly on training events.

1. Isotropic Radial Statistical Baseline:
   L2-regularized logistic regression on [log_r_norm, mag].
2. ETAS-Inspired Spatial Benchmark (NOT a complete ETAS model):
   Intensity lambda = exp(b0 + alpha*(mag - 7.5)) * (r_norm^2 + d^2)^(-q) * (cell_area_km2 / 1000)
   Probability P(Y >= 1) = 1 - exp(-lambda), fit by minimizing training Bernoulli log-loss.
"""
import math

from math_utils import (
    apply_standardizer,
    clamp_prob,
    fit_logistic_l2,
    fit_standardizer,
    nelder_mead,
    predict_probs,
)


def fit_radial_baseline(train_rows: list[dict]) -> dict:
    X_raw = [[float(r["log_r_norm"]), float(r["mag"])] for r in train_rows]
    y = [int(r["label"]) for r in train_rows]
    means, stds = fit_standardizer(X_raw)
    X = apply_standardizer(X_raw, means, stds)
    weights = fit_logistic_l2(X, y, l2=1.0, max_iter=25)
    return {"means": means, "stds": stds, "weights": weights}


def predict_radial_baseline(model: dict, rows: list[dict]) -> list[float]:
    X_raw = [[float(r["log_r_norm"]), float(r["mag"])] for r in rows]
    X = apply_standardizer(X_raw, model["means"], model["stds"])
    return predict_probs(X, model["weights"])


def _etas_probs(params: list[float], r2_list: list[float], mag_shift: list[float], area_scaled: list[float]) -> list[float]:
    b0, alpha, log_d2, log_q = params
    d2 = math.exp(max(-8.0, min(4.0, log_d2)))
    q = math.exp(max(-4.0, min(3.0, log_q)))
    out = []
    for r2, ms, area in zip(r2_list, mag_shift, area_scaled):
        lam = math.exp(max(-20.0, min(10.0, b0 + alpha * ms))) * ((r2 + d2) ** (-q)) * area
        p = 1.0 - math.exp(-min(lam, 30.0))
        out.append(clamp_prob(p))
    return out


def fit_etas_spatial_benchmark(train_rows: list[dict]) -> dict:
    r2_list = [float(r["r_norm"]) ** 2 for r in train_rows]
    mag_shift = [float(r["mag"]) - 7.5 for r in train_rows]
    area_scaled = [float(r["cell_area_km2"]) / 1000.0 for r in train_rows]
    y = [int(r["label"]) for r in train_rows]

    def objective(params: list[float]) -> float:
        probs = _etas_probs(params, r2_list, mag_shift, area_scaled)
        loss = 0.0
        for yi, pi in zip(y, probs):
            loss -= yi * math.log(pi) + (1 - yi) * math.log(1.0 - pi)
        # Mild L2 penalty on parameter magnitudes for numerical stability
        reg = 1e-4 * sum(v * v for v in params)
        return loss / len(y) + reg

    # Reasonable geophysical initialization: b0=-2.0, alpha=0.5, d2=0.15, q=1.5
    x0 = [-2.0, 0.5, math.log(0.15), math.log(1.5)]
    best = nelder_mead(objective, x0, step=0.3, max_iter=80, tol=1e-4)
    return {
        "b0": best[0],
        "alpha": best[1],
        "d2": math.exp(max(-8.0, min(4.0, best[2]))),
        "q": math.exp(max(-4.0, min(3.0, best[3]))),
        "raw_params": best,
    }


def predict_etas_spatial_benchmark(model: dict, rows: list[dict]) -> list[float]:
    r2_list = [float(r["r_norm"]) ** 2 for r in rows]
    mag_shift = [float(r["mag"]) - 7.5 for r in rows]
    area_scaled = [float(r["cell_area_km2"]) / 1000.0 for r in rows]
    return _etas_probs(model["raw_params"], r2_list, mag_shift, area_scaled)
