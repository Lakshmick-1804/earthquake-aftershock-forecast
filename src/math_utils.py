"""Pure-Python numerical helpers for regularized logistic regression, Platt scaling, and metrics."""
from __future__ import annotations

import math
import random


def sigmoid(z: float) -> float:
    if z >= 0.0:
        ez = math.exp(-min(z, 60.0))
        return 1.0 / (1.0 + ez)
    ez = math.exp(max(z, -60.0))
    return ez / (1.0 + ez)


def clamp_prob(p: float, eps: float = 1e-12) -> float:
    return min(1.0 - eps, max(eps, p))


def fit_standardizer(X: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(X)
    d = len(X[0])
    means = [0.0] * d
    for row in X:
        for j in range(d):
            means[j] += row[j]
    means = [m / n for m in means]

    vars_ = [0.0] * d
    for row in X:
        for j in range(d):
            diff = row[j] - means[j]
            vars_[j] += diff * diff
    stds = [math.sqrt(v / n) if v > 1e-18 else 1.0 for v in vars_]
    return means, stds


def apply_standardizer(X: list[list[float]], means: list[float], stds: list[float]) -> list[list[float]]:
    out = []
    d = len(means)
    for row in X:
        out.append([(row[j] - means[j]) / stds[j] for j in range(d)])
    return out


def solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Gaussian elimination with partial pivoting for small dense systems."""
    n = len(b)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-14:
            M[col][col] = 1e-14
        inv = 1.0 / M[col][col]
        for j in range(col, n + 1):
            M[col][j] *= inv
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                factor = M[r][col]
                for j in range(col, n + 1):
                    M[r][j] -= factor * M[col][j]
    return [M[i][n] for i in range(n)]


def fit_logistic_l2(
    X: list[list[float]],
    y: list[int],
    l2: float = 1.0,
    max_iter: int = 25,
    weights: list[float] | None = None,
) -> list[float]:
    """Fit L2-regularized logistic regression. Uses Newton-IRLS for d<=3, mini-batch Adam for d>3."""
    n = len(X)
    d = len(X[0])
    p_dim = d + 1
    w = [0.0] * p_dim
    sum_w = sum(weights) if weights is not None else float(n)
    sum_wy = sum((weights[i] if weights is not None else 1.0) * y[i] for i in range(n))
    p0 = clamp_prob(sum_wy / max(sum_w, 1.0), 1e-4)
    w[0] = math.log(p0 / (1.0 - p0))

    if d > 3:
        # Fast mini-batch Adam (O(d) per row instead of O(d^2))
        rng = random.Random(42)
        idxs = list(range(n))
        m_adam = [0.0] * p_dim
        v_adam = [0.0] * p_dim
        lr = 0.08
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        batch_size = 8192
        step_t = 0
        epochs = min(max_iter, 6)
        for _ in range(epochs):
            rng.shuffle(idxs)
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                bs = end - start
                g = [0.0] * p_dim
                for pos in range(start, end):
                    i = idxs[pos]
                    wi = weights[i] if weights is not None else 1.0
                    xi = X[i]
                    z = w[0]
                    for j in range(d):
                        z += w[j + 1] * xi[j]
                    err = wi * (sigmoid(z) - y[i])
                    g[0] += err
                    for j in range(d):
                        g[j + 1] += err * xi[j]
                inv_bs = 1.0 / bs
                g[0] *= inv_bs
                for j in range(1, p_dim):
                    g[j] = g[j] * inv_bs + (l2 / n) * w[j]
                step_t += 1
                b1t = 1.0 - beta1 ** step_t
                b2t = 1.0 - beta2 ** step_t
                for j in range(p_dim):
                    gj = g[j]
                    m_adam[j] = beta1 * m_adam[j] + (1.0 - beta1) * gj
                    v_adam[j] = beta2 * v_adam[j] + (1.0 - beta2) * gj * gj
                    m_hat = m_adam[j] / b1t
                    v_hat = v_adam[j] / b2t
                    w[j] -= lr * m_hat / (math.sqrt(v_hat) + eps)
        return w

    for _ in range(max_iter):
        g = [0.0] * p_dim
        H = [[0.0] * p_dim for _ in range(p_dim)]
        for i in range(n):
            wi = weights[i] if weights is not None else 1.0
            xi = X[i]
            z = w[0]
            for j in range(d):
                z += w[j + 1] * xi[j]
            pi = sigmoid(z)
            err = wi * (pi - y[i])
            wt = wi * pi * (1.0 - pi) + 1e-12

            g[0] += err
            H[0][0] += wt
            for j in range(d):
                xij = xi[j]
                g[j + 1] += err * xij
                H[0][j + 1] += wt * xij
                H[j + 1][0] += wt * xij
                wtx = wt * xij
                for k in range(j, d):
                    H[j + 1][k + 1] += wtx * xi[k]

        for j in range(1, p_dim):
            g[j] += l2 * w[j]
            H[j][j] += l2
            for k in range(j):
                H[j][k] = H[k][j]
        H[0][0] += 1e-8

        step = solve_linear(H, g)
        max_step = max(abs(s) for s in step)
        for j in range(p_dim):
            w[j] -= step[j]
        if max_step < 1e-6:
            break
    return w


def predict_logits(X: list[list[float]], w: list[float]) -> list[float]:
    d = len(w) - 1
    out = []
    for row in X:
        z = w[0]
        for j in range(d):
            z += w[j + 1] * row[j]
        out.append(z)
    return out


def predict_probs(X: list[list[float]], w: list[float]) -> list[float]:
    return [clamp_prob(sigmoid(z)) for z in predict_logits(X, w)]


def fit_platt_scaler(logits: list[float], y: list[int]) -> tuple[float, float]:
    """Fit 1D Platt scaling p = sigmoid(a * logit + b) on out-of-fold logits."""
    X1 = [[z] for z in logits]
    w = fit_logistic_l2(X1, y, l2=1e-4, max_iter=25)
    return w[1], w[0]  # (a, b)


def apply_platt_scaler(logits: list[float], a: float, b: float) -> list[float]:
    return [clamp_prob(sigmoid(a * z + b)) for z in logits]


def nelder_mead(fn, x0: list[float], step: float = 0.25, max_iter: int = 180, tol: float = 1e-5) -> list[float]:
    """Derivative-free Nelder-Mead minimization for low-dimensional smooth objectives."""
    n = len(x0)
    simplex = [x0[:]]
    for i in range(n):
        pt = x0[:]
        pt[i] += step
        simplex.append(pt)
    fvals = [fn(pt) for pt in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if max(abs(fvals[i] - fvals[0]) for i in range(1, n + 1)) < tol:
            break

        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        xr = [centroid[j] + 1.0 * (centroid[j] - simplex[n][j]) for j in range(n)]
        fr = fn(xr)
        if fvals[0] <= fr < fvals[n - 1]:
            simplex[n], fvals[n] = xr, fr
            continue
        if fr < fvals[0]:
            xe = [centroid[j] + 2.0 * (xr[j] - centroid[j]) for j in range(n)]
            fe = fn(xe)
            if fe < fr:
                simplex[n], fvals[n] = xe, fe
            else:
                simplex[n], fvals[n] = xr, fr
            continue
        xc = [centroid[j] + 0.5 * (simplex[n][j] - centroid[j]) for j in range(n)]
        fc = fn(xc)
        if fc < fvals[n]:
            simplex[n], fvals[n] = xc, fc
            continue
        for i in range(1, n + 1):
            simplex[i] = [simplex[0][j] + 0.5 * (simplex[i][j] - simplex[0][j]) for j in range(n)]
            fvals[i] = fn(simplex[i])
    return simplex[0]


def roc_auc_score(y_true: list[int], y_score: list[float]) -> float:
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    paired = sorted(zip(y_score, y_true), key=lambda t: t[0])
    rank_sum = 0.0
    i = 0
    n = len(paired)
    while i < n:
        j = i
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        pos_in_tie = sum(paired[k][1] for k in range(i, j))
        rank_sum += pos_in_tie * avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision_score(y_true: list[int], y_score: list[float]) -> float:
    n_pos = sum(y_true)
    if n_pos == 0:
        return float("nan")
    paired = sorted(zip(y_score, y_true), key=lambda t: -t[0])
    tp = 0
    ap = 0.0
    for idx, (_, y) in enumerate(paired, start=1):
        if y == 1:
            tp += 1
            ap += tp / idx
    return ap / n_pos


def brier_score(y_true: list[int], y_prob: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / len(y_true)


def log_loss(y_true: list[int], y_prob: list[float]) -> float:
    total = 0.0
    for y, p in zip(y_true, y_prob):
        pc = clamp_prob(p)
        total -= y * math.log(pc) + (1 - y) * math.log(1.0 - pc)
    return total / len(y_true)


def spatial_hit_rate(counts: list[int], y_score: list[float], top_frac: float) -> float:
    total_after = sum(counts)
    if total_after == 0:
        return float("nan")
    k = max(1, int(round(len(counts) * top_frac)))
    order = sorted(range(len(counts)), key=lambda i: -y_score[i])
    captured = sum(counts[i] for i in order[:k])
    return captured / total_after


def bootstrap_ci(event_ids: list[str], by_event: dict, metric_fn, n_boot: int = 400, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    n_ev = len(event_ids)
    vals = []
    for _ in range(n_boot):
        sample_ids = [event_ids[rng.randrange(n_ev)] for _ in range(n_ev)]
        y_all, p_all = [], []
        for eid in sample_ids:
            y_all.extend(by_event[eid]["y"])
            p_all.extend(by_event[eid]["p"])
        vals.append(metric_fn(y_all, p_all))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]
