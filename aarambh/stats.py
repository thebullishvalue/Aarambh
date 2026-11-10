"""Mathematical primitives. Pure NumPy/SciPy: no Streamlit, no global state.

Every function is causal unless its docstring says it summarises a whole sample.
"""

from __future__ import annotations

import bisect

import numpy as np
import pandas as pd
from scipy import stats

from .config import DDM_LEAK_RATE, DDM_PRIOR_VAR, Z_EXTREME, Z_THRESHOLD

_IQR_TO_SIGMA = 1.349  # interquartile range of a standard normal


# ── Ornstein-Uhlenbeck ────────────────────────────────────────────────────────

def ornstein_uhlenbeck_estimate(series: np.ndarray, dt: float = 1.0) -> tuple[float, float, float]:
    """OU parameters from a bias-corrected AR(1) fit.

    Model: dx = θ(μ − x)dt + σ dW, sampled as x_t − μ = a (x_{t−1} − μ) + ε_t with
    a = e^{−θ dt} and Var(ε) = σ²(1 − a²) / (2θ).

    Least squares underestimates a in short samples. Kendall's (1954) first-order
    correction adds the bias back: a_c = â + (1 + 3â) / n.

    Returns (theta, mu, sigma). A sample too short to fit returns a slow default.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 20:
        return 0.05, float(np.mean(x)) if len(x) else 0.0, max(float(np.std(x)) if len(x) > 1 else 1.0, 1e-6)

    x0, x1 = x[:-1], x[1:]
    n = len(x0)
    var0 = float(np.var(x0))
    if var0 < 1e-12:
        return 0.05, float(np.mean(x)), 1e-6

    a_hat = float(np.mean((x0 - x0.mean()) * (x1 - x1.mean())) / var0)
    a = float(np.clip(a_hat + (1.0 + 3.0 * a_hat) / n, 1e-4, 0.999))

    mu = float(np.mean(x))
    eps = (x1 - mu) - a * (x0 - mu)
    var_eps = float(np.sum(eps ** 2) / max(n - 2, 1))

    theta = -np.log(a) / dt
    sigma = float(np.sqrt(max(var_eps * 2.0 * theta / (1.0 - a ** 2), 1e-12)))
    return float(theta), mu, max(sigma, 1e-6)


def ou_forecast(x0: float, theta: float, mu: float, sigma: float, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation of an OU process h = 1..horizon steps ahead."""
    h = np.arange(1, horizon + 1, dtype=float)
    decay = np.exp(-theta * h)
    mean = mu + (x0 - mu) * decay
    std = np.sqrt(sigma ** 2 * (1.0 - decay ** 2) / (2.0 * max(theta, 1e-8)))
    return mean, std


def half_life(theta: float) -> float:
    return float(np.log(2.0) / max(theta, 1e-8))


# ── Leaky evidence accumulator ("DDM") ────────────────────────────────────────

def drift_diffusion_filter(
    observations: np.ndarray,
    leak_rate: float = DDM_LEAK_RATE,
    prior_var: float = DDM_PRIOR_VAR,
) -> tuple[np.ndarray, np.ndarray]:
    """Leaky accumulation of noisy evidence, with a standard error.

    State:      s_t = (1 − λ) s_{t−1} + λ e_t        (unit gain: s stays on e's scale)
    Innovation: v_t = (1 − λ) v_{t−1} + λ (e_t − s_{t−1})²
    Std error:  se_t = sqrt(λ / (2 − λ) · v_t)       (variance of an EWMA of noise with variance v)

    Starts at the first finite observation, so nothing from the future leaks in.
    Missing evidence carries the state forward. Returns (state, std_error).
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    state = np.full(n, np.nan)
    se = np.full(n, np.nan)
    s: float | None = None
    v = float(prior_var)
    k = leak_rate / (2.0 - leak_rate)
    for i in range(n):
        e = obs[i]
        if np.isfinite(e):
            if s is None:
                s = float(e)
            else:
                v = (1.0 - leak_rate) * v + leak_rate * (e - s) ** 2
                s = (1.0 - leak_rate) * s + leak_rate * float(e)
        if s is not None:
            state[i] = s
            se[i] = np.sqrt(k * v)
    return state, se


def soft_bound(score: np.ndarray | float, max_bound: float = 100.0) -> np.ndarray | float:
    """Squash onto (−max_bound, max_bound) with tanh; near-linear for |score| ≪ max_bound."""
    return max_bound * np.tanh(np.asarray(score, dtype=float) / max_bound)


# ── Detrended fluctuation analysis ────────────────────────────────────────────

def hurst_dfa(series: np.ndarray, min_scale: int = 10, max_scale: int | None = None, n_scales: int = 16) -> float:
    """DFA-1 scaling exponent α of ``series`` (Peng et al., 1994). Whole-sample summary.

    Pass *increments* to judge mean reversion: α ≈ 0.5 for random-walk increments,
    α < 0.5 for anti-persistent (mean-reverting) increments, α > 0.5 for trending.
    Scales are log-spaced from ``min_scale`` to n/4. Returns NaN if too short.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    max_scale = max_scale or n // 4
    if n < 4 * min_scale or max_scale <= min_scale:
        return float("nan")

    profile = np.cumsum(x - x.mean())
    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int))
    log_s, log_f = [], []
    for s in scales:
        n_win = n // s
        if n_win < 2:
            continue
        segs = profile[: n_win * s].reshape(n_win, s)
        t = np.arange(s, dtype=float)
        t_c = t - t.mean()
        slope = (segs - segs.mean(axis=1, keepdims=True)) @ t_c / np.dot(t_c, t_c)
        fit = segs.mean(axis=1, keepdims=True) + slope[:, None] * t_c
        f = np.sqrt(np.mean((segs - fit) ** 2))
        if f > 1e-12:
            log_s.append(np.log(s))
            log_f.append(np.log(f))
    if len(log_s) < 4:
        return float("nan")
    return float(stats.linregress(log_s, log_f).slope)


# ── Calibrated (conformal) valuation-gap z-scores ─────────────────────────────

def robust_gap_scores(series: np.ndarray, window: int) -> np.ndarray:
    """Deviation of x_t from the median of the previous ``window`` values, in units of
    their robust scale (IQR / 1.349). Uses only past values; NaN until the window fills."""
    s = pd.Series(np.asarray(series, dtype=np.float64))
    past = s.shift(1).rolling(window, min_periods=window)
    med = past.median()
    scale = (past.quantile(0.75) - past.quantile(0.25)) / _IQR_TO_SIGMA
    raw = (s - med) / scale.where(scale > 1e-12)
    return raw.to_numpy(dtype=np.float64)


def calibrated_zscores(series: np.ndarray, window: int, min_calibration: int = 60) -> np.ndarray:
    """Valuation-gap z-scores calibrated against their own history (split-free conformal).

    1. Score each day with ``robust_gap_scores`` (median/IQR of the trailing window).
    2. Rank today's score among all *earlier* scores (mid-rank empirical CDF, F).
    3. z = Φ⁻¹(F).

    Whatever the window length, autocorrelation or tail shape, the result then has the
    same meaning: |z| > 1 on ~32% of days, |z| > 2 on ~5%. No day is ranked against
    the future. NaN until ``min_calibration`` earlier scores exist.
    """
    raw = robust_gap_scores(series, window)
    n = len(raw)
    z = np.full(n, np.nan)
    past: list[float] = []
    for i in range(n):
        r = raw[i]
        if not np.isfinite(r):
            continue
        m = len(past)
        if m >= min_calibration:
            lo = bisect.bisect_left(past, r)
            hi = bisect.bisect_right(past, r)
            z[i] = stats.norm.ppf((lo + 0.5 * (hi - lo) + 0.5) / (m + 1))
        bisect.insort(past, r)
    return z


def classify_zones(z_scores: np.ndarray) -> np.ndarray:
    """Map z-scores to valuation zone labels (``N/A`` where z is missing)."""
    z = np.asarray(z_scores, dtype=float)
    zones = np.select(
        [z > Z_EXTREME, z > Z_THRESHOLD, z >= -Z_THRESHOLD, z >= -Z_EXTREME],
        ["Extreme Over", "Overvalued", "Fair Value", "Undervalued"],
        default="Extreme Under",
    ).astype(object)
    zones[~np.isfinite(z)] = "N/A"
    return zones


def detect_crossover_signals(z_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Days where z crosses below −Z_THRESHOLD (buy) or above +Z_THRESHOLD (sell)."""
    z = np.asarray(z_scores, dtype=float)
    buy = np.zeros(len(z), dtype=bool)
    sell = np.zeros(len(z), dtype=bool)
    if len(z) < 2:
        return buy, sell
    cur, prev = z[1:], z[:-1]
    valid = np.isfinite(cur) & np.isfinite(prev)
    buy[1:] = valid & (cur < -Z_THRESHOLD) & (prev >= -Z_THRESHOLD)
    sell[1:] = valid & (cur > Z_THRESHOLD) & (prev <= Z_THRESHOLD)
    return buy, sell


# ── Inference with overlapping horizons ───────────────────────────────────────

def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (Bartlett) HAC standard errors.

    Forward changes over h days overlap, so neighbouring observations share h − 1
    days of the same move; plain t-stats overstate significance roughly √h-fold.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1.0 - lag / (lags + 1.0)
        g = xu[lag:].T @ xu[:-lag]
        S += w * (g + g.T)
    cov = xtx_inv @ S @ xtx_inv * n / max(n - k, 1)
    return beta, np.sqrt(np.clip(np.diag(cov), 0.0, None))


def information_coefficient(signal: np.ndarray, forward: np.ndarray, lags: int) -> dict[str, float]:
    """Correlation of a signal with later changes, with a HAC t-stat for the slope."""
    s = np.asarray(signal, dtype=float)
    f = np.asarray(forward, dtype=float)
    m = np.isfinite(s) & np.isfinite(f)
    n = int(m.sum())
    if n < 30 or np.std(s[m]) < 1e-12 or np.std(f[m]) < 1e-12:
        return {"ic": float("nan"), "t_stat": float("nan"), "n": n}
    beta, se = newey_west_ols(f[m], np.column_stack([np.ones(n), s[m]]), lags)
    return {
        "ic": float(np.corrcoef(s[m], f[m])[0, 1]),
        "t_stat": float(beta[1] / se[1]) if se[1] > 0 else float("nan"),
        "n": n,
    }


def count_episodes(mask: np.ndarray) -> int:
    """Number of separate runs of True (e.g. distinct BUY episodes)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return 0
    return int(m[0]) + int(np.sum(m[1:] & ~m[:-1]))
