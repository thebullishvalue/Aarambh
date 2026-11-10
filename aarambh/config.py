"""Every tunable constant in one place.

Thresholds are on the scales defined in ``aarambh.stats`` and ``aarambh.engine``:
z-scores are calibrated so |z| > 1 happens on ~32% of days and |z| > 2 on ~5%;
conviction is the net % of lookbacks flagging over- vs undervaluation (±150 raw,
soft-bounded to ±100).
"""

from __future__ import annotations

VERSION = "3.3.0"
PRODUCT_NAME = "Aarambh"
COMPANY = "@thebullishvalue"

# ── Walk-forward fair-value regression ────────────────────────────────────────
MIN_TRAIN_SIZE = 20            # first fit needs this many observations
REFIT_INTERVAL = 5             # refit every N observations (fixed: never depends on sample length)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 500
ENSEMBLE_WEIGHT_WINDOW = 60    # trailing out-of-sample errors used to weight models
ENSEMBLE_WEIGHT_MIN_OBS = 10   # below this, models are weighted equally
ENSEMBLE_WEIGHT_FLOOR = 0.05   # no model's normalised weight falls below 5%

# Model memory: how much history the fair value learns from.
#   "adaptive": rolling 500-observation window, observations weighted with a
#               252-observation half-life. Fair value tracks the recent
#               relationship; gaps close in weeks.
#   "long":     expanding window, equal weights. Fair value is the long-run
#               relationship; gaps can persist for months.
MODEL_MEMORY = {
    "adaptive": {"max_train": 500, "decay_half_life": 252.0},
    "long": {"max_train": None, "decay_half_life": None},
}
DEFAULT_MODEL_MEMORY = "adaptive"

# ── Multi-lookback valuation-gap z-scores ────────────────────────────────────
LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
CALIBRATION_MIN = 60           # past scores needed before a lookback reports a z-score
Z_THRESHOLD = 1.0              # |z| beyond this: over/undervalued
Z_EXTREME = 2.0                # |z| beyond this: extreme

# ── Conviction (net breadth of lookbacks) and its smoothing ──────────────────
CONVICTION_WEAK = 20
CONVICTION_MODERATE = 40
CONVICTION_STRONG = 60
DDM_LEAK_RATE = 0.08           # leaky accumulator: half-life ln(0.5)/ln(0.92) ≈ 8.3 observations
DDM_PRIOR_VAR = 100.0          # innovation variance before any evidence arrives

# ── Ornstein-Uhlenbeck diagnostics on the valuation gap ──────────────────────
OU_WINDOW = 252                # trailing window for the "current" half-life
OU_HISTORY_STEP = 21           # sample the rolling θ every N observations
OU_PROJECTION_DAYS = 90

# ── Validation of the system's own signals ───────────────────────────────────
FORWARD_HORIZONS = (5, 10, 20)
HEADLINE_HORIZON = 20          # horizon used for the headline edge statistics

# ── Data ─────────────────────────────────────────────────────────────────────
MIN_DATA_POINTS = 80
STALENESS_DAYS = 3
TIMEFRAME_TRADING_DAYS = {"1M": 21, "6M": 126, "1Y": 252, "2Y": 504}

DEFAULT_TARGET = "NIFTY50_PE"
# Mirrors the predictors marked SELECTOR = 1 in the source sheet's own model.
DEFAULT_PREDICTORS = (
    "AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH",
    "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO",
    "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB",
)
