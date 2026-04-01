# -*- coding: utf-8 -*-
"""
AARAMBH (आरंभ) v3.2.1 — Fair Value Breadth
A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Walk-forward valuation analysis for market reversals. Identifies where market 
turns BEGIN using out-of-sample ensemble fair value modeling, Ornstein-Uhlenbeck 
mean-reversion physics with Andrews (1993) median-unbiased estimation, and 
Drift-Diffusion (SPRT) accumulated breadth conviction scoring.

Architecture:
    1. Mathematical Primitives   — OU estimation, DDM filter, Hurst exponent (DFA)
    2. Statistical Tests         — Andrews MU estimator, Bai-Perron breaks, KPSS/ADF
    3. FairValueEngine           — Walk-forward regression + all downstream analytics
    4. Data Utilities            — Loading, cleaning, chart theming
    5. UI Rendering              — Landing page, 4-tab dashboard, footer
    6. Main Application          — Sidebar config, engine orchestration, tab layout

"""

from __future__ import annotations

import html
import logging
import re
import time
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

# ── Optional Dependencies ────────────────────────────────────────────────────

try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.regime_switching.bai_perron import BaiPerronTest
    _HAS_STATSMODELS = True
except ImportError:
    sm = None
    _HAS_STATSMODELS = False

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression, RidgeCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

VERSION = "3.2.1"
PRODUCT_NAME = "Aarambh"
COMPANY = "Hemrek Capital"

# Engine defaults
LOOKBACK_WINDOWS = (5, 10, 20, 50, 100)
MIN_TRAIN_SIZE = 20
MAX_TRAIN_SIZE = 500
REFIT_INTERVAL = 5
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
HUBER_EPSILON = 1.35
HUBER_MAX_ITER = 500
OU_PROJECTION_DAYS = 90
MIN_DATA_POINTS = 80

# Signal thresholds (conviction score → signal mapping)
CONVICTION_STRONG = 60
CONVICTION_MODERATE = 40
CONVICTION_WEAK = 20

# Z-score zone boundaries
Z_EXTREME = 2.0
Z_THRESHOLD = 1.0

# Staleness
STALENESS_DAYS = 3

# Timeframe filter mapping (trading days)
TIMEFRAME_TRADING_DAYS = {"1M": 21, "6M": 126, "1Y": 252, "2Y": 504}

# Default predictors for NIFTY50 use case
DEFAULT_PREDICTORS = (
    "AD_RATIO", "COUNT", "REL_AD_RATIO", "REL_BREADTH",
    "IN10Y", "IN02Y", "IN30Y", "INIRYY", "REPO",
    "US02Y", "US10Y", "US30Y", "NIFTY50_DY", "NIFTY50_PB",
)

# Default Google Sheets URL (should be moved to environment variable)
DEFAULT_SHEET_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/"
    "edit?gid=1938234952#gid=1938234952"
)

# Chart theme
CHART_BG = "#1A1A1A"
CHART_GRID = "#2A2A2A"
CHART_ZEROLINE = "#3A3A3A"
CHART_FONT_COLOR = "#EAEAEA"

# Signal colors
COLOR_GREEN = "#10b981"
COLOR_RED = "#ef4444"
COLOR_GOLD = "#FFC300"
COLOR_CYAN = "#06b6d4"
COLOR_AMBER = "#f59e0b"
COLOR_MUTED = "#888888"

# DDM parameters (calibrated for daily conviction series)
DDM_LEAK_RATE = 0.08
DDM_DRIFT_SCALE = 0.15
DDM_LONG_RUN_VAR = 100.0  # Calibrated to conviction score variance


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION — Claude Code Style Terminal Output
# ══════════════════════════════════════════════════════════════════════════════

class TerminalLogger:
    """Curated terminal logging with Claude Code-style progress indicators."""
    
    def __init__(self):
        self.start_time = None
        self.phase = None
        
    def start(self, phase: str):
        """Mark the start of a processing phase."""
        self.phase = phase
        self.start_time = time.time()
        logging.info(f"\n{'='*80}")
        logging.info(f"  {phase}")
        logging.info(f"{'='*80}")
        
    def checkpoint(self, step: str, details: str = ""):
        """Log a checkpoint within a phase."""
        elapsed = f" ({time.time() - self.start_time:.1f}s)" if self.start_time else ""
        if details:
            logging.info(f"  ✓ {step}: {details}{elapsed}")
        else:
            logging.info(f"  ✓ {step}{elapsed}")
            
    def complete(self, summary: str = ""):
        """Mark phase completion with summary."""
        elapsed = f"{time.time() - self.start_time:.1f}s" if self.start_time else "N/A"
        logging.info(f"\n  ▶ Phase Complete: {summary}")
        logging.info(f"  ▶ Total Time: {elapsed}\n")
        
    def error(self, step: str, error: str):
        """Log an error with context."""
        logging.error(f"  ✗ {step}: {error}")
        
    def warning(self, step: str, warning: str):
        """Log a warning with context."""
        logging.warning(f"  ⚠ {step}: {warning}")

# Global logger instance
logger = TerminalLogger()

# Configure logging for detailed terminal output
logging.basicConfig(
    level=logging.INFO,
    format="\n%(asctime)s\n%(message)s\n",
    datefmt="%H:%M:%S"
)

# Force flush to ensure logs are visible in Streamlit
class StreamlitLogHandler(logging.Handler):
    """Custom handler to force log output in Streamlit environment."""
    def emit(self, record):
        msg = self.format(record)
        print(msg, flush=True)

# Add custom handler if not already present
if not logging.getLogger().handlers:
    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("\n%(asctime)s\n%(message)s\n", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="AARAMBH | Fair Value Breadth",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --purple: #8b5cf6;
        --neutral: #888888;
    }

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }

    [data-testid="collapsedControl"] {
        display: flex !important; visibility: visible !important; opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important; border-radius: 8px !important;
        padding: 10px !important; margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important; position: fixed !important;
        top: 14px !important; left: 14px !important;
        width: 40px !important; height: 40px !important;
        align-items: center !important; justify-content: center !important;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important; transform: scale(1.05);
    }
    [data-testid="collapsedControl"] svg { stroke: var(--primary-color) !important; width: 20px !important; height: 20px !important; }
    [data-testid="stSidebar"] button[kind="header"] { background-color: transparent !important; border: none !important; }
    [data-testid="stSidebar"] button[kind="header"] svg { stroke: var(--primary-color) !important; }
    button[kind="header"] { z-index: 999999 !important; }

    .premium-header {
        background: var(--secondary-background-color); padding: 1.25rem 2rem; border-radius: 16px;
        margin-bottom: 1.5rem; box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color); position: relative; overflow: hidden; margin-top: 1rem;
    }
    .premium-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }

    .metric-card {
        background-color: var(--bg-card); padding: 1.25rem; border-radius: 12px;
        border: 1px solid var(--border-color); box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative; overflow: hidden; min-height: 160px;
        display: flex; flex-direction: column; justify-content: center;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; min-height: 30px; display: flex; align-items: center; }
    .metric-card h3 { color: var(--text-primary); font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .metric-card p { color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin: 0; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.primary h2 { color: var(--primary-color); }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.purple h2 { color: var(--purple); }
    .metric-card.neutral h2 { color: var(--neutral); }

    .signal-card { background: var(--bg-card); border-radius: 16px; border: 2px solid var(--border-color); padding: 1.5rem; position: relative; overflow: hidden; }
    .signal-card.overvalued { border-color: var(--danger-red); box-shadow: 0 0 30px rgba(239, 68, 68, 0.15); }
    .signal-card.undervalued { border-color: var(--success-green); box-shadow: 0 0 30px rgba(16, 185, 129, 0.15); }
    .signal-card.fair { border-color: var(--primary-color); box-shadow: 0 0 30px rgba(255, 195, 0, 0.15); }
    .signal-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; }
    .signal-card .value { font-size: 2.5rem; font-weight: 700; line-height: 1; }
    .signal-card .subtext { font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.75rem; }
    .signal-card.overvalued .value { color: var(--danger-red); }
    .signal-card.undervalued .value { color: var(--success-green); }
    .signal-card.fair .value { color: var(--primary-color); }

    .guide-box { background: rgba(var(--primary-rgb), 0.05); border-left: 3px solid var(--primary-color); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: var(--text-secondary); font-size: 0.9rem; }
    .guide-box.success { background: rgba(16, 185, 129, 0.05); border-left-color: var(--success-green); }
    .guide-box.danger { background: rgba(239, 68, 68, 0.05); border-left-color: var(--danger-red); }
    .guide-box.warning { background: rgba(245, 158, 11, 0.05); border-left-color: var(--warning-amber); }
    .guide-box.info { background: rgba(6, 182, 212, 0.05); border-left-color: var(--info-cyan); }

    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }

    .conviction-meter { background: var(--bg-elevated); border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
    .conviction-bar { height: 8px; border-radius: 4px; background: var(--border-color); overflow: hidden; }
    .conviction-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }

    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }

    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }

    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }

    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }

    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    
    /* Table Styling (Swing System Design) */
    .table-container {
        width: 100%;
        overflow-x: auto;
        border-radius: 12px;
        background: var(--bg-card);
        border: 1px solid var(--border-color);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        padding: 0;
    }
    table.table {
        width: 100%;
        table-layout: auto;
        border-collapse: collapse;
        color: var(--text-primary);
    }
    table.table th, table.table td {
        padding: 1rem 1.2rem;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }
    table.table th {
        font-weight: 600;
        color: var(--primary-color);
        background-color: var(--bg-elevated);
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.05em;
    }
    table.table td {
        font-size: 0.95rem;
    }
    table.table tr:hover {
        background: var(--bg-elevated);
    }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    .stTextInput > div > div > input { background: var(--bg-elevated) !important; border: 1px solid var(--border-color) !important; border-radius: 8px !important; color: var(--text-primary) !important; }
    .stTextInput > div > div > input:focus { border-color: var(--primary-color) !important; box-shadow: 0 0 0 2px rgba(var(--primary-rgb), 0.2) !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def ornstein_uhlenbeck_estimate(
    series: np.ndarray,
    dt: float = 1.0,
) -> tuple[float, float, float]:
    """Estimate OU process parameters via AR(1) regression with Andrews MU correction.

    Model: dx = θ(μ − x)dt + σdW

    Uses Andrews (1993) median-unbiased estimator for near-unit-root cases.

    Returns:
        (theta, mu, sigma) — mean-reversion speed, equilibrium level, volatility.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 20:
        if len(x) > 1:
            return 0.05, 0.0, max(float(np.std(x)), 1e-6)
        return 0.05, 0.0, 1.0

    x_lag = x[:-1]
    x_curr = x[1:]
    n = len(x_lag)

    sx = np.sum(x_lag)
    sy = np.sum(x_curr)
    sxx = np.dot(x_lag, x_lag)
    sxy = np.dot(x_lag, x_curr)

    denom = n * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return 0.05, float(np.mean(x)), max(float(np.std(x)), 1e-6)

    a = (n * sxy - sx * sy) / denom
    b = (sy * sxx - sx * sxy) / denom
    a = np.clip(a, 1e-6, 0.999)

    # Andrews (1993) median-unbiased correction for AR(1) coefficient
    # Handles near-unit-root cases better than jackknife (Tang & Chen, 2009)
    if a > 0.95:
        # Second-order correction for persistent series
        a_corrected = a - (1 + 3 * a) / n - 3 * (1 + 3 * a) / (n ** 2)
    else:
        a_corrected = a - (1 + 3 * a) / n
    
    a = np.clip(a_corrected, 0.0, 0.999)

    theta = -np.log(a) / dt if a > 1e-6 else 0.05
    mu = b / (1 - a) if abs(1 - a) > 1e-6 else float(np.mean(x))

    residuals = x_curr - a * x_lag - b
    sigma_sq = np.var(residuals)
    
    if a > 0.98:
        sigma = max(float(np.std(residuals)) * np.sqrt(2 * max(theta, 1e-4)), 1e-6)
    else:
        sigma = np.sqrt(max(sigma_sq * 2 * theta / (1 - a ** 2), 1e-12))

    return max(float(theta), 1e-4), float(mu), max(float(sigma), 1e-6)


def drift_diffusion_filter(
    observations: np.ndarray,
    leak_rate: float = DDM_LEAK_RATE,
    drift_scale: float = DDM_DRIFT_SCALE,
    long_run_var: float = DDM_LONG_RUN_VAR,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D Drift-Diffusion filter with mean-reverting variance.

    State equation: state_t = (1-λ)×state_{t-1} + λ×drift_t
    Variance equation: var_t = (1-λ)×var_{t-1} + λ×σ²_LR + 0.5×|drift_t|

    Returns:
        (filtered_state, dummy_gains, estimate_variances)
    """
    obs = np.asarray(observations, dtype=np.float64)
    n = len(obs)
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    state = float(np.mean(obs[:min(20, n)])) if n > 0 else 0.0
    var = long_run_var

    filtered = np.zeros(n)
    variances = np.zeros(n)
    filtered[0] = state
    variances[0] = var

    for i in range(1, n):
        evidence = obs[i] if np.isfinite(obs[i]) else 0.0
        drift = evidence * drift_scale

        # Leaky integration with mean-reversion
        state = state * (1 - leak_rate) + drift

        # Mean-reverting variance with capped drift-dependent expansion
        # Capped to prevent ballooning variance during prolonged regimes
        var = var * (1 - leak_rate) + leak_rate * long_run_var + min(abs(drift) * 0.5, long_run_var * 0.5)
        var = max(var, 1e-6)  # Prevent collapse

        filtered[i] = state
        variances[i] = var

    return filtered, np.zeros(n), variances


def hurst_dfa(series: np.ndarray, min_scale: int = 4, max_scale: int | None = None) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis (DFA).

    DFA is more robust than R/S for short, noisy series (Peng et al., 1994).
    Uses proper lag range: min_scale = max(4, n/10), max_scale = n/4

    Returns:
        H < 0.5 → mean-reverting, H ≈ 0.5 → random walk, H > 0.5 → trending.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    
    if n < 30:
        return 0.5

    # Proper lag range per DFA literature
    if max_scale is None:
        max_scale = n // 4
    min_scale = max(min_scale, n // 10)
    
    if min_scale >= max_scale:
        return 0.5

    # Cumulative sum (profile)
    y = np.cumsum(x - np.mean(x))

    scales = range(min_scale, max_scale + 1, max(1, (max_scale - min_scale) // 15))
    log_scales = []
    log_fluctuations = []

    for scale in scales:
        # Split into non-overlapping windows of size 'scale'
        n_windows = n // scale
        if n_windows < 2:
            continue

        rms_errors = []
        for i in range(n_windows):
            segment = y[i * scale : (i + 1) * scale]
            
            # Linear detrending
            x_seg = np.arange(scale)
            slope, intercept = np.polyfit(x_seg, segment, 1)
            trend = slope * x_seg + intercept
            
            # Fluctuation around trend
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            if rms > 1e-10:
                rms_errors.append(rms)

        if rms_errors:
            fluctuation = np.sqrt(np.mean(np.array(rms_errors) ** 2))
            if fluctuation > 1e-10:
                log_scales.append(np.log(scale))
                log_fluctuations.append(np.log(fluctuation))

    if len(log_scales) < 3:
        return 0.5

    slope, _, _, _, _ = stats.linregress(log_scales, log_fluctuations)
    return float(np.clip(slope, 0.01, 0.99))


def andrews_median_unbiased_ar1(series: np.ndarray) -> tuple[float, float]:
    """Andrews (1993) median-unbiased AR(1) estimator with confidence interval.
    
    Returns:
        (ar_coef, half_life) with median-unbiased correction.
    """
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    
    if n < 20:
        return 0.0, np.inf
    
    x_lag = x[:-1]
    x_curr = x[1:]
    
    # OLS estimate
    a_ols = np.corrcoef(x_lag, x_curr)[0, 1]
    
    # Andrews MU correction
    if a_ols > 0.95:
        a_mu = a_ols - (1 + 3 * a_ols) / n - 3 * (1 + 3 * a_ols) / (n ** 2)
    else:
        a_mu = a_ols - (1 + 3 * a_ols) / n
    
    a_mu = np.clip(a_mu, 0.0, 0.999)
    
    half_life = np.log(0.5) / np.log(a_mu) if a_mu > 0.01 else np.inf
    
    return a_mu, half_life


def detect_structural_breaks(
    series: np.ndarray,
    max_breaks: int = 3,
    trim: float = 0.15,
) -> list[int]:
    """Bai-Perron multiple breakpoint detection.
    
    Returns:
        List of break indices (relative to series start).
    """
    if not _HAS_STATSMODELS or len(series) < 50:
        return []
    
    try:
        # Bai-Perron test for structural changes
        bp_test = BaiPerronTest(series)
        result = bp_test.test_breaks(max_breaks, trim=trim)
        
        if hasattr(result, 'break_dates') and result.break_dates is not None:
            return [int(bd) for bd in result.break_dates]
    except Exception as e:
        logging.warning("Bai-Perron test failed: %s", e)
    
    return []


def compute_conformal_zscores(
    series: np.ndarray,
    window: int,
    min_periods: int = 5,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Conformal prediction-based z-scores with fat-tail adjustment.
    
    Uses empirical quantiles instead of mean/std for robustness to fat tails.
    
    Returns:
        (z_scores, lower_bound, upper_bound) at conformal level 1-alpha.
    """
    n = len(series)
    z_scores = np.full(n, np.nan)
    lower_bounds = np.full(n, np.nan)
    upper_bounds = np.full(n, np.nan)
    
    s = pd.Series(series)
    
    for i in range(window, n):
        window_data = series[i - window : i]  # EXCLUDE current point (no look-ahead)
        
        if np.sum(np.isfinite(window_data)) < min_periods:
            continue
        
        # Empirical quantiles for conformal interval
        q_lower = np.nanpercentile(window_data, alpha / 2 * 100)
        q_upper = np.nanpercentile(window_data, (1 - alpha / 2) * 100)
        q_median = np.nanmedian(window_data)
        
        # Z-score via quantile normalization (robust to outliers)
        iqr = q_upper - q_lower
        if iqr > 1e-10:
            z_scores[i] = (series[i] - q_median) / (iqr / 1.35)  # 1.35 ≈ normal IQR/sigma
        
        lower_bounds[i] = q_lower
        upper_bounds[i] = q_upper
    
    return z_scores, lower_bounds, upper_bounds


# ══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES (DRY)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_array_operation(
    arr: np.ndarray,
    operation: str,
    default: float = 0.0,
) -> float:
    """Safely compute common array operations with NaN handling."""
    arr = np.asarray(arr)
    valid = np.isfinite(arr)
    
    if not np.any(valid):
        return default
    
    clean = arr[valid]
    
    if operation == "mean":
        return float(np.mean(clean))
    elif operation == "std":
        return float(np.std(clean)) if len(clean) > 1 else default
    elif operation == "min":
        return float(np.min(clean))
    elif operation == "max":
        return float(np.max(clean))
    elif operation == "sum":
        return float(np.sum(clean))
    else:
        return default


def _classify_zones(z_scores: np.ndarray) -> np.ndarray:
    """Map z-scores to valuation zone labels."""
    condlist = [
        z_scores > Z_EXTREME,
        z_scores > Z_THRESHOLD,
        z_scores > -Z_THRESHOLD,
        z_scores > -Z_EXTREME
    ]
    choicelist = [
        "Extreme Over",
        "Overvalued",
        "Fair Value",
        "Undervalued"
    ]
    zones = np.select(condlist, choicelist, default="Extreme Under")
    np.putmask(zones, np.isnan(z_scores), "N/A")
    return zones


def _detect_crossover_signals(z_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect z-score threshold crossovers as discrete signals."""
    n = len(z_scores)
    if n < 2:
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    
    z_curr = z_scores[1:]
    z_prev = z_scores[:-1]
    
    valid = np.isfinite(z_curr) & np.isfinite(z_prev)
    
    buy_cond = valid & (z_curr < -Z_THRESHOLD) & (z_prev >= -Z_THRESHOLD)
    sell_cond = valid & (z_curr > Z_THRESHOLD) & (z_prev <= Z_THRESHOLD)
    
    buy_signals = np.zeros(n, dtype=bool)
    sell_signals = np.zeros(n, dtype=bool)
    
    buy_signals[1:] = buy_cond
    sell_signals[1:] = sell_cond
    
    return buy_signals, sell_signals


def _compute_significance(
    values: list[float],
) -> dict[str, float]:
    """Compute t-statistic and p-value for a list of values."""
    n = len(values)
    
    if n < 3:
        return {"mean": 0.0, "std": 0.0, "t_stat": 0.0, "p_value": 1.0, "n": n}
    
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    
    if std_val < 1e-10:
        return {"mean": mean_val, "std": std_val, "t_stat": np.inf, "p_value": 0.0, "n": n}
    
    se = std_val / np.sqrt(n)
    t_stat = mean_val / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    
    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n": n,
    }


def _apply_conviction_bounds(score: float, max_bound: float = 100.0) -> float:
    """Apply soft bounds to conviction score via tanh transformation."""
    return max_bound * np.tanh(score / max_bound)


# ══════════════════════════════════════════════════════════════════════════════
# FAIR VALUE BREADTH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class FairValueEngine:
    """Walk-forward fair value engine with multi-lookback breadth analytics.

    Pipeline:
        1. Expanding-window ensemble regression (Ridge + Huber + OLS)
        2. Multi-lookback conformal z-score computation and zone classification
        3. Breadth aggregation and raw conviction scoring
        4. Drift-Diffusion filtering of conviction with mean-reverting variance
        5. OU estimation with Andrews MU correction for half-life and projection
        6. Hurst exponent via DFA for mean-reversion validation
        7. Swing-based divergence detection
        8. Forward change analysis with significance testing
        9. Structural break detection for regime-aware resetting
    """

    def __init__(self) -> None:
        self.ts_data: pd.DataFrame = pd.DataFrame()
        self.lookback_data: dict = {}
        self.model_stats: dict = {}
        self.ou_params: dict = {}
        self.ou_projection: np.ndarray = np.array([])
        self.ou_projection_upper: np.ndarray = np.array([])
        self.ou_projection_lower: np.ndarray = np.array([])
        self.pivots: dict = {}
        self.residual_stats: dict = {}
        self.hurst: float = 0.5
        self.latest_feature_impacts: dict = {}
        self.feature_impact_history: list[dict] = []
        self.theta_history: list[float] = []
        self.break_dates: list[int] = []

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        progress_callback=None,
    ) -> FairValueEngine:
        """Run the full walk-forward pipeline."""
        start_time = time.time()

        self.feature_names = feature_names or [f"X{i}" for i in range(X.shape[1])]
        self.n_samples = len(y)
        self.y = y.copy()

        # Start curated logging
        logger.start(f"AARAMBH v3.2.1 — Fair Value Breadth Engine")
        logger.checkpoint("Initialization", f"{self.n_samples} observations, {len(self.feature_names)} features")

        # Detect structural breaks before walk-forward
        if progress_callback:
            progress_callback(0.05, "Detecting structural breaks...")
        logger.checkpoint("Structural Break Detection", "Bai-Perron multiple breakpoint test")
        self.break_dates = detect_structural_breaks(y)
        if self.break_dates:
            logger.checkpoint("Breaks Found", f"{len(self.break_dates)} breaks at indices {self.break_dates}")
        else:
            logger.checkpoint("Breaks Found", "None detected (continuous regime)")

        # Walk-forward regression
        logger.checkpoint("Walk-Forward Regression", "Expanding window ensemble (Ridge + Huber + ElasticNet + WLS)")
        self._walk_forward_regression(X, y, progress_callback)

        # Multi-lookback signals
        if progress_callback:
            progress_callback(1.0, "Computing multi-lookback signals...")
        logger.checkpoint("Multi-Lookback Signals", f"Lookbacks: {LOOKBACK_WINDOWS}")

        self.residuals = y - self.predictions

        # Compute all downstream analytics
        logger.checkpoint("Model Statistics", "OOS R², RMSE, MAE, R² vs Random Walk")
        self._compute_model_stats()
        
        logger.checkpoint("Breadth Metrics", "Oversold/Overbought breadth aggregation")
        self._compute_multi_lookback_signals()
        self._compute_breadth_metrics()
        
        logger.checkpoint("DDM Conviction", "Drift-Diffusion filter with mean-reverting variance")
        self._compute_ddm_conviction()
        
        logger.checkpoint("Pivot Detection", "Swing-based extremum detection")
        self._find_pivots()
        
        logger.checkpoint("Divergence Detection", "Bullish/Bearish divergence identification")
        self._compute_divergences()
        
        logger.checkpoint("Forward Changes", "5D/10D/20D winsorized forward returns")
        self._compute_forward_changes()
        
        logger.checkpoint("OU Diagnostics", "Andrews MU half-life, ADF/KPSS stationarity tests")
        self._compute_ou_diagnostics()
        
        logger.checkpoint("Hurst Exponent", "DFA with stationarity guard (ADF pre-check)")
        self._compute_hurst()

        # Completion summary
        elapsed = time.time() - start_time
        logger.complete(f"Engine ready | OOS R²: {self.model_stats.get('r2_oos', 0):.3f} | Half-life: {self.ou_params.get('half_life', 0):.0f}d | Hurst: {self.hurst:.2f}")

        if progress_callback:
            progress_callback(1.0, "Done.")

        return self

    def get_current_signal(self) -> dict:
        """Derive the current composite signal from the latest observation."""
        if self.ts_data.empty:
            return {
                "signal": "HOLD", "strength": "NEUTRAL", "confidence": "N/A",
                "conviction_score": 0, "conviction_upper": 0, "conviction_lower": 0,
                "regime": "NEUTRAL", "oversold_breadth": 0, "overbought_breadth": 0,
                "residual": 0, "fair_value": 0, "actual": 0, "avg_z": 0,
                "model_spread": 0, "has_bullish_div": False, "has_bearish_div": False,
                "ou_half_life": 0, "adf_pvalue": 1.0, "kpss_pvalue": 0.0, "hurst": 0.5,
                "theta_stable": True, "break_detected": False,
            }
        
        current = self.ts_data.iloc[-1]
        conviction_bounded = current["ConvictionBounded"]

        if conviction_bounded < -CONVICTION_STRONG:
            signal, strength = "BUY", "STRONG"
        elif conviction_bounded < -CONVICTION_MODERATE:
            signal, strength = "BUY", "MODERATE"
        elif conviction_bounded < -CONVICTION_WEAK:
            signal, strength = "BUY", "WEAK"
        elif conviction_bounded > CONVICTION_STRONG:
            signal, strength = "SELL", "STRONG"
        elif conviction_bounded > CONVICTION_MODERATE:
            signal, strength = "SELL", "MODERATE"
        elif conviction_bounded > CONVICTION_WEAK:
            signal, strength = "SELL", "WEAK"
        else:
            signal, strength = "HOLD", "NEUTRAL"

        oversold_breadth = current["OversoldBreadth"]
        overbought_breadth = current["OverboughtBreadth"]

        # Confidence based on breadth conviction and conviction score magnitude
        if signal == "BUY":
            confidence = "HIGH" if oversold_breadth >= 80 else "MEDIUM" if oversold_breadth >= 60 else "LOW"
        elif signal == "SELL":
            confidence = "HIGH" if overbought_breadth >= 80 else "MEDIUM" if overbought_breadth >= 60 else "LOW"
        else:
            # For HOLD signals, confidence based on how neutral the conviction is
            conviction_abs = abs(conviction_bounded)
            if conviction_abs < 10:
                confidence = "HIGH"  # Very neutral, strong HOLD
            elif conviction_abs < 20:
                confidence = "MEDIUM"  # Moderately neutral
            else:
                confidence = "LOW"  # Weak HOLD, near threshold

        # Check theta stability
        theta_stable = True
        if len(self.theta_history) >= 10:
            theta_cv = np.std(self.theta_history[-10:]) / max(np.mean(self.theta_history[-10:]), 1e-6)
            theta_stable = theta_cv < 0.5

        return {
            "signal": signal,
            "strength": strength,
            "confidence": confidence,
            "conviction_score": conviction_bounded,
            "conviction_upper": current["ConvictionUpper"],
            "conviction_lower": current["ConvictionLower"],
            "regime": current["Regime"],
            "oversold_breadth": oversold_breadth,
            "overbought_breadth": overbought_breadth,
            "residual": current["Residual"],
            "fair_value": current["FairValue"],
            "actual": current["Actual"],
            "avg_z": current["AvgZ"],
            "model_spread": current["ModelSpread"],
            "has_bullish_div": current["BullishDiv"],
            "has_bearish_div": current["BearishDiv"],
            "ou_half_life": self.ou_params.get("half_life", 0),
            "adf_pvalue": self.ou_params.get("adf_pvalue", 1.0),
            "kpss_pvalue": self.ou_params.get("kpss_pvalue", 0.0),
            "hurst": self.hurst,
            "theta_stable": theta_stable,
            "break_detected": len(self.break_dates) > 0,
        }

    def get_model_stats(self) -> dict:
        return self.model_stats

    def get_regime_stats(self) -> dict:
        ts = self.ts_data
        regime_counts = ts["Regime"].value_counts()
        return {
            "strongly_oversold": regime_counts.get("STRONGLY OVERSOLD", 0),
            "oversold": regime_counts.get("OVERSOLD", 0),
            "neutral": regime_counts.get("NEUTRAL", 0),
            "overbought": regime_counts.get("OVERBOUGHT", 0),
            "strongly_overbought": regime_counts.get("STRONGLY OVERBOUGHT", 0),
            "current_regime": ts["Regime"].iloc[-1],
            "total_buy_signals": ts["BuySignalBreadth"].sum(),
            "total_sell_signals": ts["SellSignalBreadth"].sum(),
            "total_bull_div": ts["BullishDiv"].sum(),
            "total_bear_div": ts["BearishDiv"].sum(),
            "total_pivot_tops": ts["IsPivotTop"].sum(),
            "total_pivot_bottoms": ts["IsPivotBottom"].sum(),
        }

    def get_signal_performance(self) -> dict:
        """Forward change analysis with significance testing."""
        ts = self.ts_data
        results = {}
        
        burn_in = max(MIN_TRAIN_SIZE + 50, MIN_DATA_POINTS)
        
        for period in (5, 10, 20):
            buy_changes: list[float] = []
            sell_changes: list[float] = []
            
            for i in range(burn_in, len(ts) - period):
                score = ts["ConvictionScore"].iloc[i]
                fwd = ts[f"FwdChg_{period}"].iloc[i]
                if pd.isna(fwd):
                    continue
                if score < -CONVICTION_MODERATE:
                    buy_changes.append(fwd)
                if score > CONVICTION_MODERATE:
                    sell_changes.append(-fwd)
            
            # Compute significance statistics
            buy_stats = _compute_significance(buy_changes)
            sell_stats = _compute_significance(sell_changes)
            
            results[period] = {
                "buy_avg": buy_stats["mean"],
                "buy_hit": float(np.mean([c > 0 for c in buy_changes])) if buy_changes else 0.0,
                "buy_count": len(buy_changes),
                "buy_t_stat": buy_stats["t_stat"],
                "buy_p_value": buy_stats["p_value"],
                "sell_avg": sell_stats["mean"],
                "sell_hit": float(np.mean([c > 0 for c in sell_changes])) if sell_changes else 0.0,
                "sell_count": len(sell_changes),
                "sell_t_stat": sell_stats["t_stat"],
                "sell_p_value": sell_stats["p_value"],
            }
        
        return results

    def get_feature_impact_history(self) -> pd.DataFrame:
        """Return time-series of feature impacts."""
        if not self.feature_impact_history:
            return pd.DataFrame()
        return pd.DataFrame(self.feature_impact_history)

    # ── Private: Walk-Forward Regression ──────────────────────────────────

    def _walk_forward_regression(
        self, X: np.ndarray, y: np.ndarray, progress_callback,
    ) -> None:
        """Expanding-window ensemble regression with periodic refitting.

        Sequential implementation for deterministic, reproducible results.
        """
        n = self.n_samples
        self.predictions = np.full(n, np.nan)
        self.model_spread = np.zeros(n)

        # Pre-fill initial minimum train size
        for t in range(MIN_TRAIN_SIZE):
            self.predictions[t] = float(np.mean(y[:t])) if t > 0 else float(y[0])
            self.model_spread[t] = 0.0

        # Precompute global exponential decay weights
        decay_rate = np.log(2) / 252.0
        global_weights = np.exp(-decay_rate * np.arange(MAX_TRAIN_SIZE - 1, -1, -1))

        # Dynamic refit interval
        dynamic_refit = int(np.clip(n // 150, 5, 10))

        last_models: dict = {"ridge": None, "huber": None, "ols": None, "elasticnet": None, "pca_wls": None}
        valid_cols = np.ones(X.shape[1], dtype=bool)
        chunk_results = {}
        total_chunks = (n - MIN_TRAIN_SIZE) // dynamic_refit + 1

        logger.checkpoint("Walk-Forward Progress", f"Processing {total_chunks} chunks ({dynamic_refit} samples each)")

        # Sequential execution (deterministic, no threading warnings)
        for i, t_start in enumerate(range(MIN_TRAIN_SIZE, n, dynamic_refit)):
            t_end = min(t_start + dynamic_refit, n)
            try:
                result = self._process_wf_chunk(t_start, t_end, X, y, global_weights)
                t_start_res, t_end_res, preds, spreads, models, v_cols = result

                self.predictions[t_start_res:t_end_res] = preds
                self.model_spread[t_start_res:t_end_res] = spreads
                chunk_results[(t_start_res, t_end_res)] = (models, v_cols)

                # Log progress every 10 chunks
                if (i + 1) % 10 == 0 or (i + 1) == total_chunks:
                    logging.info(f"    → Chunk {i+1}/{total_chunks}: Samples {t_start}→{t_end} processed")

                if progress_callback and (i + 1) % max(1, total_chunks // 20) == 0:
                    progress_callback(
                        (i + 1) / total_chunks,
                        f"Walking forward... ({t_end}/{n} samples)"
                    )
            except Exception as e:
                logger.warning(f"Chunk [{t_start}:{t_end}]", str(e))

        # Extract features from last chunk
        if chunk_results:
            last_chunk_key = max(chunk_results.keys(), key=lambda k: k[1])
            last_models, valid_cols = chunk_results[last_chunk_key]

        self._compute_feature_impacts(last_models, valid_cols, n)

    def _process_wf_chunk(
        self,
        t_start: int,
        t_end: int,
        X: np.ndarray,
        y: np.ndarray,
        global_weights: np.ndarray,
    ) -> tuple[int, int, np.ndarray, np.ndarray, dict, np.ndarray]:
        """Process a single walk-forward block (fit at start, predict up to end)."""
        valid_breaks = [b for b in self.break_dates if b < t_start]
        last_break = valid_breaks[-1] if valid_breaks else 0
        
        # Reset window if a break happened, preserving at least MIN_TRAIN_SIZE memory if possible
        max_lookback = max(0, t_start - MAX_TRAIN_SIZE)
        
        if last_break > max_lookback:
            start_idx = last_break
            if t_start - start_idx < MIN_TRAIN_SIZE:
                # Fallback to a minimum window overriding the break temporarily to establish initial variance
                start_idx = max(0, t_start - MIN_TRAIN_SIZE)
        else:
            start_idx = max_lookback

        models, scaler, valid_cols = self._fit_ensemble(
            X[start_idx:t_start], y[start_idx:t_start], t_start, global_weights
        )

        X_chunk = X[t_start:t_end]
        if len(X_chunk) == 0:
            return t_start, t_end, np.array([]), np.array([]), models, valid_cols

        # Evaluate out-of-sample models on validation split dynamically for inverse-MAE weights
        val_size = min(30, max(5, int((t_start - start_idx) * 0.2)))
        X_val = X[t_start - val_size : t_start] if t_start > val_size else None
        y_val = y[t_start - val_size : t_start] if t_start > val_size else None

        preds_matrix, weights = self._predict_ensemble(
            X_chunk, models, scaler, valid_cols, t_start, X_val, y_val
        )

        if preds_matrix:
            preds_stacked = np.vstack(preds_matrix)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # Weighted average using normalized inverse-MAE from trailing validation window
                if len(preds_matrix) > 1 and len(weights) == len(preds_matrix) and sum(weights) > 0:
                    w = np.array(weights) / sum(weights)
                    preds = np.average(preds_stacked, axis=0, weights=w)
                else:
                    preds = np.nanmean(preds_stacked, axis=0)
                spreads = np.maximum(np.nanstd(preds_stacked, axis=0), 1e-6) if len(preds_matrix) > 1 else np.full(len(preds), 1e-6)

                nans = np.isnan(preds)
                if np.any(nans):
                    preds[nans] = float(np.mean(y[start_idx:t_start]))
                    spreads[nans] = 1e-6
        else:
            fallback = float(np.mean(y[start_idx:t_start]))
            preds = np.full(t_end - t_start, fallback)
            spreads = np.full(t_end - t_start, 1e-6)

        return t_start, t_end, preds, spreads, models, valid_cols

    @staticmethod
    def _fit_ensemble(
        X_train: np.ndarray, y_train: np.ndarray, t: int, global_weights: np.ndarray,
    ) -> tuple[dict, StandardScaler | None, np.ndarray]:
        """Fit Ridge + Huber + WLS ensemble on training data with exponential decay weighting."""
        models: dict = {"ridge": None, "huber": None, "ols": None, "elasticnet": None, "pca_wls": None}
        scaler = None

        valid_cols = np.std(X_train, axis=0) > 1e-8
        if not np.any(valid_cols):
            valid_cols = np.ones(X_train.shape[1], dtype=bool)

        X_train_clean = X_train[:, valid_cols]
        n_samples = len(y_train)
        weights = global_weights[-n_samples:]

        if _HAS_SKLEARN:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train_clean)

            try:
                ridge = RidgeCV(alphas=list(RIDGE_ALPHAS), cv=None)
                ridge.fit(X_scaled, y_train, sample_weight=weights)
                models["ridge"] = ridge
            except Exception as e:
                logging.warning("Ridge fit failed at t=%d: %s", t, e)

            try:
                huber = HuberRegressor(epsilon=HUBER_EPSILON, max_iter=HUBER_MAX_ITER, tol=1e-3)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    huber.fit(X_scaled, y_train, sample_weight=weights)
                models["huber"] = huber
            except Exception as e:
                logging.warning("Huber fit failed at t=%d: %s", t, e)

            try:
                enet = ElasticNetCV(
                    l1_ratio=[0.5, 0.9, 1.0],
                    n_alphas=10,
                    cv=2,
                    max_iter=2000,
                    tol=1e-2,
                    selection="random",
                    n_jobs=1
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    enet.fit(X_scaled, y_train, sample_weight=weights)
                models["elasticnet"] = enet
            except Exception as e:
                logging.warning("ElasticNet fit failed at t=%d: %s", t, e)

            try:
                pca = PCA(n_components=0.95, svd_solver="full")
                X_pca = pca.fit_transform(X_scaled)
                models["pca_wls"] = pca

                ols = LinearRegression()
                ols.fit(X_pca, y_train, sample_weight=weights)
                models["ols"] = ols
            except Exception as e:
                logging.warning("PCA/OLS fit failed at t=%d: %s", t, e)

        return models, scaler, valid_cols

    def _predict_ensemble(
        self,
        X_pred: np.ndarray,
        models: dict,
        scaler: StandardScaler | None,
        valid_cols: np.ndarray,
        t_start: int,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[float]]:
        """Collect vectorized predictions and compute inverse-MAE weights."""
        preds_list: list[np.ndarray] = []
        weights: list[float] = []

        def _add_safe_pred(arr_pred: np.ndarray, model_func) -> None:
            arr_clean = np.where(np.isfinite(arr_pred) & (np.abs(arr_pred) < 1e10), arr_pred, np.nan)
            if not np.all(np.isnan(arr_clean)):
                preds_list.append(arr_clean)
                
                # Inverse-MAE Weighting with 5% floor
                if X_val is not None and len(X_val) > 0 and scaler is not None:
                    try:
                        val_scaled = scaler.transform(X_val[:, valid_cols])
                        val_pred = model_func(val_scaled)
                        mae = mean_absolute_error(y_val, val_pred)
                        weights.append(max(1.0 / max(mae, 1e-6), 0.05))
                    except Exception:
                        weights.append(0.05)
                else:
                    weights.append(1.0)

        X_pred_clean = X_pred[:, valid_cols]

        if _HAS_SKLEARN and scaler is not None:
            try:
                X_scaled = scaler.transform(X_pred_clean)
                if models["ridge"] is not None:
                    try:
                        _add_safe_pred(models["ridge"].predict(X_scaled), models["ridge"].predict)
                    except Exception as e:
                        logging.warning("Ridge predict failed at t=%d: %s", t_start, e)
                if models["huber"] is not None:
                    try:
                        _add_safe_pred(models["huber"].predict(X_scaled), models["huber"].predict)
                    except Exception as e:
                        logging.warning("Huber predict failed at t=%d: %s", t_start, e)
                if models.get("elasticnet") is not None:
                    try:
                        _add_safe_pred(models["elasticnet"].predict(X_scaled), models["elasticnet"].predict)
                    except Exception as e:
                        logging.warning("ElasticNet predict failed at t=%d: %s", t_start, e)
                if models.get("ols") is not None and models.get("pca_wls") is not None:
                    try:
                        X_pca_pred = models["pca_wls"].transform(X_scaled)
                        def pca_wls_predict(X_input):
                            return models["ols"].predict(models["pca_wls"].transform(X_input))
                        _add_safe_pred(models["ols"].predict(X_pca_pred), pca_wls_predict)
                    except Exception as e:
                        logging.warning("OLS predict failed at t=%d: %s", t_start, e)
            except Exception as e:
                logging.warning("Prediction cascade failed at t=%d: %s", t_start, e)

        return preds_list, weights

    def _compute_feature_impacts(self, models: dict, valid_cols: np.ndarray, t_index: int) -> None:
        """Map PCA+WLS coefficients back to original features."""
        features = np.array(self.feature_names)[valid_cols]
        wls = models.get("ols")
        pca = models.get("pca_wls")

        if wls is not None and pca is not None:
            try:
                wls_weights = wls.coef_
                feature_weights = np.dot(wls_weights, pca.components_)

                abs_weights = np.abs(feature_weights)
                total_impact = np.sum(abs_weights)
                if total_impact > 1e-10:
                    pct_impacts = (abs_weights / total_impact) * 100
                    impacts = {f: float(imp) for f, imp in zip(features, pct_impacts)}
                    self.latest_feature_impacts = dict(sorted(impacts.items(), key=lambda x: x[1], reverse=True))
                    
                    # Store in history
                    self.feature_impact_history.append({
                        "index": t_index,
                        **impacts,
                    })
                    return
            except Exception as e:
                logging.warning("Failed to compute feature impacts: %s", e)

        self.latest_feature_impacts = {}

    # ── Private: Analytics Pipeline ───────────────────────────────────────

    def _compute_model_stats(self) -> None:
        """OOS model fit statistics (only walk-forward portion)."""
        oos_mask = np.arange(self.n_samples) >= MIN_TRAIN_SIZE
        y_oos = self.y[oos_mask]
        pred_oos = self.predictions[oos_mask]

        valid = np.isfinite(pred_oos)
        y_v, p_v = y_oos[valid], pred_oos[valid]

        if len(y_v) > 2 and _HAS_SKLEARN:
            r2 = r2_score(y_v, p_v)
            rmse = float(np.sqrt(mean_squared_error(y_v, p_v)))
            mae = float(mean_absolute_error(y_v, p_v))
        else:
            ss_res = float(np.sum((y_v - p_v) ** 2))
            ss_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
            r2 = 1 - ss_res / max(ss_tot, 1e-10)
            rmse = float(np.sqrt(np.mean((y_v - p_v) ** 2)))
            mae = float(np.mean(np.abs(y_v - p_v)))

        # R² vs random walk (Welch & Goyal, 2008)
        if len(y_v) > 2:
            rw_forecast = np.empty_like(y_v)
            rw_forecast[0] = y_v[0]
            rw_forecast[1:] = y_v[:-1]
            ss_res = float(np.sum((y_v - p_v) ** 2))
            ss_rw = float(np.sum((y_v - rw_forecast) ** 2))
            r2_vs_rw = 1 - ss_res / max(ss_rw, 1e-10)
        else:
            r2_vs_rw = 0.0

        self.model_stats = {
            "r2_oos": r2,
            "r2_vs_rw": r2_vs_rw,
            "rmse_oos": rmse,
            "mae_oos": mae,
            "n_obs": len(y_v),
            "n_features": len(self.feature_names),
            "avg_model_spread": float(np.mean(self.model_spread[oos_mask])),
        }

    def _compute_multi_lookback_signals(self) -> None:
        """Conformal z-scores and zone classifications for each lookback window.
        
        FIX: Uses shift(1) to prevent look-ahead bias.
        """
        r = self.residuals
        n = len(r)
        self.lookback_data = {}

        def _process_lookback(lb: int) -> tuple[int, dict] | None:
            if n < lb:
                return None
            min_periods = max(lb // 2, 5)

            # Native conformal prediction z-scores (fat-tail preservation)
            z_scores, lower_bounds, upper_bounds = compute_conformal_zscores(
                r, window=lb, min_periods=min_periods, alpha=0.05
            )

            zones = _classify_zones(z_scores)
            buy_signals, sell_signals = _detect_crossover_signals(z_scores)

            return lb, {
                "z_scores": z_scores,
                "zones": zones,
                "buy_signals": buy_signals,
                "sell_signals": sell_signals,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
            }

        for lb in LOOKBACK_WINDOWS:
            res = _process_lookback(lb)
            if res is not None:
                self.lookback_data[res[0]] = res[1]

        self.ts_data = pd.DataFrame({
            "Actual": self.y,
            "FairValue": self.predictions,
            "Residual": self.residuals,
            "ModelSpread": self.model_spread,
        })
        for lb, data in self.lookback_data.items():
            self.ts_data[f"Z_{lb}"] = data["z_scores"]
            self.ts_data[f"Zone_{lb}"] = data["zones"]
            self.ts_data[f"Buy_{lb}"] = data["buy_signals"]
            self.ts_data[f"Sell_{lb}"] = data["sell_signals"]

        self.lookback_data.clear()

    def _compute_breadth_metrics(self) -> None:
        """Aggregate zone/signal counts across lookback windows."""
        n = len(self.ts_data)
        valid_lookbacks = [lb for lb in LOOKBACK_WINDOWS if f"Z_{lb}" in self.ts_data.columns]
        num_lb = max(len(valid_lookbacks), 1)

        oversold = np.zeros(n)
        overbought = np.zeros(n)
        extreme_os = np.zeros(n)
        extreme_ob = np.zeros(n)
        buy_count = np.zeros(n)
        sell_count = np.zeros(n)

        z_scores_list = []

        for lb in valid_lookbacks:
            zones = self.ts_data[f"Zone_{lb}"].values
            z = self.ts_data[f"Z_{lb}"].values

            extreme_os += (zones == "Extreme Under")
            oversold += (zones == "Undervalued")
            extreme_ob += (zones == "Extreme Over")
            overbought += (zones == "Overvalued")
            
            buy_count += self.ts_data[f"Buy_{lb}"].values
            sell_count += self.ts_data[f"Sell_{lb}"].values

            z_scores_list.append(z)

        if z_scores_list:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                avg_z = np.nan_to_num(np.nanmean(np.vstack(z_scores_list), axis=0), nan=0.0)
        else:
            avg_z = np.zeros(n)

        self.ts_data["OversoldBreadth"] = (oversold + extreme_os) / num_lb * 100
        self.ts_data["OverboughtBreadth"] = (overbought + extreme_ob) / num_lb * 100
        self.ts_data["ExtremeOversold"] = extreme_os / num_lb * 100
        self.ts_data["ExtremeOverbought"] = extreme_ob / num_lb * 100
        self.ts_data["BuySignalBreadth"] = buy_count
        self.ts_data["SellSignalBreadth"] = sell_count
        self.ts_data["AvgZ"] = avg_z

        self.ts_data["ConvictionRaw"] = (
            (overbought - oversold) / num_lb * 100
            + (extreme_ob - extreme_os) / num_lb * 100 * 1.5
        )

    def _compute_ddm_conviction(self) -> None:
        """Drift-Diffusion filter with mean-reverting variance."""
        raw = self.ts_data["ConvictionRaw"].values
        filtered, _gains, variances = drift_diffusion_filter(raw)
        ddm_std = np.sqrt(np.maximum(variances, 0))

        self.ts_data["ConvictionScore"] = filtered
        bounded = _apply_conviction_bounds(filtered)
        self.ts_data["ConvictionBounded"] = bounded
        self.ts_data["ConvictionUpper"] = _apply_conviction_bounds(filtered + 1.96 * ddm_std)
        self.ts_data["ConvictionLower"] = _apply_conviction_bounds(filtered - 1.96 * ddm_std)

        regimes = []
        for score_bounded in bounded:
            if score_bounded < -CONVICTION_STRONG:
                regimes.append("STRONGLY OVERSOLD")
            elif score_bounded < -CONVICTION_WEAK:
                regimes.append("OVERSOLD")
            elif score_bounded > CONVICTION_STRONG:
                regimes.append("STRONGLY OVERBOUGHT")
            elif score_bounded > CONVICTION_WEAK:
                regimes.append("OVERBOUGHT")
            else:
                regimes.append("NEUTRAL")
        self.ts_data["Regime"] = regimes

    def _compute_divergences(self) -> None:
        """Swing-based divergence detection between target and residual."""
        n = len(self.ts_data)
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)

        order = 5
        if n < order * 3:
            self.ts_data["BullishDiv"] = bull_div
            self.ts_data["BearishDiv"] = bear_div
            return

        price = np.asarray(self.y)
        residual = np.asarray(self.residuals)

        last_low_idx = -1
        last_high_idx = -1

        expanding_std = pd.Series(residual).expanding(min_periods=min(20, max(2, len(residual) // 3))).std().bfill().values

        for i in range(order * 2, n):
            window_price = price[i - 2 * order : i + 1]

            if np.argmin(window_price) == order:
                curr_low = i - order
                if last_low_idx != -1 and price[curr_low] < price[last_low_idx] and residual[curr_low] > residual[last_low_idx]:
                    if residual[curr_low] < -expanding_std[curr_low] * 0.5:
                        bull_div[i] = True
                last_low_idx = curr_low

            if np.argmax(window_price) == order:
                curr_high = i - order
                if last_high_idx != -1 and price[curr_high] > price[last_high_idx] and residual[curr_high] < residual[last_high_idx]:
                    if residual[curr_high] > expanding_std[curr_high] * 0.5:
                        bear_div[i] = True
                last_high_idx = curr_high

        self.ts_data["BullishDiv"] = bull_div
        self.ts_data["BearishDiv"] = bear_div

    def _find_pivots(self, order: int = 5) -> None:
        """Identify pivot highs/lows in the residual series."""
        r = np.asarray(self.residuals)
        n = len(r)

        conf_tops = []
        conf_bottoms = []
        top_vals = []
        bottom_vals = []

        for i in range(order * 2, n):
            window = r[i - 2 * order : i + 1]
            if np.argmax(window) == order:
                conf_tops.append(i)
                top_vals.append(r[i - order])
            if np.argmin(window) == order:
                conf_bottoms.append(i)
                bottom_vals.append(r[i - order])

        self.pivots = {
            "tops": conf_tops,
            "bottoms": conf_bottoms,
            "avg_top": float(np.mean(top_vals)) if top_vals else float(np.mean(r) + np.std(r)),
            "avg_bottom": float(np.mean(bottom_vals)) if bottom_vals else float(np.mean(r) - np.std(r)),
        }

        self.ts_data["IsPivotTop"] = False
        self.ts_data["IsPivotBottom"] = False
        if conf_tops:
            self.ts_data.loc[conf_tops, "IsPivotTop"] = True
        if conf_bottoms:
            self.ts_data.loc[conf_bottoms, "IsPivotBottom"] = True

        self.residual_stats = {
            "mean": float(np.mean(r)),
            "std": float(np.std(r)),
            "current": float(r[-1]),
            "current_zscore": float((r[-1] - np.mean(r)) / max(np.std(r), 1e-8)),
            "percentile": float(stats.percentileofscore(r, r[-1])),
            "min": float(np.min(r)),
            "max": float(np.max(r)),
        }

    def _compute_forward_changes(self) -> None:
        """Forward % change with winsorization."""
        n = len(self.ts_data)
        y_arr = np.asarray(self.y)
        
        for period in (5, 10, 20):
            fwd = np.full(n, np.nan)
            y_curr = y_arr[:-period]
            y_fwd = y_arr[period:]

            valid = np.abs(y_curr) > 1e-10
            fwd[:-period][valid] = (y_fwd[valid] - y_curr[valid]) / np.abs(y_curr[valid]) * 100

            # Winsorize at ±100% to prevent extreme outliers
            fwd = np.clip(fwd, -100, 100)

            self.ts_data[f"FwdChg_{period}"] = fwd

    def _compute_ou_diagnostics(self) -> None:
        """OU estimation with rolling θ for uncertainty-aware projections."""
        r = self.residuals
        oos_r = r[MIN_TRAIN_SIZE:]

        if len(oos_r) > 30:
            theta, mu, sigma = ornstein_uhlenbeck_estimate(oos_r)
            
            try:
                adf_pvalue = float(adfuller(oos_r, autolag="AIC")[1])
            except Exception:
                adf_pvalue = 1.0

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_pvalue = float(kpss(oos_r, regression="c", nlags="auto")[1])
            except Exception:
                kpss_pvalue = 0.0

            # V3.1: Decoupled vol scaler
            vol_multiplier = 1.0
            dynamic_theta = theta
            
            # Rolling θ estimation for projection uncertainty
            window_size = min(60, len(oos_r) // 3)
            self.theta_history = []
            for i in range(window_size, len(oos_r)):
                theta_roll, _, _ = ornstein_uhlenbeck_estimate(oos_r[i - window_size : i])
                self.theta_history.append(theta_roll)
            
            theta_std = np.std(self.theta_history) if len(self.theta_history) > 1 else 0.0
            
            # Dynamic θ binds to trailing memory specifically
            dynamic_theta = self.theta_history[-1] if self.theta_history else theta
        else:
            theta, mu, sigma = 0.05, 0.0, max(float(np.std(r)), 1e-6)
            adf_pvalue = 1.0
            kpss_pvalue = 0.0
            vol_multiplier = 1.0
            dynamic_theta = theta
            theta_std = 0.0

        self.ou_params = {
            "theta": theta,
            "dynamic_theta": dynamic_theta,
            "mu": mu,
            "sigma": sigma,
            "half_life_base": np.log(2) / max(theta, 1e-4),
            "half_life": np.log(2) / max(dynamic_theta, 1e-4),
            "stationary_std": sigma / np.sqrt(2 * max(theta, 1e-4)),
            "adf_pvalue": adf_pvalue,
            "kpss_pvalue": kpss_pvalue,
            "vol_multiplier": vol_multiplier,
            "theta_std": theta_std,
        }

        current_r = float(r[-1])
        proj_days = np.arange(1, OU_PROJECTION_DAYS + 1)
        
        # Base projection
        self.ou_projection = mu + (current_r - mu) * np.exp(-dynamic_theta * proj_days)
        
        # Confidence bands from θ uncertainty
        if theta_std > 0:
            self.ou_projection_upper = mu + (current_r - mu) * np.exp(-(dynamic_theta - theta_std) * proj_days)
            self.ou_projection_lower = mu + (current_r - mu) * np.exp(-(dynamic_theta + theta_std) * proj_days)
        else:
            self.ou_projection_upper = self.ou_projection.copy()
            self.ou_projection_lower = self.ou_projection.copy()

    def _compute_hurst(self) -> None:
        """Hurst exponent via DFA on OOS residuals with strict stationarity guard."""
        oos_r = self.residuals[MIN_TRAIN_SIZE:]
        if len(oos_r) > 30:
            if self.ou_params.get("adf_pvalue", 1.0) > 0.05:
                # If absolute non-stationarity is detected, DFA scaling can falsely imply trends
                self.hurst = 0.5
            else:
                self.hurst = hurst_dfa(oos_r)
        else:
            self.hurst = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, max_entries=5)
def load_google_sheet(sheet_url: str) -> tuple[pd.DataFrame | None, str | None]:
    """Extract sheet ID and GID from a Google Sheets URL, fetch as CSV."""
    try:
        sheet_id_match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
        if not sheet_id_match:
            return None, "Invalid Google Sheets URL — could not extract sheet ID."
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r"gid=(\d+)", sheet_url)
        gid = gid_match.group(1) if gid_match else "0"
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)


def clean_data(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    date_col: str | None = None,
) -> pd.DataFrame:
    """Select, coerce, and clean numeric columns; optionally parse dates."""
    features = [f for f in features if f != target]
    cols = [target] + features
    if date_col and date_col in df.columns:
        cols.append(date_col)

    valid_cols = [c for c in cols if c in df.columns]
    if target not in valid_cols:
        return pd.DataFrame()

    data = df[valid_cols].copy()

    if date_col and date_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
            data = data.dropna(subset=[date_col]).sort_values(date_col)
        except Exception:
            pass

    for col in [target] + features:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    numeric_cols = [target] + features
    data[numeric_cols] = data[numeric_cols].ffill().bfill()
    data = data.dropna(subset=numeric_cols)

    return data.reset_index(drop=True)


def apply_chart_theme(fig: go.Figure) -> None:
    """Apply the Hemrek dark theme to any Plotly figure (mutates in place)."""
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=CHART_BG,
        paper_bgcolor=CHART_BG,
        font=dict(family="Inter", color=CHART_FONT_COLOR),
        margin=dict(t=40, l=20, r=20, b=20),
        hoverlabel=dict(bgcolor=CHART_BG, font_size=12),
    )
    fig.update_xaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_ZEROLINE)
    fig.update_yaxes(gridcolor=CHART_GRID, zerolinecolor=CHART_ZEROLINE)


# ══════════════════════════════════════════════════════════════════════════════
# UI RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _render_header() -> None:
    st.markdown("""
    <div class="premium-header">
        <h1>AARAMBH : Fair Value Breadth</h1>
        <div class="tagline">Walk-Forward Valuation · OU Mean-Reversion · DDM Conviction | Quantitative Reversal Analysis</div>
    </div>
    """, unsafe_allow_html=True)


def _render_landing_page() -> None:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='metric-card purple' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--purple); margin-bottom: 0.5rem;'>🎯 Walk-Forward Fair Value</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Expanding-window regression with strict out-of-sample validation.
                No look-ahead bias. Conformal z-scores preserve fat tails.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Ensemble:</strong> Ridge + Huber + ElasticNet + WLS<br>
                <strong>Validation:</strong> Walk-forward OOS + Structural Break Detection<br>
                <strong>Uncertainty:</strong> Model disagreement + Conformal bounds
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>📉 OU Mean-Reversion</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Andrews (1993) median-unbiased half-life estimation.
                Rolling θ for projection uncertainty bands.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Output:</strong> Half-life ± confidence<br>
                <strong>Projection:</strong> 90-day path with bands<br>
                <strong>Validation:</strong> DFA Hurst exponent (ADF-guarded)
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 0.5rem;'>📊 DDM Conviction</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Drift-Diffusion (SPRT) with mean-reverting variance.
                Evidence accumulation with 95% confidence bands.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Lookbacks:</strong> 5D, 10D, 20D, 50D, 100D<br>
                <strong>Smoothing:</strong> Leaky DDM + MR variance<br>
                <strong>Range:</strong> Soft-bounded ±100
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 Getting Started</h4>
        <p>Use the <strong>Sidebar</strong> to load data (CSV/Excel or Google Sheet).
        Select a <strong>Target</strong> and <strong>Predictors</strong>, then click <strong>Run Analysis</strong> to execute the walk-forward engine.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_footer() -> None:
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© {ist_now.year} {PRODUCT_NAME} | {COMPANY} | v{VERSION} | {current_time_ist}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB RENDERING FUNCTIONS (Phase III - Consolidated)
# ══════════════════════════════════════════════════════════════════════════════

def _render_metric_card(label: str, value: str, subtext: str = "", color_class: str = "neutral", inline: bool = False) -> None:
    """DRY helper for rendering metric cards consistently."""
    esc = html.escape
    card_class = f"metric-card {esc(color_class)}"
    if inline:
        card_style = 'style="min-height: 100px;"'
    else:
        card_style = 'style="min-height: 120px;"'
    
    st.markdown(
        f'<div class="{card_class}" {card_style}>'
        f"<h4>{esc(label)}</h4>"
        f"<h2>{esc(value)}</h2>"
        f'{"" if not subtext else f"<div class=\"sub-metric\">{esc(subtext)}</div>"}'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_primary_signal(signal, model_stats, regime_stats, ts) -> None:
    """Render Primary Signal card - always visible above tabs."""
    
    signal_class = (
        "undervalued" if signal["signal"] == "BUY"
        else "overvalued" if signal["signal"] == "SELL"
        else "fair"
    )
    signal_emoji = "🟢" if signal["signal"] == "BUY" else "🔴" if signal["signal"] == "SELL" else "🟡"

    # Determine divergence indicator
    if signal["has_bullish_div"]:
        div_badge = '<span class="status-badge buy">🔔 BULLISH DIVERGENCE</span>'
    elif signal["has_bearish_div"]:
        div_badge = '<span class="status-badge sell">🔔 BEARISH DIVERGENCE</span>'
    else:
        div_badge = ""

    # Generate layman explanation based on signal
    if signal["signal"] == "BUY":
        explanation = f"Market is trading below fair value across {signal['oversold_breadth']:.0f}% of lookback windows. Historical data suggests this is a buying opportunity."
    elif signal["signal"] == "SELL":
        explanation = f"Market is trading above fair value across {signal['overbought_breadth']:.0f}% of lookback windows. Historical data suggests this is a selling opportunity."
    else:
        explanation = f"Market is near fair value. Conviction score ({signal['conviction_score']:+.0f}) is in neutral zone (−20 to +20). No strong directional signal."

    st.markdown(
        f'<div class="signal-card {html.escape(signal_class)}" style="padding: 1.5rem;">'
        f'<div class="label">WALK-FORWARD SIGNAL</div>'
        f'<div class="value">{signal_emoji} {html.escape(signal["signal"])}</div>'
        f'<div class="subtext">'
        f'<strong>{html.escape(signal["strength"])}</strong> Strength • '
        f'<strong>{html.escape(signal["confidence"])}</strong> Confidence • '
        f'OU t½ = <strong>{signal["ou_half_life"]:.0f}d</strong>'
        f'{"" if not div_badge else " • " + div_badge}'
        f'</div>'
        f'<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color); font-size: 0.85rem; line-height: 1.5; color: var(--text-secondary);">'
        f'<strong>Why this signal?</strong> {html.escape(explanation)}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    
    st.markdown("---")


def _render_tab_dashboard_content(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target) -> None:
    """Dashboard tab content (excluding Primary Signal which is now above tabs)."""

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 1: BASE CONVICTION — Full width (the raw signal generator)
    # ══════════════════════════════════════════════════════════════════════════════

    st.markdown("##### Base Conviction Score")
    st.markdown(
        '<p style="color: #888; font-size: 0.85rem;">Raw breadth differential: Oversold% − Overbought% across all lookbacks. '
        'Negative = oversold bias, Positive = overbought bias.</p>',
        unsafe_allow_html=True,
    )

    if "ConvictionRaw" in ts_filtered.columns:
        fig_raw = go.Figure()

        # Area fills for visual clarity
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"].clip(lower=0),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.12)", line=dict(width=0), showlegend=False,
        ))
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"].clip(upper=0),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.12)", line=dict(width=0), showlegend=False,
        ))

        # Color-coded markers with size variation for threshold emphasis
        conv_colors = []
        marker_sizes = []
        for c in ts_filtered["ConvictionRaw"]:
            if c > 40:
                conv_colors.append(COLOR_RED)
                marker_sizes.append(10)  # Large for extreme overbought
            elif c >= 20:
                conv_colors.append("rgba(239, 68, 68, 0.75)")
                marker_sizes.append(8)  # Smaller size at threshold
            elif c < -40:
                conv_colors.append(COLOR_GREEN)
                marker_sizes.append(10)  # Large for extreme oversold
            elif c <= -20:
                conv_colors.append("rgba(16, 185, 129, 0.75)")
                marker_sizes.append(8)  # Smaller size at threshold
            else:
                conv_colors.append(COLOR_MUTED)
                marker_sizes.append(8)  # Medium for neutral zone

        # Main conviction line with markers
        fig_raw.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionRaw"], mode="lines+markers", name="Raw Conviction",
            line=dict(color=COLOR_GOLD, width=2.0),
            marker=dict(size=marker_sizes, color=conv_colors),
            hovertemplate="Date: %{x}<br>Conviction: %{y:.1f}<extra></extra>"
        ))

        # Reference lines at key thresholds
        fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
        fig_raw.add_hline(y=40, line_dash="dash", line_color="rgba(239,68,68,0.25)", line_width=1)
        fig_raw.add_hline(y=20, line_dash="dot", line_color="rgba(239,68,68,0.15)", line_width=1)
        fig_raw.add_hline(y=-20, line_dash="dot", line_color="rgba(16,185,129,0.15)", line_width=1)
        fig_raw.add_hline(y=-40, line_dash="dash", line_color="rgba(16,185,129,0.25)", line_width=1)

        fig_raw.update_layout(
            height=350,
            xaxis_title=None,
            yaxis_title="Conviction",
            yaxis=dict(range=[-100, 100], tickfont=dict(size=10)),
            margin=dict(t=10, l=60, r=20, b=10),
            showlegend=False,
        )
        apply_chart_theme(fig_raw)
        st.plotly_chart(fig_raw, )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 2: DDM CONVICTION — Full width (the smoothed, bounded signal)
    # ══════════════════════════════════════════════════════════════════════════════

    st.markdown("##### SPRT DDM Confidence Boundaries")
    st.markdown(
        '<p style="color: #888; font-size: 0.85rem;">Drift-Diffusion accumulation with mean-reverting variance.</p>',
        unsafe_allow_html=True,
    )

    fig_conv = go.Figure()

    if "ConvictionUpper" in ts_filtered.columns:
        # 95% confidence band
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionUpper"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig_conv.add_trace(go.Scatter(
            x=x_axis, y=ts_filtered["ConvictionLower"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(255,195,0,0.10)", name="95% Band",
        ))

    # Area fills for conviction zones
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(lower=0),
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)", line=dict(width=0), showlegend=False,
    ))
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"].clip(upper=0),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.12)", line=dict(width=0), showlegend=False,
    ))

    # DDM conviction line
    fig_conv.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["ConvictionScore"], mode="lines", name="DDM Conviction",
        line=dict(color=COLOR_GOLD, width=2.5),
    ))

    # Subtle zero reference line only
    fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)

    fig_conv.update_layout(
        height=350,
        xaxis_title=None,
        yaxis_title="Conviction",
        yaxis=dict(range=[-100, 100], tickfont=dict(size=10)),
        margin=dict(t=10, l=60, r=20, b=10),
        showlegend=False,
    )
    apply_chart_theme(fig_conv)
    st.plotly_chart(fig_conv, )

    # Interpretation Card (matching Regime Context style)
    conviction_val = signal["conviction_score"]
    conviction_upper = signal["conviction_upper"]
    conviction_lower = signal["conviction_lower"]
    
    if conviction_val > 40:
        regime_title = "STRONG OVERBOUGHT"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} indicates extreme overbought conditions. The accumulated evidence strongly favors mean-reversion lower. Consider reducing exposure or preparing short opportunities."
    elif conviction_val > 20:
        regime_title = "MODERATE OVERBOUGHT"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} suggests overbought bias. Evidence accumulation is positive but not at extreme levels. Monitor for potential weakening momentum."
    elif conviction_val > -20:
        regime_title = "NEUTRAL ZONE"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} indicates balanced evidence. The Drift-Diffusion model shows no strong directional bias. Wait for clearer signals before taking action."
    elif conviction_val > -40:
        regime_title = "MODERATE OVERSOLD"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} suggests oversold bias. Evidence accumulation is negative but not at extreme levels. Consider preparing for potential long opportunities."
    else:
        regime_title = "STRONG OVERSOLD"
        regime_desc = f"DDM conviction at {conviction_val:+.0f} indicates extreme oversold conditions. The accumulated evidence strongly favors mean-reversion higher. Consider building long positions."

    # Confidence band width analysis
    band_width = conviction_upper - conviction_lower
    if band_width < 30:
        confidence_note = f"Narrow band ({band_width:.0f} points) suggests high conviction in current reading."
    elif band_width > 60:
        confidence_note = f"Wide band ({band_width:.0f} points) suggests elevated uncertainty; weight signal less heavily."
    else:
        confidence_note = f"Moderate uncertainty range ({band_width:.0f} points)."

    st.markdown(f"""
    <div class="metric-card" style="padding: 1.25rem;">
        <h4 style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{html.escape(regime_title)}</h4>
        <p style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.7; margin: 0;">{html.escape(regime_desc)}</p>
        <p style="color: var(--text-muted); font-size: 0.75rem; line-height: 1.6; margin: 0.75rem 0 0 0; font-style: italic;">{html.escape(confidence_note)}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 4: MARKET STATE — Category: Regime & Breadth
    # ══════════════════════════════════════════════════════════════════════════════

    st.markdown("##### Market State")

    # Top row: 3 metric cards (Breadth + Regime)
    reg_col1, reg_col2, reg_col3 = st.columns(3)

    with reg_col1:
        _render_metric_card("OVERSOLD BREADTH", f'{signal["oversold_breadth"]:.0f}%', "Lookbacks in undervalued zones",
                           "success" if signal["oversold_breadth"] > 60 else "neutral")

    with reg_col2:
        _render_metric_card("OVERBOUGHT BREADTH", f'{signal["overbought_breadth"]:.0f}%', "Lookbacks in overvalued zones",
                           "danger" if signal["overbought_breadth"] > 60 else "neutral")

    with reg_col3:
        curr_regime = signal["regime"]
        regime_short = curr_regime.replace("STRONGLY ", "").replace("OVERSOLD", "OS").replace("OVERBOUGHT", "OB")
        regime_color = "success" if "OVERSOLD" in curr_regime else "danger" if "OVERBOUGHT" in curr_regime else "neutral"
        _render_metric_card("CURRENT REGIME", regime_short, f"OU t½ = {signal['ou_half_life']:.0f}d", regime_color)

    st.markdown("")

    # Bottom row: Regime Interpretation card (full width spanning all 3 columns)
    # Generate dynamic interpretation based on regime distribution
    total = len(ts)
    os_strong = regime_stats["strongly_oversold"]
    os_mod = regime_stats["oversold"]
    os_total = os_strong + os_mod
    ob_strong = regime_stats["strongly_overbought"]
    ob_mod = regime_stats["overbought"]
    ob_total = ob_strong + ob_mod
    neutral_count = regime_stats["neutral"]
    
    os_pct = os_total / total * 100
    ob_pct = ob_total / total * 100
    neutral_pct = neutral_count / total * 100
    
    if os_pct > 50:
        interp_title = "OVERSOLD DOMINANT"
        if os_strong > os_mod:
            intensity = f"Strong oversold signals ({os_strong} periods) outweigh moderate oversold ({os_mod} periods), indicating deep valuation dislocations."
        else:
            intensity = f"Moderate oversold signals ({os_mod} periods) dominate with {os_strong} strong oversold periods, suggesting sustained buying pressure."
        interp_text = (
            f"Market has spent {os_pct:.0f}% of the analyzed period in oversold conditions ({os_total} out of {total} observations). "
            f"{intensity} "
            f"Neutral regimes account for only {neutral_pct:.0f}% ({neutral_count} periods), while overbought conditions are rare at {ob_pct:.0f}% ({ob_total} periods). "
            f"This regime distribution historically correlates with favorable forward returns for patient capital deployment."
        )
    elif ob_pct > 50:
        interp_title = "OVERBOUGHT DOMINANT"
        if ob_strong > ob_mod:
            intensity = f"Strong overbought signals ({ob_strong} periods) outweigh moderate overbought ({ob_mod} periods), indicating elevated valuation risk."
        else:
            intensity = f"Moderate overbought signals ({ob_mod} periods) dominate with {ob_strong} strong overbought periods, suggesting stretched valuations."
        interp_text = (
            f"Market has spent {ob_pct:.0f}% of the analyzed period in overbought conditions ({ob_total} out of {total} observations). "
            f"{intensity} "
            f"Neutral regimes account for only {neutral_pct:.0f}% ({neutral_count} periods), while oversold conditions are rare at {os_pct:.0f}% ({os_total} periods). "
            f"This regime distribution historically correlates with elevated downside risk and favors defensive positioning or profit-taking."
        )
    else:
        interp_title = "BALANCED REGIME"
        interp_text = (
            f"Market has oscillated between regimes without establishing clear directional dominance. "
            f"Neutral conditions prevail at {neutral_pct:.0f}% ({neutral_count} periods), while oversold ({os_pct:.0f}%, {os_total} periods) and overbought ({ob_pct:.0f}%, {ob_total} periods) conditions are roughly balanced. "
            f"This mixed regime distribution suggests a range-bound market environment where stock selection and tactical entry points matter more than broad directional bets. "
            f"Current signal should be weighted against this broader regime context and confirmed with additional technical or fundamental catalysts."
        )
    
    st.markdown(f"""
    <div class="metric-card" style="padding: 1.25rem;">
        <h4 style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{interp_title}</h4>
        <p style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.7; margin: 0;">{interp_text}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 5: MODEL QUALITY — Category: Engine Diagnostics
    # ══════════════════════════════════════════════════════════════════════════════
    
    st.markdown("##### Model Quality")
    
    qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
    
    with qual_col1:
        r2_class = "success" if model_stats["r2_oos"] > 0.7 else "warning" if model_stats["r2_oos"] > 0.4 else "danger"
        _render_metric_card("OOS R²", f"{model_stats['r2_oos']:.3f}", "Walk-forward fit", r2_class)

    with qual_col2:
        r2_rw = model_stats.get("r2_vs_rw", 0.0)
        rw_class = "success" if r2_rw > 0.05 else "warning" if r2_rw > -0.05 else "danger"
        _render_metric_card("R² vs RW", f"{r2_rw:+.3f}", "Vs random walk", rw_class)

    with qual_col3:
        h = signal["hurst"]
        h_label = "MR" if h < 0.40 else "Trend" if h > 0.60 else "RW"
        h_class = "success" if h < 0.40 else "danger" if h > 0.60 else "neutral"
        _render_metric_card("DFA Hurst", f"{h:.2f}", h_label, h_class)

    with qual_col4:
        sp_class = "success" if signal["model_spread"] < 0.5 else "warning" if signal["model_spread"] < 1.5 else "danger"
        _render_metric_card("Model Spread", f"{signal['model_spread']:.2f}", "Ensemble std dev", sp_class)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════════
    # ROW 6: FAIR VALUE — Category: Valuation Output
    # ══════════════════════════════════════════════════════════════════════════════
    
    st.markdown("##### Actual vs Walk-Forward Fair Value")
    st.markdown('<p style="color: #888; font-size: 0.85rem;">Out-of-sample ensemble prediction with model uncertainty bands</p>', unsafe_allow_html=True)

    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["Actual"], mode="lines", name="Actual",
        line=dict(color=COLOR_GOLD, width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair Value (OOS)",
        line=dict(color=COLOR_CYAN, width=2, dash="dot"),
    ), row=1, col=1)

    if "ModelSpread" in ts_filtered.columns:
        upper = ts_filtered["FairValue"] + ts_filtered["ModelSpread"]
        lower = ts_filtered["FairValue"] - ts_filtered["ModelSpread"]
        fig.add_trace(go.Scatter(
            x=x_axis, y=upper, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_axis, y=lower, mode="lines", line=dict(width=0),
            fill="tonexty", fillcolor="rgba(6,182,212,0.08)",
            name="Model Uncertainty", hoverinfo="skip",
        ), row=1, col=1)

    bar_colors = [COLOR_GREEN if r < 0 else COLOR_RED for r in ts_filtered["Residual"]]
    fig.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["Residual"], name="Residual (OOS)",
        marker_color=bar_colors, showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=COLOR_GOLD, line_width=1, row=2, col=1)

    if hasattr(engine, "ou_projection") and len(engine.ou_projection) > 0 and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
        last_date = ts["Date"].iloc[-1]
        proj_dates = pd.bdate_range(start=last_date, periods=OU_PROJECTION_DAYS + 1)[1:]

        fig.add_trace(go.Scatter(
            x=proj_dates, y=engine.ou_projection,
            mode="lines", name="OU Projection",
            line=dict(color=COLOR_GOLD, width=1.5, dash="dot"), opacity=0.5,
        ), row=2, col=1)

        if len(engine.ou_projection_upper) > 0:
            fig.add_trace(go.Scatter(
                x=proj_dates, y=engine.ou_projection_upper,
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=proj_dates, y=engine.ou_projection_lower,
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(255,195,0,0.15)",
                name="θ Uncertainty", hoverinfo="skip",
            ), row=2, col=1)

    fig.update_layout(height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text=active_target, row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    apply_chart_theme(fig)
    st.plotly_chart(fig, )


def _render_tab_breadth(ts_filtered, x_axis, x_title) -> None:
    """Breadth Topology: Conformal zones and extreme distributions."""

    st.markdown("##### Market Breadth")
    st.markdown(
        '<p style="color: #888; font-size: 0.85rem;">Oversold/overbought breadth across lookback windows</p>',
        unsafe_allow_html=True,
    )

    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OversoldBreadth"],
        fill="tozeroy", fillcolor="rgba(16,185,129,0.15)",
        line=dict(color=COLOR_GREEN, width=2), name="Oversold",
    ))
    fig_zones.add_trace(go.Scatter(
        x=x_axis, y=ts_filtered["OverboughtBreadth"],
        fill="tozeroy", fillcolor="rgba(239,68,68,0.15)",
        line=dict(color=COLOR_RED, width=2), name="Overbought",
    ))
    fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(255,195,0,0.2)", line_width=1)
    fig_zones.update_layout(
        height=350,
        xaxis_title=None,
        yaxis_title="Breadth %",
        yaxis=dict(range=[0, 100], tickfont=dict(size=10)),
        margin=dict(t=10, l=60, r=20, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
    )
    apply_chart_theme(fig_zones)
    st.plotly_chart(fig_zones, )

    st.markdown("---")
    st.markdown("##### Signal Frequency")
    st.markdown(
        '<p style="color: #888; font-size: 0.85rem;">Z-score threshold crossovers (±1σ)</p>',
        unsafe_allow_html=True,
    )

    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=ts_filtered["BuySignalBreadth"], name="Buy",
        marker=dict(color=COLOR_GREEN, opacity=0.8),
    ))
    fig_signals.add_trace(go.Bar(
        x=x_axis, y=-ts_filtered["SellSignalBreadth"], name="Sell",
        marker=dict(color=COLOR_RED, opacity=0.8),
    ))
    fig_signals.update_layout(
        height=300,
        xaxis_title=None,
        yaxis_title="Count",
        barmode="relative",
        margin=dict(t=10, l=60, r=20, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
    )
    apply_chart_theme(fig_signals)
    st.plotly_chart(fig_signals, )

    st.markdown("---")
    st.markdown("##### Average Z-Score")
    st.markdown(
        '<p style="color: #888; font-size: 0.85rem;">Multi-lookback z-score composite</p>',
        unsafe_allow_html=True,
    )

    fig_z = go.Figure()
    bar_colors = [COLOR_GREEN if z < -1 else COLOR_RED if z > 1 else COLOR_MUTED for z in ts_filtered["AvgZ"]]
    fig_z.add_trace(go.Bar(x=x_axis, y=ts_filtered["AvgZ"], marker_color=bar_colors, name="Avg Z", opacity=0.8))
    fig_z.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig_z.add_hline(y=2, line_dash="dash", line_color="rgba(239,68,68,0.25)", line_width=1)
    fig_z.add_hline(y=-2, line_dash="dash", line_color="rgba(16,185,129,0.25)", line_width=1)
    fig_z.update_layout(
        height=300,
        xaxis_title=None,
        yaxis_title="Z-Score",
        margin=dict(t=10, l=60, r=20, b=10),
        showlegend=False,
    )
    apply_chart_theme(fig_z)
    st.plotly_chart(fig_z, )

    st.markdown("---")
    st.markdown("##### Current Lookback States")

    for lb in LOOKBACK_WINDOWS:
        if f"Z_{lb}" not in ts_filtered.columns:
            continue
        z = ts_filtered[f"Z_{lb}"].iloc[-1]
        zone = ts_filtered[f"Zone_{lb}"].iloc[-1]
        zone_color = COLOR_GREEN if "Under" in zone else COLOR_RED if "Over" in zone else COLOR_MUTED
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #2A2A2A;">
            <span style="color: #888; font-size: 0.85rem;">{lb}-Day</span>
            <span style="color: {zone_color}; font-weight: 600; font-size: 0.85rem;">{zone} ({z:+.2f})</span>
        </div>
        """, unsafe_allow_html=True)


def _render_tab_data(ts_filtered, ts, active_target) -> None:
    """Data Table: Time series data with export functionality."""
    st.markdown(f"##### Time Series Data ({len(ts_filtered)} observations)")

    display_cols = [
        "Date", "Actual", "FairValue", "Residual", "ModelSpread", "AvgZ",
        "OversoldBreadth", "OverboughtBreadth", "ConvictionScore", "Regime",
        "BullishDiv", "BearishDiv",
    ]
    display_cols = [c for c in display_cols if c in ts_filtered.columns]

    display_df = ts_filtered[display_cols].copy()
    rounding = {
        "AvgZ": 3, "ModelSpread": 3, "FairValue": 2,
        "Residual": 1, "ConvictionScore": 1, "OversoldBreadth": 1, "OverboughtBreadth": 1,
    }
    for col, decimals in rounding.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].round(decimals)

    if "BullishDiv" in display_df.columns:
        display_df["BullishDiv"] = display_df["BullishDiv"].apply(lambda x: "🟢" if x else "")
    if "BearishDiv" in display_df.columns:
        display_df["BearishDiv"] = display_df["BearishDiv"].apply(lambda x: "🔴" if x else "")

    st.dataframe(display_df, hide_index=True, height=500)

    csv_data = ts.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Full CSV", csv_data,
        f"aarambh_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
    )


def _render_tab_ml_diagnostics(engine, ts_filtered, x_axis, x_title, signal, model_stats) -> None:
    """ML Diagnostics: Ensemble weight behavior, PCA mappings, and feature-impact tracking."""

    st.markdown("##### OU Mean-Reversion Diagnostics")
    
    theta_status = "✅ Stable" if signal.get("theta_stable", True) else "⚠️ Unstable"
    stationarity = "Stationary ✅" if signal['adf_pvalue'] < 0.05 and signal['kpss_pvalue'] > 0.05 else "Non-Stationary ⚠️"

    ou_col1, ou_col2, ou_col3 = st.columns(3)
    
    with ou_col1:
        _render_metric_card("OU Half-Life", f"{signal['ou_half_life']:.0f}d", "Andrews MU estimator", "primary")
    
    with ou_col2:
        adf_class = "success" if signal['adf_pvalue'] < 0.05 else "danger"
        _render_metric_card("ADF p-value", f"{signal['adf_pvalue']:.3f}", "Unit root test", adf_class)
    
    with ou_col3:
        kpss_class = "success" if signal['kpss_pvalue'] > 0.05 else "danger"
        _render_metric_card("KPSS p-value", f"{signal['kpss_pvalue']:.3f}", "Stationarity test", kpss_class)

    st.markdown("")
    st.markdown(f"**Stationarity:** {stationarity} | **θ Stability:** {theta_status}")

    st.markdown("---")
    st.markdown("##### Feature Impact")
    st.markdown('<p style="color: #888; font-size: 0.85rem;">Current predictor contributions to fair value estimation</p>', unsafe_allow_html=True)

    feature_history = engine.get_feature_impact_history()
    if not feature_history.empty:
        # Show latest feature impacts as horizontal bar chart
        if hasattr(engine, "latest_feature_impacts") and engine.latest_feature_impacts:
            impacts = engine.latest_feature_impacts
            labels = list(impacts.keys())[::-1]
            vals = list(impacts.values())[::-1]

            # Create gradient colors based on impact magnitude
            colors = []
            max_val = max(vals) if vals else 1
            for v in vals:
                intensity = v / max_val
                r = int(6 + (255 - 6) * intensity)
                g = int(182 + (212 - 182) * intensity)
                b = int(212 + (182 - 212) * intensity)
                colors.append(f"rgba({r},{g},{b},0.8)")

            fig_imp = go.Figure(go.Bar(
                x=vals, y=labels, orientation="h",
                marker=dict(color=colors),
                hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
            ))
            fig_imp.update_layout(
                height=max(280, len(labels) * 32),
                xaxis_title="Contribution %",
                xaxis=dict(tickfont=dict(size=9), zeroline=True, zerolinecolor="rgba(255,255,255,0.1)"),
                yaxis=dict(tickfont=dict(size=9)),
                margin=dict(t=10, l=10, r=20, b=10),
                showlegend=False,
            )
            apply_chart_theme(fig_imp)
            st.plotly_chart(fig_imp, )

        # Feature impact history table (only show if data exists)
        if not feature_history.empty and len(feature_history) > 0:
            st.markdown("###### Impact History")
            st.dataframe(feature_history.tail(10), hide_index=True, height=200)
    else:
        st.info("Feature impact data not available for current configuration.")

    st.markdown("---")
    st.markdown("##### Signal Performance (with Significance)")
    st.markdown('<p style="color: #888;">Hit rates and t-statistics for conviction-based signals</p>', unsafe_allow_html=True)

    perf = engine.get_signal_performance()
    perf_rows = []
    for period in (5, 10, 20):
        p = perf[period]
        buy_sig = "✅" if p["buy_p_value"] < 0.05 else "⚠️" if p["buy_p_value"] < 0.10 else ""
        sell_sig = "✅" if p["sell_p_value"] < 0.05 else "⚠️" if p["sell_p_value"] < 0.10 else ""

        perf_rows.append({
            "Holding Period": f"{period} Days",
            "Buy Hit Rate": f"{p['buy_hit'] * 100:.1f}%" if p["buy_count"] > 0 else "N/A",
            "Buy Avg Fwd Chg": f"{p['buy_avg']:.2f}%" if p["buy_count"] > 0 else "N/A",
            "Buy t-stat": f"{p['buy_t_stat']:.2f} {buy_sig}" if p["buy_count"] > 0 else "N/A",
            "Buy Count": p["buy_count"],
            "Sell Hit Rate": f"{p['sell_hit'] * 100:.1f}%" if p["sell_count"] > 0 else "N/A",
            "Sell Avg Fwd Chg": f"{p['sell_avg']:.2f}%" if p["sell_count"] > 0 else "N/A",
            "Sell t-stat": f"{p['sell_t_stat']:.2f} {sell_sig}" if p["sell_count"] > 0 else "N/A",
            "Sell Count": p["sell_count"],
        })

    st.dataframe(pd.DataFrame(perf_rows), hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Sidebar: Data Source ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">AARAMBH</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">आरंभ | Fair Value Breadth</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">📁 Data Source</div>', unsafe_allow_html=True)
        data_source = st.radio(
            "Source", ["📊 Google Sheets", "📤 Upload"],
            horizontal=True, label_visibility="collapsed",
            index=0  # Default to Google Sheets
        )

        df = None

        # State tracking for button text
        has_data = "data" in st.session_state and "run_analysis" in st.session_state

        if data_source == "📤 Upload":
            uploaded_file = st.file_uploader("CSV/Excel", type=["csv", "xlsx"], label_visibility="collapsed")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return

                # Show Run Analysis button for uploaded file
                if not has_data:
                    if st.button("🚀 Run Analysis", type="primary", ):
                        st.session_state.pop("engine", None)
                        st.session_state.pop("engine_cache", None)
                        st.session_state["data"] = df
                        st.session_state["run_analysis"] = True
                        st.rerun()
            else:
                st.info("📤 Upload a CSV or Excel file to begin analysis")
        else:
            sheet_url = st.text_input("Sheet URL", value=DEFAULT_SHEET_URL, label_visibility="collapsed")

            # Show Run Analysis button only if no data loaded yet
            if not has_data:
                if st.button("🚀 Run Analysis", type="primary", ):
                    with st.spinner("Loading data and running analysis..."):
                        df, error = load_google_sheet(sheet_url)
                        if error:
                            st.error(f"Failed: {error}")
                            return
                        # Store data and trigger engine run
                        st.session_state.pop("engine", None)
                        st.session_state.pop("engine_cache", None)
                        st.session_state["data"] = df
                        st.session_state["run_analysis"] = True
                        st.rerun()

            # Check if data already loaded and analysis complete
            if "data" in st.session_state and "run_analysis" in st.session_state:
                df = st.session_state["data"]

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Landing page when no data loaded ──────────────────────────────────
    if df is None:
        _render_header()
        _render_landing_page()
        _render_footer()
        return

    # ── Sidebar: Model Configuration ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">🧠 Model Configuration</div>', unsafe_allow_html=True)

        default_target = "NIFTY50_PE" if "NIFTY50_PE" in numeric_cols else numeric_cols[0]
        active_target_state = st.session_state.get("active_target", default_target)
        if active_target_state not in numeric_cols:
            active_target_state = numeric_cols[0]

        target_col = st.selectbox(
            "Target Variable", numeric_cols,
            index=numeric_cols.index(active_target_state),
        )

        date_candidates = [c for c in all_cols if "date" in c.lower()]
        default_date = date_candidates[0] if date_candidates else "None"
        active_date_state = st.session_state.get("active_date_col", default_date)
        if active_date_state not in ["None"] + all_cols:
            active_date_state = "None"

        date_col = st.selectbox(
            "Date Column", ["None"] + all_cols,
            index=(["None"] + all_cols).index(active_date_state),
        )

        available = [c for c in numeric_cols if c != target_col]
        valid_defaults = [p for p in DEFAULT_PREDICTORS if p in available]

        if "active_features" not in st.session_state:
            st.session_state["active_features"] = tuple(valid_defaults or available[:3])

        with st.expander("📊 Predictor Columns", expanded=False):
            st.caption("Select predictors, then click Apply to recompute.")

            staging_features = st.multiselect(
                "Predictor Columns", options=available,
                default=[f for f in st.session_state["active_features"] if f in available],
                label_visibility="collapsed",
                help="These columns are used as predictors for walk-forward fair value regression.",
            )

            if not staging_features:
                st.warning("⚠️ Select at least one predictor.")
                staging_features = [f for f in st.session_state["active_features"] if f in available]
                if not staging_features:
                    staging_features = available[:3]

            staging_set = set(staging_features)
            active_set = set(st.session_state["active_features"])
            has_pred_changes = staging_set != active_set
            has_other_changes = (target_col != active_target_state) or (date_col != active_date_state)
            has_changes = has_pred_changes or has_other_changes

            if has_pred_changes:
                added = staging_set - active_set
                removed = active_set - staging_set
                parts = []
                if added:
                    parts.append(f"+{len(added)} added")
                if removed:
                    parts.append(f"−{len(removed)} removed")
                st.caption(f"Pending: {', '.join(parts)}")
            elif has_other_changes:
                st.caption("Pending: Target/Date changes")

            apply_clicked = st.button(
                "✅ Apply Configuration" if has_changes else "No changes",
                disabled=not has_changes,
                type="primary" if has_changes else "secondary",
            )

            if apply_clicked and has_changes:
                st.session_state["active_target"] = target_col
                st.session_state["active_features"] = tuple(staging_features)
                st.session_state["active_date_col"] = date_col
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.rerun()

            active_count = len(st.session_state["active_features"])
            total_count = len(available)
            if active_count != total_count:
                st.info(f"Active: {active_count}/{total_count} predictors")

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Show "Reset Analysis" button after analysis is complete
        if "run_analysis" in st.session_state and st.session_state.get("run_analysis"):
            if st.button("🔄 Reset Analysis", type="secondary", ):
                st.session_state.pop("data", None)
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state.pop("run_analysis", None)
                st.rerun()
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # System Information Box (Collapsed by default)
        with st.expander("⚙️ System Information", expanded=False):
            st.markdown(f"""
            <div style='padding: 0.5rem 0;'>
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem;'>
                    <div>
                        <div style='font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;'>Version</div>
                        <div style='font-size: 0.8rem; color: var(--text-primary); font-weight: 600;'>{VERSION}</div>
                    </div>
                    <div>
                        <div style='font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.25rem;'>Status</div>
                        <div style='font-size: 0.8rem; color: var(--success-green); font-weight: 600;'>● Production</div>
                    </div>
                </div>
                <div style='margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border-color);'>
                    <div style='font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;'>Core Components</div>
                    <div style='display: flex; flex-wrap: wrap; gap: 0.5rem;'>
                        <span style='font-size: 0.7rem; color: var(--text-secondary); background: rgba(255,195,0,0.1); padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid rgba(255,195,0,0.2);'>Walk-Forward</span>
                        <span style='font-size: 0.7rem; color: var(--text-secondary); background: rgba(6,182,212,0.1); padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid rgba(6,182,212,0.2);'>OU (Andrews MU)</span>
                        <span style='font-size: 0.7rem; color: var(--text-secondary); background: rgba(255,195,0,0.1); padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid rgba(255,195,0,0.2);'>DDM (MR Var)</span>
                        <span style='font-size: 0.7rem; color: var(--text-secondary); background: rgba(136,136,136,0.1); padding: 0.25rem 0.5rem; border-radius: 4px; border: 1px solid rgba(136,136,136,0.2);'>DFA Hurst</span>
                    </div>
                </div>
                <div style='margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid var(--border-color);'>
                    <div style='font-size: 0.65rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 0.5rem;'>Configuration</div>
                    <div style='font-size: 0.75rem; color: var(--text-secondary); line-height: 1.6;'>
                        <div style='margin-bottom: 0.25rem;'><strong style='color: var(--text-muted);'>Lookbacks:</strong> {', '.join(f'{lb}D' for lb in LOOKBACK_WINDOWS)}</div>
                        <div><strong style='color: var(--text-muted);'>Break Detection:</strong> {'<span style="color: var(--success-green);">● Bai-Perron Active</span>' if _HAS_STATSMODELS else '<span style="color: var(--warning-amber);">⚠ statsmodels missing</span>'}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Resolve active configuration ──────────────────────────────────────
    active_target = st.session_state.get("active_target", target_col)
    active_features = list(st.session_state.get("active_features", staging_features))
    active_date = st.session_state.get("active_date_col", date_col)

    # ── Header (Only on Landing Page) ────────────────────────────────────
    if df is None:
        _render_header()

    # ── Data staleness warning ────────────────────────────────────────────
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors="coerce").dropna()
            if len(dates) > 0:
                latest_date = dates.max().to_pydatetime()
                if latest_date.tzinfo is not None:
                    latest_date = latest_date.replace(tzinfo=None)
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                data_age = (now_utc - latest_date).days
                if data_age > STALENESS_DAYS:
                    st.markdown(f"""
                    <div style="background: rgba(239,68,68,0.1); border: 1px solid {COLOR_RED}; border-radius: 10px;
                                padding: 0.75rem 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 1.4rem;">⚠️</span>
                        <div>
                            <span style="color: {COLOR_RED}; font-weight: 700;">Stale Data</span>
                            <span style="color: #888; font-size: 0.85rem;"> — Last data point is <b>{latest_date.strftime('%d %b %Y')}</b> ({data_age} days ago). Update your data source.</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        except Exception:
            pass

    # ── Clean & Fit Engine ────────────────────────────────────────────────
    data = clean_data(df, active_target, active_features, active_date if active_date != "None" else None)

    if len(data) < MIN_DATA_POINTS:
        st.error(f"Need {MIN_DATA_POINTS}+ data points for walk-forward analysis.")
        return

    active_features = [f for f in active_features if f in data.columns]
    if not active_features:
        st.error("No valid features found after data cleaning.")
        return

    X = data[active_features].values
    y = data[active_target].values

    if active_date != "None" and active_date in data.columns:
        latest_sig = str(data[active_date].max())
    else:
        latest_sig = str(pd.util.hash_pandas_object(data).sum())
    cache_key = f"{active_target}|{'|'.join(sorted(active_features))}|{len(data)}|{latest_sig}"

    if st.session_state.get("engine_cache") != cache_key:
        # Cache miss — run engine
        logger.start("AARAMBH UI — Data Processing")
        logger.checkpoint("Configuration", f"Target: {active_target} | Features: {len(active_features)} | Samples: {len(data)}")
        
        with st.spinner("Preparing walk-forward engine..."):
            if "engine" in st.session_state:
                del st.session_state["engine"]
            progress_bar = st.progress(0, text="Initializing engine...")
            logger.checkpoint("Engine Initialization", "FairValueEngine instance created")
            
            engine = FairValueEngine()
            engine.fit(X, y, feature_names=active_features, progress_callback=progress_bar.progress)
            
            st.session_state["engine"] = engine
            st.session_state["engine_cache"] = cache_key
            progress_bar.empty()
            
        logger.complete("Engine cached for session")
    else:
        # Cache hit — skip engine run
        logger.start("AARAMBH UI — Cached Results")
        logger.checkpoint("Cache Hit", f"Using cached engine for configuration: {cache_key[:50]}...")

    engine: FairValueEngine = st.session_state["engine"]
    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    regime_stats = engine.get_regime_stats()
    ts = engine.ts_data.copy()

    if active_date != "None" and active_date in data.columns:
        ts["Date"] = pd.to_datetime(data[active_date].values)
    else:
        ts["Date"] = np.arange(len(ts))

    logger.checkpoint("UI Rendering", "Dashboard tabs and visualizations")

    # ── Primary Signal (Above Tabs, Always Visible) ───────────────────────
    _render_primary_signal(signal, model_stats, regime_stats, ts)

    # ── Timeframe Filter (Swing-style button row) ─────────────────────────
    # Initialize session state for timeframe selection
    if 'tf_selected' not in st.session_state:
        st.session_state.tf_selected = '1Y'
    
    TIMEFRAMES = {
        '1M': 21,
        '6M': 126,
        '1Y': 252,
        '2Y': 504,
        'ALL': None
    }

    # Custom CSS for full-width timeframe buttons
    st.markdown("""
    <style>
        div[data-testid="stHorizontalBlock"] button[kind="secondary"],
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            width: 100% !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Create button row with equal-width columns
    tf_cols = st.columns(len(TIMEFRAMES))

    for i, tf in enumerate(TIMEFRAMES.keys()):
        with tf_cols[i]:
            # Determine button type based on selection
            btn_type = "primary" if st.session_state.tf_selected == tf else "secondary"

            # Create button
            if st.button(tf, key=f"tf_{tf}", type=btn_type):
                st.session_state.tf_selected = tf
                st.rerun()
    
    # Get selected timeframe
    selected_tf = st.session_state.tf_selected

    ts_filtered = ts.copy()
    if selected_tf != "ALL":
        if active_date != "None" and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
            max_date = ts["Date"].max()
            offsets = {
                "1M": pd.DateOffset(months=1), "6M": pd.DateOffset(months=6),
                "1Y": pd.DateOffset(years=1), "2Y": pd.DateOffset(years=2),
            }
            cutoff = max_date - offsets.get(selected_tf, pd.DateOffset(years=1))
            ts_filtered = ts[ts["Date"] >= cutoff]
        else:
            n_days = TIMEFRAME_TRADING_DAYS.get(selected_tf, 252)
            ts_filtered = ts.iloc[max(0, len(ts) - n_days) :]

    x_axis = ts_filtered["Date"]
    x_title = "Date" if active_date != "None" else "Index"

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_dashboard, tab_breadth, tab_ml, tab_data = st.tabs([
        "**📊 Dashboard**",
        "**🗺️ Breadth Topology**",
        "**🧠 ML Diagnostics**",
        "**📋 Data Table**",
    ])

    with tab_dashboard:
        _render_tab_dashboard_content(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target)

    with tab_breadth:
        _render_tab_breadth(ts_filtered, x_axis, x_title)

    with tab_ml:
        _render_tab_ml_diagnostics(engine, ts_filtered, x_axis, x_title, signal, model_stats)

    with tab_data:
        _render_tab_data(ts_filtered, ts, active_target)

    _render_footer()


if __name__ == "__main__":
    main()
