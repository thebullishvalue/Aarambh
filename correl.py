# -*- coding: utf-8 -*-
"""
AARAMBH (आरंभ) — Fair Value Breadth · Streamlit app
A @thebullishvalue Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Question: is the target (default NIFTY50 PE) above or below the value its
predictors justify, by an unusual amount, across several lookbacks at once?

Pipeline (aarambh/engine.py):
    1. Walk-forward fair value   ensemble regression, refit every 5 rows on past data only
    2. Valuation gap             actual − fair value (out of sample)
    3. Calibrated z-scores       gap vs its trailing 5/10/20/50/100-row median, ranked
                                 against its own history so every lookback means the same
    4. Breadth → conviction      net % of lookbacks over- vs undervalued, smoothed by a
                                 leaky accumulator and soft-bounded to ±100
    5. Signal                    BUY / SELL / HOLD in WEAK / MODERATE / STRONG tiers
    6. Diagnostics & validation  OU half-life, DFA α, ADF/KPSS, and a HAC-tested record of
                                 how each signal has done over the next 5/10/20 rows

This file is only the UI. Engine, data handling and maths live in the ``aarambh`` package.
"""

from __future__ import annotations

import html
import logging
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Must be the first Streamlit command.
st.set_page_config(
    page_title="AARAMBH | Fair Value Breadth",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from aarambh.config import (  # noqa: E402  (after set_page_config by design)
    COMPANY, CONVICTION_MODERATE, CONVICTION_STRONG, CONVICTION_WEAK, DEFAULT_MODEL_MEMORY,
    DEFAULT_PREDICTORS, DEFAULT_TARGET, HEADLINE_HORIZON, LOOKBACK_WINDOWS, MIN_DATA_POINTS,
    MODEL_MEMORY, OU_PROJECTION_DAYS, PRODUCT_NAME, STALENESS_DAYS, TIMEFRAME_TRADING_DAYS,
    VERSION, Z_EXTREME, Z_THRESHOLD,
)
from aarambh.data import clean_data, load_sheet, read_upload  # noqa: E402
from aarambh.engine import HAS_STATSMODELS, FairValueEngine, classify_conviction  # noqa: E402

# ── Logging: engine progress to the terminal ─────────────────────────────────
log = logging.getLogger("aarambh")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False


def _default_sheet_url() -> str:
    """GOOGLE_SHEET_URL from the environment, else Streamlit secrets.

    Checks for a secrets file first: touching ``st.secrets`` without one renders an
    error box in the page.
    """
    env = os.environ.get("GOOGLE_SHEET_URL", "").strip()
    if env:
        return env
    try:
        if st.secrets.load_if_toml_exists():
            return str(st.secrets.get("GOOGLE_SHEET_URL", "") or "").strip()
    except Exception as exc:  # malformed secrets.toml
        log.warning("Could not read Streamlit secrets: %s", exc)
    return ""


DEFAULT_SHEET_URL = _default_sheet_url()

MEMORY_LABELS = {
    "adaptive": "Adaptive — last 500 rows, 1-year decay",
    "long": "Long — all history, equal weight",
}

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
SERIES_COLORS = (COLOR_GOLD, COLOR_CYAN, COLOR_GREEN, COLOR_RED, "#8b5cf6", COLOR_AMBER)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

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
# DATA LOADING (cached) & FORMATTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
def _fetch_sheet(sheet_url: str) -> pd.DataFrame:
    df, error = load_sheet(sheet_url)
    if error:
        raise RuntimeError(error)  # exceptions are not cached, so a fix takes effect at once
    return df


def load_google_sheet(sheet_url: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        return _fetch_sheet(sheet_url), None
    except RuntimeError as exc:
        return None, str(exc)


def _num(x, spec: str = ".2f", na: str = "—") -> str:
    try:
        return format(x, spec) if x is not None and np.isfinite(x) else na
    except (TypeError, ValueError):
        return na


def _edge_class(t_stat: float) -> str:
    """Colour for an edge t-stat: works (t ≥ 2), unproven (|t| < 2), works in reverse (t ≤ −2)."""
    if not np.isfinite(t_stat):
        return "neutral"
    return "success" if t_stat >= 2 else "danger" if t_stat <= -2 else "warning"


def _edge_verdict(t_stat: float) -> str:
    if not np.isfinite(t_stat):
        return "Not enough history to judge."
    if t_stat >= 2:
        return "Historically reliable (HAC t ≥ 2)."
    if t_stat <= -2:
        return "Historically it has worked in reverse (HAC t ≤ −2): moves went against it."
    return "Not distinguishable from noise (|HAC t| < 2)."


def _price_sharing_predictors(target: str, features: list[str]) -> list[str]:
    """Predictors that are another valuation multiple of the same index as the target.

    E.g. NIFTY50_PB and NIFTY50_DY share NIFTY50_PE's price, so the fair value partly
    follows the price it is meant to judge.
    """
    multiples = {"PE", "PB", "DY", "EY", "PS"}
    if "_" not in target or target.rsplit("_", 1)[1].upper() not in multiples:
        return []
    prefix = target.rsplit("_", 1)[0]
    return [f for f in features if "_" in f and f.rsplit("_", 1)[0] == prefix
            and f.rsplit("_", 1)[1].upper() in multiples and f != target]


def apply_chart_theme(fig: go.Figure) -> None:
    """Apply the @thebullishvalue dark theme to any Plotly figure (mutates in place)."""
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


def _section(title: str, caption: str = "") -> None:
    st.markdown(f"##### {title}")
    if caption:
        st.markdown(f'<p style="color: #888; font-size: 0.85rem;">{caption}</p>', unsafe_allow_html=True)


def _text_card(title: str, body: str, note: str = "") -> None:
    note_html = (
        f'<p style="color: var(--text-muted); font-size: 0.75rem; line-height: 1.6; margin: 0.75rem 0 0 0; '
        f'font-style: italic;">{html.escape(note)}</p>' if note else ""
    )
    st.markdown(
        f'<div class="metric-card" style="padding: 1.25rem;">'
        f'<h4 style="color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.75rem; font-weight: 600; '
        f'text-transform: uppercase; letter-spacing: 1px;">{html.escape(title)}</h4>'
        f'<p style="color: var(--text-secondary); font-size: 0.85rem; line-height: 1.7; margin: 0;">{html.escape(body)}</p>'
        f'{note_html}</div>',
        unsafe_allow_html=True,
    )


def _history_dates(ts: pd.DataFrame, index: pd.Series):
    """Map engine row indices (in feature/θ histories) to the display x-axis."""
    idx = index.clip(0, len(ts) - 1).astype(int)
    return ts["Date"].iloc[idx].to_numpy()


# ══════════════════════════════════════════════════════════════════════════════
# UI RENDERING
# ══════════════════════════════════════════════════════════════════════════════

def _render_header() -> None:
    st.markdown("""
    <div class="premium-header">
        <h1>AARAMBH : Fair Value Breadth</h1>
        <div class="tagline">Walk-Forward Valuation · Calibrated Multi-Lookback Gaps · Conviction with a Measured Track Record</div>
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
                Every fair value is predicted by models fitted only on earlier rows.
                The valuation gap is actual minus fair value, out of sample.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Ensemble:</strong> Ridge + Huber + ElasticNet + PCA→WLS<br>
                <strong>Weights:</strong> Inverse of each model's out-of-sample error<br>
                <strong>Memory:</strong> Adaptive (500 rows, 1-yr decay) or Long (all history)
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>📏 Calibrated Breadth</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                The gap is scored against its trailing 5/10/20/50/100-row range, then ranked
                against its own past so |z| &gt; 1 means the same thing in every lookback.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Zones:</strong> |z| &gt; 1 over/under · |z| &gt; 2 extreme<br>
                <strong>Conviction:</strong> Net % of lookbacks, leaky-accumulated<br>
                <strong>Range:</strong> Soft-bounded ±100
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 0.5rem;'>🧪 Measured, Not Assumed</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Every signal comes with its own track record: what the target did over the
                next 5/10/20 rows after past signals, with overlap-robust t-stats.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Gap dynamics:</strong> OU half-life, DFA α, ADF/KPSS<br>
                <strong>Edge:</strong> Newey-West (HAC) t-statistics<br>
                <strong>Drivers:</strong> Ensemble coefficients over time
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 Getting Started</h4>
        <p>Open the <strong>Sidebar</strong>, load a Google Sheet or a CSV/Excel file and click
        <strong>Run Analysis</strong>. Then choose the <strong>Target</strong>, <strong>Predictors</strong> and
        <strong>Model memory</strong> and click <strong>Apply Configuration</strong>.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_footer() -> None:
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© {ist_now.year} {PRODUCT_NAME} | {COMPANY} | v{VERSION} | {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}")


def _render_metric_card(label: str, value: str, subtext: str = "", color_class: str = "neutral") -> None:
    """DRY helper for rendering metric cards consistently."""
    esc = html.escape
    sub = f'<div class="sub-metric">{esc(subtext)}</div>' if subtext else ""
    st.markdown(
        f'<div class="metric-card {esc(color_class)}" style="min-height: 120px;">'
        f"<h4>{esc(label)}</h4><h2>{esc(value)}</h2>{sub}</div>",
        unsafe_allow_html=True,
    )


def _record_sentence(record: dict | None, signal: str) -> str:
    if not record or not record.get("days"):
        return f"No past {signal} signals to judge this one by."
    return (
        f"Track record: after {record['episodes']} past {signal} episodes ({record['days']} days), "
        f"the target moved {_num(record['avg_fwd'], '+.2f')}% on average over the next {HEADLINE_HORIZON} rows, "
        f"{_num(record['edge'], '+.2f')} pts in the signal's favour vs neutral days (HAC t = {_num(record['t_stat'], '+.2f')}); "
        f"{_num(record['hit_rate'] * 100, '.0f')}% moved the signal's way. {_edge_verdict(record['t_stat'])}"
    )


def _render_primary_signal(signal: dict) -> None:
    """Primary signal card, always visible above the tabs."""
    sig = signal["signal"]
    signal_class = "undervalued" if sig == "BUY" else "overvalued" if sig == "SELL" else "fair"
    emoji = "🟢" if sig == "BUY" else "🔴" if sig == "SELL" else "🟡"
    if signal["has_bullish_div"]:
        div_badge = ' • <span class="status-badge buy">🔔 BULLISH DIVERGENCE</span>'
    elif signal["has_bearish_div"]:
        div_badge = ' • <span class="status-badge sell">🔔 BEARISH DIVERGENCE</span>'
    else:
        div_badge = ""

    score = signal["conviction_score"]
    if signal["warming_up"]:
        explanation = "Not enough history yet: every lookback needs its own calibration period before conviction is defined."
    elif sig == "BUY":
        explanation = (f"The valuation gap is in an undervalued zone on {signal['oversold_breadth']:.0f}% of lookbacks; "
                       f"conviction is {score:+.0f}. " + _record_sentence(signal.get("record"), "BUY"))
    elif sig == "SELL":
        explanation = (f"The valuation gap is in an overvalued zone on {signal['overbought_breadth']:.0f}% of lookbacks; "
                       f"conviction is {score:+.0f}. " + _record_sentence(signal.get("record"), "SELL"))
    else:
        explanation = (f"Conviction ({score:+.0f}) is inside the neutral band (±{CONVICTION_WEAK}). "
                       "No directional signal.")

    st.markdown(
        f'<div class="signal-card {signal_class}" style="padding: 1.5rem;">'
        f'<div class="label">WALK-FORWARD SIGNAL</div>'
        f'<div class="value">{emoji} {html.escape(sig)}</div>'
        f'<div class="subtext">'
        f'<strong>{html.escape(signal["strength"])}</strong> Strength • '
        f'<strong>{html.escape(signal["confidence"])}</strong> Breadth agreement • '
        f'Gap t½ = <strong>{_num(signal["ou_half_life"], ".0f")}d</strong>{div_badge}'
        f'</div>'
        f'<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid var(--border-color); '
        f'font-size: 0.85rem; line-height: 1.5; color: var(--text-secondary);">'
        f'<strong>Why this signal?</strong> {html.escape(explanation)}'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")


def _render_tab_dashboard_content(engine, ts_filtered, x_axis, signal, model_stats, regime_stats, ts, active_target) -> None:
    # ── Row 1: raw conviction ──────────────────────────────────────────────
    _section(
        "Base Conviction Score",
        "Raw breadth differential before smoothing: Overbought% − Oversold% of lookbacks "
        "(+ half the extreme-zone difference), range ±150. Negative = undervalued, positive = overvalued.",
    )
    raw = ts_filtered["ConvictionRaw"]
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=x_axis, y=raw.clip(lower=0), fill="tozeroy",
                                 fillcolor="rgba(239,68,68,0.12)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_raw.add_trace(go.Scatter(x=x_axis, y=raw.clip(upper=0), fill="tozeroy",
                                 fillcolor="rgba(16,185,129,0.12)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    colors = np.select(
        [raw > CONVICTION_MODERATE, raw >= CONVICTION_WEAK, raw < -CONVICTION_MODERATE, raw <= -CONVICTION_WEAK],
        [COLOR_RED, "rgba(239, 68, 68, 0.75)", COLOR_GREEN, "rgba(16, 185, 129, 0.75)"], default=COLOR_MUTED,
    )
    sizes = np.where(raw.abs() > CONVICTION_MODERATE, 10, 8)
    fig_raw.add_trace(go.Scatter(
        x=x_axis, y=raw, mode="lines+markers", name="Raw Conviction",
        line=dict(color=COLOR_GOLD, width=2.0), marker=dict(size=sizes, color=colors),
        hovertemplate="Date: %{x}<br>Raw conviction: %{y:.1f}<extra></extra>",
    ))
    fig_raw.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    for lvl, dash in ((CONVICTION_MODERATE, "dash"), (CONVICTION_WEAK, "dot")):
        fig_raw.add_hline(y=lvl, line_dash=dash, line_color="rgba(239,68,68,0.25)", line_width=1)
        fig_raw.add_hline(y=-lvl, line_dash=dash, line_color="rgba(16,185,129,0.25)", line_width=1)
    fig_raw.update_layout(height=350, xaxis_title=None, yaxis_title="Raw conviction",
                          yaxis=dict(range=[-155, 155], tickfont=dict(size=10)),
                          margin=dict(t=10, l=60, r=20, b=10), showlegend=False)
    apply_chart_theme(fig_raw)
    st.plotly_chart(fig_raw)
    st.markdown("---")

    # ── Row 2: smoothed conviction (the signal) ───────────────────────────
    _section(
        "Conviction (Leaky Accumulator) with 95% Band",
        f"Raw conviction accumulated with leak λ = 0.08 (≈8-row half-life) and soft-bounded to ±100. "
        f"This drives the signal: beyond ±{CONVICTION_WEAK} weak, ±{CONVICTION_MODERATE} moderate, ±{CONVICTION_STRONG} strong. "
        f"The band is the accumulator's own standard error.",
    )
    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(x=x_axis, y=ts_filtered["ConvictionUpper"], mode="lines",
                                  line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_conv.add_trace(go.Scatter(x=x_axis, y=ts_filtered["ConvictionLower"], mode="lines", line=dict(width=0),
                                  fill="tonexty", fillcolor="rgba(255,195,0,0.10)", name="95% band", hoverinfo="skip"))
    bounded = ts_filtered["ConvictionBounded"]
    fig_conv.add_trace(go.Scatter(x=x_axis, y=bounded.clip(lower=0), fill="tozeroy",
                                  fillcolor="rgba(239,68,68,0.12)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_conv.add_trace(go.Scatter(x=x_axis, y=bounded.clip(upper=0), fill="tozeroy",
                                  fillcolor="rgba(16,185,129,0.12)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_conv.add_trace(go.Scatter(x=x_axis, y=bounded, mode="lines", name="Conviction",
                                  line=dict(color=COLOR_GOLD, width=2.5),
                                  hovertemplate="Date: %{x}<br>Conviction: %{y:.1f}<extra></extra>"))
    fig_conv.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    for lvl in (CONVICTION_WEAK, CONVICTION_STRONG):
        fig_conv.add_hline(y=lvl, line_dash="dot", line_color="rgba(239,68,68,0.25)", line_width=1)
        fig_conv.add_hline(y=-lvl, line_dash="dot", line_color="rgba(16,185,129,0.25)", line_width=1)
    fig_conv.update_layout(height=350, xaxis_title=None, yaxis_title="Conviction",
                           yaxis=dict(range=[-100, 100], tickfont=dict(size=10)),
                           margin=dict(t=10, l=60, r=20, b=10), showlegend=False)
    apply_chart_theme(fig_conv)
    st.plotly_chart(fig_conv)

    score = signal["conviction_score"]
    sig, strength, regime = classify_conviction(score)
    if sig == "HOLD":
        title = "NEUTRAL ZONE"
        desc = (f"Conviction at {_num(score, '+.0f')} is inside ±{CONVICTION_WEAK}: the lookbacks do not agree on "
                f"over- or undervaluation.")
    else:
        side = "overvalued" if sig == "SELL" else "undervalued"
        title = f"{strength} {'OVERBOUGHT' if sig == 'SELL' else 'OVERSOLD'}"
        desc = (f"Conviction at {_num(score, '+.0f')}: the valuation gap has been {side} across lookbacks for long "
                f"enough to accumulate. " + _record_sentence(signal.get("record"), sig))
    width = signal["conviction_upper"] - signal["conviction_lower"]
    note = (f"Band width {_num(width, '.0f')} points: "
            + ("narrow, the reading is stable." if width < 30 else
               "wide, the reading is noisy; weight it less." if width > 60 else "moderate."))
    _text_card(title, desc, note)
    st.markdown("---")

    # ── Row 3: market state ───────────────────────────────────────────────
    _section("Market State")
    c1, c2, c3 = st.columns(3)
    with c1:
        _render_metric_card("OVERSOLD BREADTH", f"{_num(signal['oversold_breadth'], '.0f')}%",
                            f"Lookbacks with z < −{Z_THRESHOLD:g}",
                            "success" if signal["oversold_breadth"] > 60 else "neutral")
    with c2:
        _render_metric_card("OVERBOUGHT BREADTH", f"{_num(signal['overbought_breadth'], '.0f')}%",
                            f"Lookbacks with z > +{Z_THRESHOLD:g}",
                            "danger" if signal["overbought_breadth"] > 60 else "neutral")
    with c3:
        reg = signal["regime"]
        short = reg.replace("STRONGLY ", "STRONG ").replace("OVERSOLD", "OS").replace("OVERBOUGHT", "OB")
        _render_metric_card("CURRENT REGIME", short, f"Gap t½ = {_num(signal['ou_half_life'], '.0f')}d",
                            "success" if "OVERSOLD" in reg else "danger" if "OVERBOUGHT" in reg else "neutral")
    st.markdown("")

    total = max(regime_stats["n_valid"], 1)
    os_n = regime_stats["strongly_oversold"] + regime_stats["oversold"]
    ob_n = regime_stats["strongly_overbought"] + regime_stats["overbought"]
    neu_n = regime_stats["neutral"]
    os_pct, ob_pct, neu_pct = os_n / total * 100, ob_n / total * 100, neu_n / total * 100
    title = "OVERSOLD DOMINANT" if os_pct > 50 else "OVERBOUGHT DOMINANT" if ob_pct > 50 else "BALANCED HISTORY"
    body = (
        f"Of {total} rows with a defined conviction, {neu_pct:.0f}% were neutral ({neu_n}), "
        f"{os_pct:.0f}% oversold ({os_n}, of which {regime_stats['strongly_oversold']} strongly) and "
        f"{ob_pct:.0f}% overbought ({ob_n}, of which {regime_stats['strongly_overbought']} strongly). "
        f"Because the z-scores are calibrated against their own history, a long-run balance of oversold and "
        f"overbought time is expected; persistent dominance of one side means the gap has drifted."
    )
    _text_card(title, body)
    st.markdown("---")

    # ── Row 4: model quality ──────────────────────────────────────────────
    _section(
        "Model Quality",
        f"Edges are −correlation with the target's change over the next {HEADLINE_HORIZON} rows, so positive = "
        f"high readings preceded falls, as designed. t-stats are Newey-West (HAC) to allow for overlapping windows.",
    )
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        r2 = model_stats["r2_oos"]
        _render_metric_card("OOS R²", _num(r2, ".3f"), "Fair value vs actual (walk-forward)",
                            "success" if r2 > 0.7 else "warning" if r2 > 0.4 else "danger")
    with q2:
        t = model_stats.get("gap_edge_t", float("nan"))
        _render_metric_card("GAP EDGE", _num(model_stats.get("gap_edge"), "+.3f"),
                            f"t = {_num(t, '+.2f')} · gap → next {HEADLINE_HORIZON}", _edge_class(t))
    with q3:
        t = model_stats.get("signal_edge_t", float("nan"))
        _render_metric_card("SIGNAL EDGE", _num(model_stats.get("signal_edge"), "+.3f"),
                            f"t = {_num(t, '+.2f')} · conviction → next {HEADLINE_HORIZON}", _edge_class(t))
    with q4:
        spread, rmse = signal["model_spread"], model_stats.get("rmse_oos", float("nan"))
        ratio = spread / rmse if np.isfinite(rmse) and rmse > 0 else float("nan")
        _render_metric_card("MODEL SPREAD", _num(spread, ".2f"), f"Model disagreement = {_num(ratio * 100, '.0f')}% of RMSE",
                            "success" if ratio < 0.25 else "warning" if ratio < 0.5 else "danger")
    st.markdown("---")

    # ── Row 5: fair value ─────────────────────────────────────────────────
    _section("Actual vs Walk-Forward Fair Value",
             "Out-of-sample ensemble fair value ± model disagreement; below, the gap with its OU projection and 95% band.")
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], shared_xaxes=True, vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered["Actual"], mode="lines", name="Actual",
                             line=dict(color=COLOR_GOLD, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered["FairValue"], mode="lines", name="Fair Value (OOS)",
                             line=dict(color=COLOR_CYAN, width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered["FairValue"] + ts_filtered["ModelSpread"], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_axis, y=ts_filtered["FairValue"] - ts_filtered["ModelSpread"], mode="lines",
                             line=dict(width=0), fill="tonexty", fillcolor="rgba(6,182,212,0.08)",
                             name="Model disagreement", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Bar(x=x_axis, y=ts_filtered["Residual"], name="Gap (OOS)", showlegend=False,
                         marker_color=np.where(ts_filtered["Residual"] < 0, COLOR_GREEN, COLOR_RED)), row=2, col=1)
    fig.add_hline(y=0, line_color=COLOR_GOLD, line_width=1, row=2, col=1)

    if len(engine.ou_projection):
        if pd.api.types.is_datetime64_any_dtype(ts["Date"]):
            proj_x = pd.bdate_range(start=ts["Date"].iloc[-1], periods=OU_PROJECTION_DAYS + 1)[1:]
        else:
            proj_x = np.arange(len(ts), len(ts) + OU_PROJECTION_DAYS)
        fig.add_trace(go.Scatter(x=proj_x, y=engine.ou_projection, mode="lines", name="OU projection",
                                 line=dict(color=COLOR_GOLD, width=1.5, dash="dot"), opacity=0.6), row=2, col=1)
        fig.add_trace(go.Scatter(x=proj_x, y=engine.ou_projection_upper, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=proj_x, y=engine.ou_projection_lower, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,195,0,0.15)", name="OU 95% band",
                                 hoverinfo="skip"), row=2, col=1)
    fig.update_layout(height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(title_text=active_target, row=1, col=1)
    fig.update_yaxes(title_text="Gap", row=2, col=1)
    apply_chart_theme(fig)
    st.plotly_chart(fig)


def _render_tab_breadth(ts_filtered, x_axis, lookbacks) -> None:
    _section("Market Breadth", "Share of lookbacks whose calibrated z-score puts the gap in an under- or overvalued zone.")
    fig_zones = go.Figure()
    fig_zones.add_trace(go.Scatter(x=x_axis, y=ts_filtered["OversoldBreadth"], fill="tozeroy",
                                   fillcolor="rgba(16,185,129,0.15)", line=dict(color=COLOR_GREEN, width=2), name="Oversold"))
    fig_zones.add_trace(go.Scatter(x=x_axis, y=ts_filtered["OverboughtBreadth"], fill="tozeroy",
                                   fillcolor="rgba(239,68,68,0.15)", line=dict(color=COLOR_RED, width=2), name="Overbought"))
    fig_zones.add_hline(y=60, line_dash="dash", line_color="rgba(255,195,0,0.2)", line_width=1)
    fig_zones.update_layout(height=350, xaxis_title=None, yaxis_title="Breadth %",
                            yaxis=dict(range=[0, 100], tickfont=dict(size=10)),
                            margin=dict(t=10, l=60, r=20, b=10), showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)))
    apply_chart_theme(fig_zones)
    st.plotly_chart(fig_zones)

    st.markdown("---")
    _section("Signal Frequency", f"Lookbacks whose z-score crossed ∓{Z_THRESHOLD:g} that day (buy below, sell above).")
    fig_signals = go.Figure()
    fig_signals.add_trace(go.Bar(x=x_axis, y=ts_filtered["BuySignalBreadth"], name="Buy",
                                 marker=dict(color=COLOR_GREEN, opacity=0.8)))
    fig_signals.add_trace(go.Bar(x=x_axis, y=-ts_filtered["SellSignalBreadth"], name="Sell",
                                 marker=dict(color=COLOR_RED, opacity=0.8)))
    fig_signals.update_layout(height=300, xaxis_title=None, yaxis_title="Count", barmode="relative",
                              margin=dict(t=10, l=60, r=20, b=10), showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)))
    apply_chart_theme(fig_signals)
    st.plotly_chart(fig_signals)

    st.markdown("---")
    _section("Average Z-Score", f"Mean calibrated z across lookbacks; dashed lines at ±{Z_EXTREME:g}.")
    avg_z = ts_filtered["AvgZ"]
    fig_z = go.Figure(go.Bar(x=x_axis, y=avg_z, name="Avg Z", opacity=0.8,
                             marker_color=np.select([avg_z < -Z_THRESHOLD, avg_z > Z_THRESHOLD],
                                                    [COLOR_GREEN, COLOR_RED], default=COLOR_MUTED)))
    fig_z.add_hline(y=0, line_color="rgba(255,255,255,0.15)", line_width=1)
    fig_z.add_hline(y=Z_EXTREME, line_dash="dash", line_color="rgba(239,68,68,0.25)", line_width=1)
    fig_z.add_hline(y=-Z_EXTREME, line_dash="dash", line_color="rgba(16,185,129,0.25)", line_width=1)
    fig_z.update_layout(height=300, xaxis_title=None, yaxis_title="Z-Score",
                        margin=dict(t=10, l=60, r=20, b=10), showlegend=False)
    apply_chart_theme(fig_z)
    st.plotly_chart(fig_z)

    st.markdown("---")
    _section("Current Lookback States")
    for lb in lookbacks:
        z = ts_filtered[f"Z_{lb}"].iloc[-1]
        zone = ts_filtered[f"Zone_{lb}"].iloc[-1]
        color = COLOR_GREEN if "Under" in zone else COLOR_RED if "Over" in zone else COLOR_MUTED
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid #2A2A2A;">
            <span style="color: #888; font-size: 0.85rem;">{lb}-Day</span>
            <span style="color: {color}; font-weight: 600; font-size: 0.85rem;">{html.escape(zone)} ({_num(z, '+.2f')})</span>
        </div>
        """, unsafe_allow_html=True)


def _render_tab_data(ts_filtered, ts, active_target) -> None:
    _section(f"Time Series Data ({len(ts_filtered)} observations)")
    cols = ["Date", "Actual", "FairValue", "Residual", "ModelSpread", "AvgZ", "OversoldBreadth",
            "OverboughtBreadth", "ConvictionBounded", "Regime", "BullishDiv", "BearishDiv"]
    display = ts_filtered[[c for c in cols if c in ts_filtered.columns]].copy()
    for col, dec in {"AvgZ": 3, "ModelSpread": 3, "FairValue": 2, "Residual": 2, "ConvictionBounded": 1,
                     "OversoldBreadth": 1, "OverboughtBreadth": 1}.items():
        display[col] = display[col].round(dec)
    display["BullishDiv"] = display["BullishDiv"].map(lambda x: "🟢" if x else "")
    display["BearishDiv"] = display["BearishDiv"].map(lambda x: "🔴" if x else "")
    st.dataframe(display.rename(columns={"Residual": "Gap", "ConvictionBounded": "Conviction"}), hide_index=True, height=500)
    st.download_button("📥 Download Full CSV", ts.to_csv(index=False).encode("utf-8"),
                       f"aarambh_{active_target}_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")


def _render_tab_ml_diagnostics(engine, ts, signal, active_target, active_features) -> None:
    ou = engine.ou_params
    _section("Valuation-Gap Dynamics",
             f"OU fit of the gap with Kendall's small-sample bias correction. Current = last {ou.get('window', 0)} rows; "
             f"DFA α is measured on the gap's daily changes (< 0.5 = mean-reverting).")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _render_metric_card("GAP HALF-LIFE", f"{_num(ou.get('half_life'), '.0f')}d",
                            f"Full sample {_num(ou.get('half_life_full'), '.0f')}d", "primary")
    with c2:
        p = ou.get("adf_pvalue", float("nan"))
        _render_metric_card("ADF p-value", _num(p, ".3f", "n/a"), "H0: unit root (want < 0.05)",
                            "neutral" if not np.isfinite(p) else "success" if p < 0.05 else "danger")
    with c3:
        p = ou.get("kpss_pvalue", float("nan"))
        _render_metric_card("KPSS p-value", _num(p, ".3f", "n/a"), "H0: stationary (want > 0.05)",
                            "neutral" if not np.isfinite(p) else "success" if p > 0.05 else "danger")
    with c4:
        a = signal["hurst"]
        label = "n/a" if not np.isfinite(a) else "Mean-reverting" if a < 0.45 else "Trending" if a > 0.55 else "Random walk"
        _render_metric_card("DFA EXPONENT", _num(a, ".2f"), label,
                            "neutral" if not np.isfinite(a) else "success" if a < 0.45 else "danger" if a > 0.55 else "neutral")
    adf, kp = ou.get("adf_pvalue", float("nan")), ou.get("kpss_pvalue", float("nan"))
    if not HAS_STATSMODELS:
        stationarity = "n/a (install statsmodels for ADF/KPSS)"
    elif adf < 0.05 and kp > 0.05:
        stationarity = "Stationary ✅ (both tests agree)"
    elif adf < 0.05 or kp > 0.05:
        stationarity = "Mixed ⚠️ (tests disagree)"
    else:
        stationarity = "Non-stationary ⚠️"
    consistency = ("✅ consistent with full sample" if signal.get("theta_consistent", True)
                   else "⚠️ more than 2× away from the full-sample half-life")
    st.markdown(f"**Stationarity:** {stationarity} | **Current half-life:** {consistency}")

    if engine.theta_history:
        th = pd.DataFrame(engine.theta_history)
        fig_th = go.Figure(go.Scatter(x=_history_dates(ts, th["index"]), y=th["half_life"], mode="lines",
                                      line=dict(color=COLOR_CYAN, width=2), name="Half-life",
                                      hovertemplate="%{x}<br>Half-life: %{y:.1f}d<extra></extra>"))
        fig_th.add_hline(y=ou.get("half_life_full", np.nan), line_dash="dash", line_color="rgba(255,195,0,0.4)", line_width=1)
        fig_th.update_layout(height=260, yaxis_title="Half-life (rows)", yaxis_type="log",
                             margin=dict(t=10, l=60, r=20, b=10), showlegend=False)
        apply_chart_theme(fig_th)
        st.plotly_chart(fig_th)

    st.markdown("---")
    _section("Fair-Value Drivers",
             "Effective standardized coefficient of the whole ensemble (weights × each model's coefficients): the change "
             "in fair value for a one-standard-deviation move in each predictor. Share = |coefficient| / total.")
    shared = _price_sharing_predictors(active_target, active_features)
    if shared:
        st.info(f"{', '.join(shared)} share the index price with {active_target}. When price rises they move too, "
                f"so the fair value partly follows the price it is meant to judge and the gap shrinks. "
                f"Deselect them for a fair value built only from outside drivers.")
    coefs = engine.latest_feature_coefs
    if coefs:
        names = list(coefs)[::-1]
        vals = np.array([coefs[n] for n in names])
        shares = [engine.latest_feature_impacts.get(n, 0.0) for n in names]
        fig_imp = go.Figure(go.Bar(
            x=vals, y=names, orientation="h", customdata=shares,
            marker=dict(color=np.where(vals >= 0, COLOR_CYAN, COLOR_AMBER)),
            hovertemplate="%{y}: %{x:+.3f} (%{customdata:.1f}% share)<extra></extra>",
        ))
        fig_imp.update_layout(height=max(280, len(names) * 32), xaxis_title=f"Δ {active_target} per 1σ",
                              xaxis=dict(tickfont=dict(size=9), zeroline=True, zerolinecolor="rgba(255,255,255,0.2)"),
                              yaxis=dict(tickfont=dict(size=9)), margin=dict(t=10, l=10, r=20, b=10), showlegend=False)
        apply_chart_theme(fig_imp)
        st.plotly_chart(fig_imp)

        hist = engine.get_feature_impact_history()
        if len(hist) > 1:
            top = list(coefs)[:6]
            fig_h = go.Figure()
            x_hist = _history_dates(ts, hist["index"])
            for i, name in enumerate(top):
                fig_h.add_trace(go.Scatter(x=x_hist, y=hist[name], mode="lines", name=name,
                                           line=dict(width=1.8, color=SERIES_COLORS[i % len(SERIES_COLORS)])))
            fig_h.update_layout(height=300, yaxis_title="Share of impact %", margin=dict(t=10, l=60, r=20, b=10),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)))
            apply_chart_theme(fig_h)
            st.markdown("###### Driver shares over time (top 6 today)")
            st.plotly_chart(fig_h)
    else:
        st.info("Feature impact data not available for current configuration.")

    if engine.ensemble_weight_history:
        wh = pd.DataFrame(engine.ensemble_weight_history)
        fig_w = go.Figure()
        for i, name in enumerate(c for c in wh.columns if c != "index"):
            fig_w.add_trace(go.Scatter(x=_history_dates(ts, wh["index"]), y=wh[name] * 100, mode="lines",
                                       name=name, stackgroup="w", line=dict(width=0.5, color=SERIES_COLORS[i])))
        fig_w.update_layout(height=260, yaxis_title="Weight %", yaxis=dict(range=[0, 100]),
                            margin=dict(t=10, l=60, r=20, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=9)))
        apply_chart_theme(fig_w)
        st.markdown("###### Ensemble weights (inverse of trailing out-of-sample error)")
        st.plotly_chart(fig_w)

    st.markdown("---")
    _section("Signal Performance (Track Record)",
             f"What {active_target} did over the next h rows after each day the live signal rule said BUY "
             f"(conviction < −{CONVICTION_WEAK}) or SELL (> +{CONVICTION_WEAK}). Edge = move in the signal's favour "
             f"relative to neutral days; t is Newey-West with h lags because h-row windows overlap. "
             f"Episodes = separate runs of the signal, the honest sample size.")
    rows = []
    for h, p in engine.get_signal_performance().items():
        row = {"Horizon": f"{h} rows"}
        for side in ("buy", "sell"):
            s = p[side]
            tag = "✅" if s["t_stat"] >= 2 else "⚠️" if s["t_stat"] <= -2 else ""
            row[f"{side.upper()} days (episodes)"] = f"{s['days']} ({s['episodes']})"
            row[f"{side.upper()} avg fwd"] = f"{_num(s['avg_fwd'], '+.2f')}%"
            row[f"{side.upper()} edge"] = f"{_num(s['edge'], '+.2f')} pts"
            row[f"{side.upper()} HAC t"] = f"{_num(s['t_stat'], '+.2f')} {tag}".strip()
            row[f"{side.upper()} hit"] = f"{_num(s['hit_rate'] * 100, '.0f')}%"
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("✅ t ≥ 2: unlikely to be noise. ⚠️ t ≤ −2: the signal has historically worked in reverse. "
                   "Unmarked: not distinguishable from noise. "
                   "Whole-sample statistics: they describe the past, they do not guarantee the future.")
    else:
        st.info("Not enough history with a defined conviction to evaluate the signal yet.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def _reset_analysis() -> None:
    for key in ("data", "engine", "engine_cache", "run_analysis"):
        st.session_state.pop(key, None)
    _fetch_sheet.clear()


def _render_data_notices(report, date_col: str | None, target: str) -> None:
    """Tell the user every change the cleaner made to their data."""
    if report.duplicate_dates:
        shown = ", ".join(report.duplicate_dates[:5]) + (" …" if len(report.duplicate_dates) > 5 else "")
        log.warning("  ⚠ Duplicate dates dropped: %s", shown)
        st.warning(f"⚠️ {len(report.duplicate_dates)} row(s) in **{date_col}** repeat an earlier date ({shown}). "
                   "Only the first row for each date is used; check the source data for a typo.")
    if report.unparsed_dates:
        st.warning(f"⚠️ {report.unparsed_dates} row(s) have no readable date in **{date_col}** and were skipped.")
    if report.missing_target_rows:
        st.info(f"{report.missing_target_rows} row(s) have no **{target}** value and were skipped.")
    if report.empty_features:
        st.warning(f"⚠️ No data at all in: {', '.join(report.empty_features)}. Those predictors were left out.")
    if report.leading_rows_trimmed:
        start = report.start.strftime("%d %b %Y") if hasattr(report.start, "strftime") else f"row {report.leading_rows_trimmed}"
        st.info(f"Analysis starts {start}: **{report.start_limited_by}** has no data before then, and filling it "
                f"from later values would leak the future into the past. {report.leading_rows_trimmed} earlier rows "
                f"are not used; deselect {report.start_limited_by} to include them.")


def _render_staleness(report) -> None:
    end = report.end
    if end is None or not hasattr(end, "to_pydatetime"):
        return
    latest = end.to_pydatetime().replace(tzinfo=None)
    age = (datetime.now(timezone.utc).replace(tzinfo=None) - latest).days
    if age > STALENESS_DAYS:
        st.markdown(f"""
        <div style="background: rgba(239,68,68,0.1); border: 1px solid {COLOR_RED}; border-radius: 10px;
                    padding: 0.75rem 1.25rem; margin-bottom: 1rem; display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.4rem;">⚠️</span>
            <div>
                <span style="color: {COLOR_RED}; font-weight: 700;">Stale Data</span>
                <span style="color: #888; font-size: 0.85rem;"> — Last data point is <b>{latest.strftime('%d %b %Y')}</b>
                ({age} days ago). Update your data source, then click Reset Analysis.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


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
        data_source = st.radio("Source", ["📊 Google Sheets", "📤 Upload"], horizontal=True,
                               label_visibility="collapsed", index=0)
        has_data = "data" in st.session_state and "run_analysis" in st.session_state
        df = None

        if data_source == "📤 Upload":
            uploaded = st.file_uploader("CSV/Excel", type=["csv", "xlsx"], label_visibility="collapsed")
            if uploaded is None:
                st.info("📤 Upload a CSV or Excel file to begin analysis")
            elif not has_data and st.button("🚀 Run Analysis", type="primary"):
                try:
                    st.session_state["data"] = read_upload(uploaded.name, uploaded)
                except Exception as exc:
                    st.error(f"Could not read {uploaded.name}: {exc}")
                    return
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state["run_analysis"] = True
                st.rerun()
        else:
            sheet_url = st.text_input("Sheet URL", value=DEFAULT_SHEET_URL, label_visibility="collapsed",
                                      placeholder="https://docs.google.com/spreadsheets/d/…")
            if not has_data and st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Loading data…"):
                    loaded, error = load_google_sheet(sheet_url)
                if error:
                    st.error(f"Failed: {error}")
                    return
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state["data"] = loaded
                st.session_state["run_analysis"] = True
                st.rerun()

        if has_data:
            df = st.session_state["data"]
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if df is None:
        _render_header()
        _render_landing_page()
        _render_footer()
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return

    # ── Sidebar: Model Configuration (staged; nothing recomputes until Apply) ──
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🧠 Model Configuration</div>', unsafe_allow_html=True)

        active_target = st.session_state.get("active_target")
        if active_target not in numeric_cols:
            active_target = DEFAULT_TARGET if DEFAULT_TARGET in numeric_cols else numeric_cols[0]
        target_col = st.selectbox("Target Variable", numeric_cols, index=numeric_cols.index(active_target))

        date_options = ["None"] + all_cols
        active_date = st.session_state.get("active_date_col")
        if active_date not in date_options:
            active_date = next((c for c in all_cols if "date" in str(c).lower()), "None")
        date_col = st.selectbox("Date Column", date_options, index=date_options.index(active_date))

        active_memory = st.session_state.get("active_memory", DEFAULT_MODEL_MEMORY)
        memory = st.selectbox("Model memory", list(MODEL_MEMORY), index=list(MODEL_MEMORY).index(active_memory),
                              format_func=MEMORY_LABELS.get,
                              help="Adaptive: fair value follows the recent relationship, gaps close in weeks. "
                                   "Long: fair value is the long-run relationship, gaps can last months.")

        available = [c for c in numeric_cols if c != target_col]
        active_features = [f for f in st.session_state.get("active_features", ()) if f in available and f != active_target]
        if not active_features:
            active_features = [p for p in DEFAULT_PREDICTORS if p in available] or available[:3]
        with st.expander(f"📊 Predictor Columns ({len(active_features)} active)", expanded=False):
            staged = st.multiselect("Predictor Columns", options=available,
                                    default=[f for f in active_features if f in available],
                                    label_visibility="collapsed",
                                    help="Predictors of the walk-forward fair value regression.")
        if not staged:
            st.warning("⚠️ Select at least one predictor.")
            staged = active_features

        changes = []
        if set(staged) != set(active_features):
            added, removed = set(staged) - set(active_features), set(active_features) - set(staged)
            changes.append(" ".join(p for p in (f"+{len(added)}" if added else "", f"−{len(removed)}" if removed else "") if p)
                           + " predictors")
        if target_col != active_target:
            changes.append("target")
        if date_col != active_date:
            changes.append("date")
        if memory != active_memory:
            changes.append("memory")
        if changes:
            st.caption(f"Pending: {', '.join(changes)}")
        if st.button("✅ Apply Configuration" if changes else "No changes", disabled=not changes,
                     type="primary" if changes else "secondary"):
            st.session_state.update(active_target=target_col, active_features=tuple(staged),
                                    active_date_col=date_col, active_memory=memory)
            st.session_state.pop("engine", None)
            st.session_state.pop("engine_cache", None)
            st.rerun()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        if st.button("🔄 Reset Analysis", type="secondary", help="Reload the data source and start over."):
            _reset_analysis()
            st.rerun()
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.expander("⚙️ System Information", expanded=False):
            st.markdown(f"""
            <div style='padding: 0.5rem 0; font-size: 0.75rem; color: var(--text-secondary); line-height: 1.7;'>
                <div><strong style='color: var(--text-muted);'>Version:</strong> {VERSION}</div>
                <div><strong style='color: var(--text-muted);'>Lookbacks:</strong> {', '.join(f'{lb}D' for lb in LOOKBACK_WINDOWS)}</div>
                <div><strong style='color: var(--text-muted);'>Memory:</strong> {html.escape(MEMORY_LABELS[active_memory])}</div>
                <div><strong style='color: var(--text-muted);'>ADF/KPSS:</strong> {'<span style="color: var(--success-green);">● statsmodels</span>' if HAS_STATSMODELS else '<span style="color: var(--warning-amber);">⚠ statsmodels missing</span>'}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Clean data, then fit (cached per configuration) ───────────────────
    use_date = active_date if active_date != "None" else None
    data, report = clean_data(df, active_target, active_features, use_date)
    _render_data_notices(report, use_date, active_target)
    _render_staleness(report)

    features = [f for f in active_features if f in data.columns]
    if len(data) < MIN_DATA_POINTS:
        st.error(f"Need {MIN_DATA_POINTS}+ rows with the target and every selected predictor; have {len(data)}.")
        return
    if not features:
        st.error("No valid predictors left after cleaning.")
        return

    X, y = data[features].to_numpy(dtype=float), data[active_target].to_numpy(dtype=float)
    data_sig = str(pd.util.hash_pandas_object(data, index=False).sum())
    cache_key = f"{active_target}|{'|'.join(features)}|{active_memory}|{data_sig}"
    if st.session_state.get("engine_cache") != cache_key:
        log.info("AARAMBH v%s — target %s, %d predictors, %d rows, memory %s",
                 VERSION, active_target, len(features), len(data), active_memory)
        progress_bar = st.progress(0.0, text="Initializing engine…")
        engine = FairValueEngine(memory=active_memory).fit(X, y, feature_names=features,
                                                           progress_callback=progress_bar.progress)
        progress_bar.empty()
        st.session_state["engine"] = engine
        st.session_state["engine_cache"] = cache_key
    engine: FairValueEngine = st.session_state["engine"]

    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    regime_stats = engine.get_regime_stats()
    ts = engine.ts_data.copy()
    ts.insert(0, "Date", data[use_date].to_numpy() if use_date else np.arange(len(ts)))

    _render_primary_signal(signal)

    # ── Timeframe filter ──────────────────────────────────────────────────
    if "tf_selected" not in st.session_state:
        st.session_state.tf_selected = "1Y"
    timeframes = ["1M", "6M", "1Y", "2Y", "ALL"]
    st.markdown("""
    <style>
        div[data-testid="stHorizontalBlock"] button[kind="secondary"],
        div[data-testid="stHorizontalBlock"] button[kind="primary"] { width: 100% !important; }
        div[data-testid="stHorizontalBlock"] button[kind="primary"] {
            background: var(--primary-color) !important; color: #1A1A1A !important;
        }
    </style>
    """, unsafe_allow_html=True)
    for col, tf in zip(st.columns(len(timeframes)), timeframes):
        with col:
            if st.button(tf, key=f"tf_{tf}", type="primary" if st.session_state.tf_selected == tf else "secondary"):
                st.session_state.tf_selected = tf
                st.rerun()
    selected_tf = st.session_state.tf_selected

    ts_filtered = ts
    if selected_tf != "ALL":
        if use_date and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
            months = {"1M": 1, "6M": 6, "1Y": 12, "2Y": 24}[selected_tf]
            ts_filtered = ts[ts["Date"] >= ts["Date"].max() - pd.DateOffset(months=months)]
        else:
            ts_filtered = ts.iloc[max(0, len(ts) - TIMEFRAME_TRADING_DAYS[selected_tf]):]
    x_axis = ts_filtered["Date"]

    tab_dashboard, tab_breadth, tab_ml, tab_data = st.tabs([
        "**📊 Dashboard**", "**🗺️ Breadth Topology**", "**🧠 ML Diagnostics**", "**📋 Data Table**",
    ])
    with tab_dashboard:
        _render_tab_dashboard_content(engine, ts_filtered, x_axis, signal, model_stats, regime_stats, ts, active_target)
    with tab_breadth:
        _render_tab_breadth(ts_filtered, x_axis, engine.active_lookbacks)
    with tab_ml:
        _render_tab_ml_diagnostics(engine, ts, signal, active_target, features)
    with tab_data:
        _render_tab_data(ts_filtered, ts, active_target)
    _render_footer()


if __name__ == "__main__":
    main()
