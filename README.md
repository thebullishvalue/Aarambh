# AARAMBH (आरंभ) v3.2.2 — Fair Value Breadth

**A @thebullishvalue Product**

Walk-forward valuation analysis for market reversals. Identifies where market turns BEGIN using out-of-sample ensemble fair value modeling, Ornstein-Uhlenbeck mean-reversion physics with Andrews (1993) median-unbiased estimation, and Drift-Diffusion (DDM) evidence accumulation with mean-reverting variance.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [System Architecture](#system-architecture)
3. [Code Structure](#code-structure)
4. [Data Schema](#data-schema)
5. [Mathematical Primitives](#mathematical-primitives)
6. [Key Equations](#key-equations)
7. [Application Layout Guide](#application-layout-guide-v322)
8. [Running the Application](#running-the-application)
9. [Version History](#version-history)
10. [References](#references)

---

## What It Does

Aarambh answers one question: **"Is the market's price justified by its fundamentals right now?"**

It computes a theoretical "Fair Value" for any target variable (typically NIFTY50 PE ratio) using macro predictors (yields, breadth, valuations), then measures the gap between actual and fair. When the gap stretches to extreme levels across *multiple timeframes simultaneously*, it signals a high-conviction reversal origin.

The system is complementary to **Arthagati** (mood-based sentiment), giving a 2D view: Arthagati says "how does the market *feel*", Aarambh says "how does the market *value*."

---

## System Architecture

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         1. DATA INGESTION LAYER                             │
│  Google Sheets / CSV / Excel → Cleaning → Chronological Sorting → F-Fill   │
│  Functions: load_google_sheet(), clean_data()                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    2. STRUCTURAL BREAK DETECTION                            │
│  Bai-Perron multiple breakpoint test → Regime-aware window resetting        │
│  Function: detect_structural_breaks()                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  3. WALK-FORWARD ENSEMBLE REGRESSION                        │
│  Expanding window (252-day exponential decay) · Sequential execution        │
│  FairValueEngine._walk_forward_regression()                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Ensemble: RidgeCV · HuberRegressor · ElasticNetCV · PCA+WLS        │   │
│  │  Dynamic Inverse-MAE weighting with 5% baseline floor               │   │
│  │  Output: OOS Fair Value · Model Spread (disagreement)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              4. MULTI-LOOKBACK CONFORMAL Z-SCORES                           │
│  Lookbacks: 5D, 10D, 20D, 50D, 100D · No look-ahead bias                  │
│  FairValueEngine._compute_multi_lookback_signals()                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Empirical quantile normalization (fat-tail preservation)           │   │
│  │  Zone Classification: Extreme Under → Under → Fair → Over → Extreme │   │
│  │  Crossover signal detection (±1σ threshold)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    5. BREADTH AGGREGATION                                   │
│  FairValueEngine._compute_breadth_metrics()                                 │
│  Oversold% = % lookbacks in undervalued zones                               │
│  Overbought% = % lookbacks in overvalued zones                              │
│  Raw Conviction = OB% - OS% + 0.5×(Extreme OB - Extreme OS)                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              6. DRIFT-DIFFUSION FILTER                                      │
│  FairValueEngine._compute_ddm_conviction()                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  State:  s_t = (1-λ)×s_{t-1} + λ×drift_t                            │   │
│  │  Var:    σ²_t = (1-λ)×σ²_{t-1} + λ×σ²_LR + 0.5×|drift_t|            │   │
│  │  Output: Smoothed conviction · 95% confidence bands · ±100 bounds   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  7. OU MEAN-REVERSION DIAGNOSTICS                           │
│  FairValueEngine._compute_ou_diagnostics()                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Andrews (1993) Median-Unbiased AR(1) Estimator                     │   │
│  │  Rolling θ estimation (60-day window) → Projection bands            │   │
│  │  ADF/KPSS stationarity tests → 90-day forward path                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  8. DIVERGENCE & PIVOT DETECTION                            │
│  FairValueEngine._find_pivots(), _compute_divergences()                     │
│  Swing-based extremum detection → Bullish/Bearish divergence flags          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  9. HURST EXPONENT (DFA)                                    │
│  FairValueEngine._compute_hurst()                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Detrended Fluctuation Analysis (Peng et al., 1994)                 │   │
│  │  ADF stationarity guard → H < 0.5 (MR) / ≈0.5 (RW) / >0.5 (Trend)  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  10. SIGNAL PERFORMANCE ANALYSIS                            │
│  FairValueEngine.get_signal_performance()                                   │
│  Forward change analysis at 5D, 10D, 20D · t-statistics & p-values          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Structure

```
correl.py (2,859 lines)
│
├── Constants & Configuration (lines 70–136)
│   VERSION, PRODUCT_NAME, COMPANY
│   Engine defaults: LOOKBACK_WINDOWS, MIN_TRAIN_SIZE, MAX_TRAIN_SIZE, etc.
│   Signal thresholds: CONVICTION_STRONG, CONVICTION_MODERATE, CONVICTION_WEAK
│   DDM parameters: DDM_LEAK_RATE, DDM_DRIFT_SCALE, DDM_LONG_RUN_VAR
│   Default predictors, Google Sheets URL, chart theme colors
│
├── Logging Infrastructure (lines 138–199)
│   class TerminalLogger          — Curated terminal logging with progress indicators
│   class StreamlitLogHandler     — Custom handler for Streamlit environment
│
├── Streamlit Page Config & CSS (lines 201–383)
│   st.set_page_config()
│   Premium CSS: @thebullishvalue Design System, signal cards, metric cards, tables
│
├── Mathematical Primitives (lines 385–658)
│   def ornstein_uhlenbeck_estimate()     — OU params with Andrews (1993) MU correction
│   def drift_diffusion_filter()          — Leaky DDM with mean-reverting variance
│   def hurst_dfa()                       — Hurst via Detrended Fluctuation Analysis
│   def andrews_median_unbiased_ar1()     — AR(1) with median-unbiased correction
│   def detect_structural_breaks()        — Bai-Perron multiple breakpoint test
│   def compute_conformal_zscores()       — Quantile-based z-scores (no look-ahead)
│
├── Helper Utilities (DRY) (lines 660–765)
│   def _safe_array_operation()           — Safe array ops with NaN handling
│   def _classify_zones()                 — Map z-scores to valuation zone labels
│   def _detect_crossover_signals()       — Z-score threshold crossovers
│   def _compute_significance()           — t-statistic and p-value computation
│   def _apply_conviction_bounds()        — Soft tanh bounds for conviction scores
│
├── FairValueEngine Class (lines 767–1660)
│   │
│   ├── Public API
│   │   def fit()                         — Full walk-forward pipeline orchestration
│   │   def get_current_signal()          — Current composite signal derivation
│   │   def get_model_stats()             — Model fit statistics
│   │   def get_regime_stats()            — Regime distribution counts
│   │   def get_signal_performance()      — Forward returns with significance testing
│   │   def get_feature_impact_history()  — Time-series of feature impacts
│   │
│   ├── Walk-Forward Regression (Private)
│   │   def _walk_forward_regression()    — Expanding-window ensemble with dynamic refit
│   │   def _process_wf_chunk()           — Single chunk processing with break awareness
│   │   def _fit_ensemble()               — Ridge + Huber + ElasticNet + PCA-WLS fitting
│   │   def _predict_ensemble()           — Vectorized prediction with inverse-MAE weighting
│   │   def _compute_feature_impacts()    — PCA back-projection to original features
│   │
│   ├── Analytics Pipeline (Private)
│   │   def _compute_model_stats()        — OOS R², RMSE, MAE, R² vs Random Walk
│   │   def _compute_multi_lookback_signals() — Conformal z-scores per lookback
│   │   def _compute_breadth_metrics()    — Oversold/Overbought breadth aggregation
│   │   def _compute_ddm_conviction()     — Drift-Diffusion filtering with bounds
│   │   def _compute_divergences()        — Swing-based divergence detection
│   │   def _find_pivots()                — Pivot high/low identification
│   │   def _compute_forward_changes()    — Winsorized forward returns
│   │   def _compute_ou_diagnostics()     — OU estimation with rolling θ
│   │   def _compute_hurst()              — DFA Hurst with ADF stationarity guard
│
├── Data Utilities (lines 1662–1730)
│   @st.cache_data load_google_sheet()    — Google Sheets CSV extraction
│   def clean_data()                      — Column selection, coercion, cleaning
│   def apply_chart_theme()               — @thebullishvalue dark theme for Plotly figures
│
├── UI Rendering (lines 1732–2463)
│   def _render_header()                  — Premium header banner
│   def _render_landing_page()            — Pre-analysis landing page (3 feature cards)
│   def _render_footer()                  — Copyright + version footer
│   def _render_metric_card()             — DRY helper for consistent metric cards
│   def _render_primary_signal()          — Always-visible signal card above tabs
│   def _render_tab_dashboard_content()   — Dashboard tab (6 rows)
│   def _render_tab_breadth()             — Breadth Topology tab
│   def _render_tab_data()                — Data Table tab with export
│   def _render_tab_ml_diagnostics()      — ML Diagnostics tab
│
└── Main Application (lines 2465–2859)
    def main()                            — Sidebar config, data loading, engine orchestration,
                                          — tab layout, timeframe filtering, caching
```

---

## Data Schema

### Input

Any tabular dataset with numeric columns. Default Google Sheet layout:

| Column | Description | Role |
|--------|-------------|------|
| DATE | Trading date | Index |
| NIFTY50_PE | Nifty 50 Price-to-Earnings | Default target |
| AD_RATIO | Advance/Decline ratio | Predictor |
| REL_AD_RATIO | Relative A/D ratio | Predictor |
| REL_BREADTH | Relative breadth | Predictor |
| COUNT | Market breadth count | Predictor |
| IN10Y, IN02Y, IN30Y | Indian yields | Predictors |
| US10Y, US02Y, US30Y | US yields | Predictors |
| NIFTY50_DY, NIFTY50_PB | Valuation metrics | Predictors |

### Engine Output (ts_data)

| Column | Type | Description |
|--------|------|-------------|
| Actual | float | Target variable value |
| FairValue | float | Walk-forward ensemble prediction (OOS) |
| Residual | float | Actual − FairValue (OOS gap) |
| ModelSpread | float | Std across Ridge/Huber/ElasticNet/WLS predictions |
| Z_{5,10,20,50,100} | float | Conformal z-score at each lookback (no look-ahead) |
| Zone_{5,10,20,50,100} | str | Classification per lookback |
| ConvictionScore | float | DDM-filtered conviction (mean-reverting var) |
| ConvictionBounded | float | Soft-bounded conviction (±100 via tanh) |
| ConvictionUpper/Lower | float | 95% DDM confidence bands |
| OversoldBreadth | float | % lookbacks in undervalued zones |
| OverboughtBreadth | float | % lookbacks in overvalued zones |
| Regime | str | STRONGLY OVERSOLD / OVERSOLD / NEUTRAL / OVERBOUGHT / STRONGLY OVERBOUGHT |
| BullishDiv / BearishDiv | bool | Swing-based divergence flags |
| FwdChg_{5,10,20} | float | Forward % change (winsorized ±100%) |

---

## Mathematical Primitives

| Function | Purpose | Reference | Used In |
|----------|---------|-----------|---------|
| `ornstein_uhlenbeck_estimate` | OU params with Andrews MU correction | Andrews (1993) | Residual half-life, projection |
| `drift_diffusion_filter` | Leaky DDM with mean-reverting variance | SPRT literature | Conviction smoothing |
| `hurst_dfa` | Hurst via Detrended Fluctuation Analysis | Peng et al. (1994) | Mean-reversion validation |
| `andrews_median_unbiased_ar1` | AR(1) with median-unbiased correction | Andrews (1993) | Half-life estimation |
| `detect_structural_breaks` | Bai-Perron multiple breakpoint test | Bai & Perron (1998) | Regime detection |
| `compute_conformal_zscores` | Quantile-based z-scores | Conformal prediction | Fat-tail preservation |

---

## Key Equations

### Ornstein-Uhlenbeck Process
```
dx = θ(μ - x)dt + σdW
```
- **θ (theta)**: Mean-reversion speed (Andrews MU corrected)
- **μ (mu)**: Equilibrium level (long-run mean)
- **σ (sigma)**: Volatility of innovations
- **Half-life**: t₁/₂ = ln(2) / θ

### Drift-Diffusion Filter
```
State:  s_t = (1 - λ) × s_{t-1} + λ × drift_t
Variance: σ²_t = (1 - λ) × σ²_{t-1} + λ × σ²_LR + 0.5 × |drift_t|
```
- **λ (leak_rate)**: 0.08 (calibrated)
- **σ²_LR**: Long-run variance (100.0 for conviction scores)

### DFA Hurst Exponent
```
F(s) = √(1/s × Σ(RMS_detrended²))
H = slope of log(F(s)) vs log(s)
```
- **H < 0.5**: Mean-reverting
- **H ≈ 0.5**: Random walk
- **H > 0.5**: Trending

### Conviction Score (Soft-Bounded)
```
Conviction_bounded = 100 × tanh(Conviction_raw / 100)
```

---

## Application Layout Guide (v3.2.2)

The Streamlit interface has been reorganized into **four logically flowing tabs** for optimal quantitative comprehension:

### 📊 Dashboard
**Purpose:** Main execution view — clean information flow from signal → conviction → regime → valuation.

**Layout Structure (6 Rows):**

**Row 1 — Primary Signal (Full Width):**
- Large signal card showing BUY/SELL/HOLD with strength, confidence, OU half-life, and divergence badges

**Row 2 — Base Conviction Score (Full Width):**
- *The core signal generator* — Raw breadth differential (Oversold% − Overbought%) across all lookbacks
- Full-width chart with zone shading and **threshold markers**:
  - **Bright Green dots (●)**: Conviction < −40 (Strong oversold)
  - **Dim Green dots (●)**: Conviction −20 to −40 (Moderate oversold)
  - **Dim Red dots (●)**: Conviction +20 to +40 (Moderate overbought)
  - **Bright Red dots (●)**: Conviction > +40 (Strong overbought)

**Row 3 — SPRT DDM Confidence Boundaries (Full Width):**
- *The smoothed signal* — Drift-Diffusion accumulation with mean-reverting variance
- 95% confidence bands shown as shaded gold region

**Row 4 — Market State (3 cards + table + context):**
- **Cards:** Oversold Breadth %, Overbought Breadth %, Current Regime (with OU t½)
- **Table:** Regime distribution history (Strongly OS → Strongly OB)
- **Context Box:** Plain-language interpretation of current regime

**Row 5 — Model Quality (4 cards):**
- OOS R², R² vs Random Walk, DFA Hurst, Model Spread

**Row 6 — Fair Value Plot (Full Width):**
- Actual vs Walk-Forward Fair Value with model uncertainty bands and OU 90-day projection

**When to use:** Start here for comprehensive real-time snapshot of market valuation, conviction trajectory, and actionable signals.

---

### 🗺️ Breadth Topology
**Purpose:** Conformal zones, extreme distributions, and signal generation across all lookback windows.

**Components:**
- **Zone Breadth Chart:** Time-series of oversold/overbought breadth percentages (5D, 10D, 20D, 50D, 100D lookbacks)
- **Buy/Sell Signal Count:** Crossover signals when Z-score crosses ±1σ threshold — shows historical signal frequency
- **Average Z-Score Chart:** Multi-lookback average conformal z-score with extreme thresholds (±2σ)
- **Current Lookback Breakdown:** Real-time zone classification for each individual lookback window

**When to use:** Analyze market breadth convergence/divergence across timeframes. Ideal for identifying "extreme" readings where multiple lookbacks agree and reviewing historical signal frequency.

---

### 🧠 ML Diagnostics
**Purpose:** OU mean-reversion diagnostics, feature-impact tracking, and signal performance validation.

**Components:**
- **OU Mean-Reversion Diagnostics:** Andrews (1993) median-unbiased half-life, ADF/KPSS stationarity tests, θ stability
- **Feature Impact History:** Time-varying feature weights from PCA+WLS back-projection (shows which macro predictors drive fair value)
- **Signal Performance Table:** Hit rates, average forward changes, and t-statistics for 5D/10D/20D holding periods

**When to use:** Debug model behavior, track factor rotation, validate stationarity assumptions, and assess statistical significance of signals.

---

### 📋 Data Table
**Purpose:** Underlying data streams and export functionality.

**Components:**
- **Time-Series Table:** Full historical data with all computed columns (FairValue, Residual, ConvictionScore, Regime, etc.)
- **Download Button:** Export complete dataset as CSV for external analysis

**When to use:** Audit raw outputs, verify calculations, or export for further research.

---

### Sidebar Structure

The sidebar is organized into **two collapsible expanders** to prevent visual overflow:

1. **📊 Predictor Columns:** Select/deselect macro predictors for the ensemble model
2. **Model Configuration:** Target variable, date column, and apply button

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# (Optional) Set environment variable for Google Sheets
# Create a .env file from the example template
cp .env.example .env
# Edit .env and add your Google Sheet URL

# Run Streamlit app
streamlit run correl.py
```

The app will open at `http://localhost:8501`.

### Environment Variables

- **`GOOGLE_SHEET_URL`**: URL of the Google Sheet containing market data. Copy from `.env.example` to `.env` and set your sheet URL.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | — | Initial release: in-sample regression, fixed z-scores |
| v2.0.0 | — | Walk-forward regression, OU estimation, Kalman conviction, model disagreement, Hurst validation |
| v2.2.0 | — | Vectorized walk-forward engine, sklearn LinearRegression swap |
| v3.0.0 | — | ADAM Refactor: Conformal inference, Drift-Diffusion Model, decoupled OU |
| v3.1.0 | — | Initial ADAM Critical Fixes: Look-ahead avoidance, Bai-Perron break implementation |
| v3.2.0 | — | Initial ADAM Phase I: True Conformal quantiles, DDM variance capped, Bai-Perron binding |
| **v3.2.1** | **2026-04-01** | **ADAM Phase II + Phase III Production Release**: Dynamic Inverse-MAE weighting, double-bounding excision, temporal mapping fixes, Hurst safeguards, UI/UX reorganization (4-tab layout, Primary Signal above tabs, Swing-style timeframe buttons), visualization standards, code quality improvements |
| **v3.2.2** | **2026-04-05** | **Production Patch**: Devcontainer configuration corrected (entry point fixed to `correl.py`), logging standardized to use dynamic `VERSION` constant, production deployment hardening |

---

## License

Proprietary — @thebullishvalue. All rights reserved. See `LICENSE.md` for full terms.

---

## References

- **Andrews, D. W. K. (1993)**. "Exactly Median-Unbiased Estimation of First Order Autoregressive/Unit Root Models". *Econometrica*.
- **Bai, J. & Perron, P. (1998)**. "Estimating and Testing Linear Models with Multiple Structural Changes". *Econometrica*.
- **Peng, C.-K. et al. (1994)**. "Mosaic organization of DNA nucleotides". *Physical Review E*.
- **Welch, I. & Goyal, A. (2008)**. "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction". *Review of Financial Studies*.
- **Gu, S., Kelly, B., & Xiu, D. (2020)**. "Empirical Asset Pricing via Machine Learning". *Review of Financial Studies*.
