# AARAMBH (आरंभ) v3.3.0 — Fair Value Breadth

**A @thebullishvalue Product**

Walk-forward valuation analysis for market reversals. Aarambh estimates an out-of-sample fair value for a target (default: NIFTY50 PE), measures the gap between actual and fair, scores that gap across five lookbacks on one calibrated scale, accumulates the agreement into a conviction score, and reports what the target did after every past signal.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [System Architecture](#system-architecture)
3. [Code Structure](#code-structure)
4. [Data Schema](#data-schema)
5. [Mathematical Primitives](#mathematical-primitives)
6. [Key Equations](#key-equations)
7. [Application Layout Guide](#application-layout-guide)
8. [Running the Application](#running-the-application)
9. [Version History](#version-history)
10. [References](#references)

---

## What It Does

Aarambh answers one question: **"Is the market's price justified by its predictors right now, and has that kind of reading preceded reversals before?"**

1. **Fair value.** An ensemble regression is refitted every 5 rows on earlier data only, and predicts each row's fair value of the target from that row's predictors.
2. **Gap.** Actual − fair value, strictly out of sample.
3. **Breadth.** The gap is scored against its trailing 5/10/20/50/100-row range. Each score is ranked against its own history, so |z| > 1 means the same thing in every lookback (≈32% of days) and |z| > 2 marks the extreme ≈5%.
4. **Conviction.** The net share of lookbacks flagging overvalued vs undervalued, smoothed by a leaky accumulator and soft-bounded to ±100. Beyond ±20 a signal is WEAK, ±40 MODERATE, ±60 STRONG.
5. **Track record.** For every signal the app shows how the target moved over the next 5/10/20 rows after past signals of that type, with Newey-West t-statistics that allow for overlapping windows. The system checks its own claim instead of asserting it.

The system is complementary to **Arthagati** (mood-based sentiment), giving a 2D view: Arthagati says how the market *feels*, Aarambh says how it is *valued*.

See [`AUDIT.md`](AUDIT.md) for the v3.3.0 audit: what was wrong in v3.2.3, the evidence, and what the corrected system measures on the source sheet.

---

## System Architecture

### Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. DATA INGESTION                                       aarambh/data.py    │
│  Google Sheet / CSV / Excel → numeric text ("6.48%", "1,234") → numbers     │
│  One date format per column → sort → drop repeated dates (keep first)       │
│  Forward-fill predictors only (never back-fill) → trim rows before the      │
│  latest-starting predictor → report every change to the user               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. WALK-FORWARD ENSEMBLE REGRESSION          FairValueEngine (engine.py)   │
│  Refit every 5 rows on rows < t · Memory: adaptive (500 rows, 252-row       │
│  half-life weights) or long (all history, equal weights)                    │
│  Ridge · Huber · ElasticNet · PCA→WLS, weighted by inverse trailing         │
│  out-of-sample MAE (60 rows, 5% floor) → fair value, model disagreement     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. CALIBRATED MULTI-LOOKBACK Z-SCORES                 stats.calibrated_z…  │
│  score_t = (gap_t − median(previous L)) / (IQR(previous L) / 1.349)         │
│  z_t = Φ⁻¹(rank of score_t among all earlier scores) for L = 5…100          │
│  Zones: |z|>1 over/undervalued · |z|>2 extreme · crossings of ±1            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  4. BREADTH & CONVICTION                        stats.drift_diffusion_filter│
│  Raw = (OB% − OS%) + ½(ExtremeOB% − ExtremeOS%)   ∈ [−150, 150]             │
│  Leaky accumulator s_t = (1−λ)s_{t−1} + λ·Raw_t, λ = 0.08, ± 95% std error  │
│  Bounded = 100·tanh(s/100) → classify_conviction → BUY / SELL / HOLD        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  5. GAP DIAGNOSTICS                                                         │
│  OU fit (Kendall bias-corrected AR(1)): current 252 rows + full sample,     │
│  rolling half-life, 90-row projection with OU predictive band               │
│  ADF / KPSS (statsmodels) · DFA α of the gap's daily changes                │
│  Swing divergences (target lower low, gap higher low, and the reverse)      │
│  Effective ensemble coefficients per predictor, recorded at every refit     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  6. VALIDATION                                  _compute_validation()       │
│  Forward % change of the target at 5/10/20 rows                             │
│  HAC regression on BUY/SELL dummies → edge vs neutral days, t, episodes     │
│  Gap edge and Signal edge = −corr with next-20-row change, HAC t            │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Causality contract.** Every column of `ts_data` except `FwdChg_*` uses only data up to its own row. `tests/test_engine.py::test_no_look_ahead` refits on a truncated sample and requires the past to be reproduced exactly.

---

## Code Structure

```
correl.py                 Streamlit app (entry point): CSS, rendering, sidebar, main()
aarambh/
├── config.py             Every tunable constant (windows, thresholds, memory, horizons)
├── data.py               parse_dates, coerce_numeric_frame, sheet_csv_url, load_sheet,
│                         read_upload, clean_data → (data, CleanReport)
├── stats.py              ornstein_uhlenbeck_estimate, ou_forecast, half_life,
│                         drift_diffusion_filter, soft_bound, hurst_dfa,
│                         robust_gap_scores, calibrated_zscores, classify_zones,
│                         detect_crossover_signals, newey_west_ols,
│                         information_coefficient, count_episodes
└── engine.py             classify_conviction, FairValueEngine
                            fit() → walk-forward → gap → z → breadth → conviction
                                  → pivots/divergences → forward changes → OU → DFA
                                  → validation
                            get_current_signal() · get_model_stats() · get_regime_stats()
                            get_signal_performance() · get_feature_impact_history()
tests/                    44 tests: data cleaning, calibration, OU, DFA, HAC, no look-ahead,
                          planted-edge detection
.streamlit/config.toml    Dark theme matching the design system
```

The engine has no Streamlit dependency and can be used directly:

```python
from aarambh.data import clean_data, load_sheet
from aarambh.engine import FairValueEngine

df, err = load_sheet(SHEET_URL)
data, report = clean_data(df, "NIFTY50_PE", ["IN10Y", "REPO", "US10Y"], "DATE")
engine = FairValueEngine(memory="adaptive").fit(data[["IN10Y", "REPO", "US10Y"]].values,
                                                data["NIFTY50_PE"].values,
                                                feature_names=["IN10Y", "REPO", "US10Y"])
engine.get_current_signal(), engine.validation
```

---

## Data Schema

### Input

Any table with a date column and numeric columns. Default Google Sheet layout:

| Column | Description | Role |
|--------|-------------|------|
| DATE | Trading date (DD/MM/YYYY) | Index |
| NIFTY50_PE | Nifty 50 Price-to-Earnings | Default target |
| AD_RATIO, REL_AD_RATIO, REL_BREADTH, COUNT | Breadth and price-action internals | Predictors |
| IN10Y, IN02Y, IN30Y, INIRYY, REPO | Indian yields, inflation, policy rate | Predictors |
| US10Y, US02Y, US30Y | US yields | Predictors |
| NIFTY50_DY, NIFTY50_PB | Valuation multiples | Predictors (see note) |

The default predictors mirror the variables marked `SELECTOR = 1` in the sheet's own model. **Note:** NIFTY50_PB and NIFTY50_DY contain the same index price as NIFTY50_PE, so a fair value built on them partly follows the price it is meant to judge. The app flags this next to the drivers chart.

### Cleaning rules (`clean_data`)

1. **Dates:** one format for the whole column (DD/MM/YYYY, MM/DD/YYYY, ISO and others detected; DD/MM/YYYY wins when the data is ambiguous). Unreadable dates are skipped and counted.
2. **Order:** rows are sorted by date. A repeated date keeps its first row, and the app names the repeated dates.
3. **Numbers:** text like `6.48%` or `24,536.60` becomes a number on load.
4. **Target:** rows without a target value are dropped. The target is never invented.
5. **Predictors:** gaps are forward-filled. **Nothing is back-filled.** Rows before the latest-starting predictor are trimmed, and the app says which predictor set the start date (on the default sheet INIRYY sets it to 12 Jun 2013).

### Engine Output (`ts_data`)

| Column | Type | Description |
|--------|------|-------------|
| Actual | float | Target value |
| FairValue | float | Walk-forward ensemble prediction (NaN for the first 20 rows) |
| Residual | float | The gap: Actual − FairValue |
| ModelSpread | float | Std across the ensemble's model predictions |
| Z_{5,10,20,50,100} | float | Calibrated z-score of the gap per lookback |
| Zone_{…} / Buy_{…} / Sell_{…} | str / bool | Zone label; crossing of ∓1 |
| OversoldBreadth / OverboughtBreadth | float | % of lookbacks with z < −1 / z > +1 |
| ExtremeOversold / ExtremeOverbought | float | % of lookbacks with z < −2 / z > +2 |
| AvgZ | float | Mean z across lookbacks |
| ConvictionRaw | float | (OB% − OS%) + ½(XOB% − XOS%) |
| ConvictionScore | float | Leaky-accumulated raw conviction |
| ConvictionBounded | float | 100·tanh(score/100): the signal input |
| ConvictionUpper / Lower | float | 95% band from the accumulator's standard error |
| Regime | str | NEUTRAL / OVERSOLD / OVERBOUGHT / STRONGLY … (N/A in warm-up) |
| IsPivotTop / IsPivotBottom | bool | Swing points of the gap, on the confirming row |
| BullishDiv / BearishDiv | bool | Target/gap swing divergences, on the confirming row |
| FwdChg_{5,10,20} | float | Forward % change of the target (the outcome; future by definition) |

---

## Mathematical Primitives

| Function | Purpose | Reference |
|----------|---------|-----------|
| `ornstein_uhlenbeck_estimate` | OU θ, μ, σ from a bias-corrected AR(1) | Kendall (1954) |
| `ou_forecast` | OU mean path and predictive std h steps ahead | — |
| `drift_diffusion_filter` | Unit-gain leaky accumulator with standard error | EWMA / leaky integrator |
| `hurst_dfa` | DFA-1 scaling exponent | Peng et al. (1994) |
| `calibrated_zscores` | Robust gap score ranked against its own past (conformal) | Vovk et al. (2005) |
| `newey_west_ols` | OLS with Bartlett HAC standard errors | Newey & West (1987) |

---

## Key Equations

### Ornstein-Uhlenbeck gap
```
dx = θ(μ − x)dt + σdW        sampled as  x_t − μ = a(x_{t−1} − μ) + ε,  a = e^{−θ}
â (OLS) → a = â + (1 + 3â)/n                 (Kendall bias correction)
σ² = Var(ε) · 2θ / (1 − a²)      half-life = ln 2 / θ
E[x_{t+h}] = μ + (x_t − μ)e^{−θh}      Var = σ²(1 − e^{−2θh}) / 2θ
```

### Calibrated z-score (lookback L)
```
score_t = (gap_t − median(gap_{t−L..t−1})) / (IQR(gap_{t−L..t−1}) / 1.349)
z_t     = Φ⁻¹( (#{past scores < score_t} + ½·#{= score_t} + ½) / (m + 1) )
```

### Conviction
```
Raw_t  = (OB% − OS%) + 0.5 × (ExtremeOB% − ExtremeOS%)
s_t    = (1 − λ)s_{t−1} + λ·Raw_t,           λ = 0.08 (≈ 8-row half-life)
v_t    = (1 − λ)v_{t−1} + λ(Raw_t − s_{t−1})²
band   = s_t ± 1.96 √(λ/(2 − λ) · v_t)
Bounded = 100 × tanh(s_t / 100)
```

### DFA exponent (of the gap's daily changes)
```
F(s) = RMS of the linearly detrended profile in windows of size s
α    = slope of log F(s) on log s       α < 0.5 mean-reverting · ≈ 0.5 random walk · > 0.5 trending
```

### Signal edge
```
FwdChg_h,t = β₀ + β_B·BUY_t + β_S·SELL_t + u_t      Newey-West(h) standard errors
BUY edge = β_B,  SELL edge = −β_S                    (positive = moved the signal's way)
```

---

## Application Layout Guide

Above the tabs, always visible: **data notices** (repeated dates, rows trimmed and why, staleness), then the **primary signal card**: BUY/SELL/HOLD, strength, breadth agreement, gap half-life, divergence badge, and a "Why this signal?" line that quotes the signal's own track record. Below it, the timeframe buttons (1M, 6M, 1Y, 2Y, ALL) filter every chart.

### 📊 Dashboard
1. **Base Conviction Score**: raw breadth differential (±150) with ±20/±40 markers.
2. **Conviction (Leaky Accumulator)**: the bounded signal with its 95% band and ±20/±60 lines, plus an interpretation card with the track record.
3. **Market State**: oversold and overbought breadth, current regime, and the regime distribution of the whole history.
4. **Model Quality**: OOS R², **Gap edge**, **Signal edge** (each with its HAC t; green t ≥ 2, red t ≤ −2), and model spread relative to RMSE.
5. **Actual vs Fair Value**: with model disagreement; below, the gap with its OU projection and 95% predictive band.

### 🗺️ Breadth Topology
Oversold/overbought breadth over time, z-crossing counts, average z with ±2 lines, and the current zone of each lookback.

### 🧠 ML Diagnostics
- **Valuation-gap dynamics**: current and full-sample half-life, ADF, KPSS, DFA α, and the rolling half-life over time.
- **Fair-value drivers**: signed effective ensemble coefficient per predictor (Δ target per 1σ); driver shares over time; ensemble weights over time. Price-sharing predictors are flagged.
- **Signal performance**: per horizon, BUY and SELL days (episodes), average forward change, edge vs neutral days, HAC t, hit rate.

### 📋 Data Table
The time series with the key columns, and a full CSV export of `ts_data`.

### Sidebar
1. **Data Source**: Google Sheet URL or CSV/Excel upload → Run Analysis.
2. **Model Configuration**: target, date column, model memory, predictors (expander). Changes are staged and listed as "Pending"; **Apply Configuration** recomputes.
3. **Reset Analysis**: clears the session and the sheet cache so a refreshed sheet is fetched.
4. **System Information**: version, lookbacks, memory, ADF/KPSS availability.

---

## Running the Application

```bash
pip install -r requirements.txt
export GOOGLE_SHEET_URL="https://docs.google.com/spreadsheets/d/<id>/edit#gid=<tab>"   # optional
streamlit run correl.py
```

The app opens at `http://localhost:8501`.

**Sheet URL.** The `GOOGLE_SHEET_URL` environment variable is read first, then Streamlit secrets: copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored), or paste it into *Streamlit Cloud → App settings → Secrets*. Without either, paste a link into the sidebar. The sheet must be shared as "Anyone with the link".

**Tests.**

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v1.0.0 | — | Initial release: in-sample regression, fixed z-scores |
| v2.0.0 | — | Walk-forward regression, OU estimation, Kalman conviction, model disagreement, Hurst validation |
| v2.2.0 | — | Vectorized walk-forward engine, sklearn LinearRegression swap |
| v3.0.0 | — | Conformal-style z-scores, drift-diffusion conviction, decoupled OU |
| v3.1.0 | — | Look-ahead fixes, break detection (non-functional; removed in 3.3.0) |
| v3.2.0 | — | Quantile z-scores, DDM variance cap |
| v3.2.1 | 2026-04-01 | Inverse-MAE ensemble weights, 4-tab UI, primary signal above tabs |
| v3.2.2 | 2026-04-05 | Production patch: version logging |
| v3.2.3 | 2026-04-06 | Dead-code removal, secrets file removed |
| **v3.3.0** | **2026-11-10** | **System audit**: date parsing, calibrated z-scores, no back-fill, true OOS ensemble weights, corrected OU and DFA, HAC track record shown with every signal, model memory option, package split, 44 tests. See `AUDIT.md`. |

---

## License

Proprietary — @thebullishvalue. All rights reserved. See `LICENSE.md` for full terms.

---

## References

- **Kendall, M. G. (1954)**. "Note on bias in the estimation of autocorrelation". *Biometrika*.
- **Andrews, D. W. K. (1993)**. "Exactly Median-Unbiased Estimation of First Order Autoregressive/Unit Root Models". *Econometrica*.
- **Newey, W. K. & West, K. D. (1987)**. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix". *Econometrica*.
- **Peng, C.-K. et al. (1994)**. "Mosaic organization of DNA nucleotides". *Physical Review E*.
- **Vovk, V., Gammerman, A. & Shafer, G. (2005)**. *Algorithmic Learning in a Random World*. Springer.
- **Welch, I. & Goyal, A. (2008)**. "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction". *Review of Financial Studies*.
- **Gu, S., Kelly, B., & Xiu, D. (2020)**. "Empirical Asset Pricing via Machine Learning". *Review of Financial Studies*.
