# AARAMBH (आरंभ) v2.0 — Fair Value Breadth

**A Hemrek Capital Product**

Walk-forward valuation analysis for market reversals. Identifies where market turns BEGIN using out-of-sample ensemble fair value modeling, OU mean-reversion physics, and Kalman-filtered breadth conviction scoring.

---

## What It Does

Aarambh answers one question: **"Is the market's price justified by its fundamentals right now?"**

It computes a theoretical "Fair Value" for any target variable (typically NIFTY50 PE ratio) using macro predictors (yields, breadth, valuations), then measures the gap between actual and fair. When the gap stretches to extreme levels across *multiple timeframes simultaneously*, it signals a high-conviction reversal origin.

The system is complementary to **Arthagati** (mood-based sentiment), giving a 2D view: Arthagati says "how does the market *feel*", Aarambh says "how does the market *value*."

---

## Architecture

```
Google Sheet / CSV / Excel
         │
         ▼
┌────────────────────────────┐
│  Walk-Forward Regression   │  ← At each time T, fit ONLY on [0..T), predict T
│  Ridge + Huber + OLS       │  ← Ensemble: 3 models, averaged predictions
│  Model Spread = std(preds) │  ← Disagreement = uncertainty quantification
└────────────────────────────┘
         │
         ▼ Out-of-sample residuals
┌────────────────────────────┐
│  Multi-Lookback Z-Scores   │  ← 5D, 10D, 20D, 50D, 100D rolling z-scores
│  Zone Classification       │  ← Extreme Under / Under / Fair / Over / Extreme Over
│  Breadth Aggregation       │  ← "What % of lookbacks agree on oversold?"
└────────────────────────────┘
         │
         ▼ Raw Conviction Score (-100 to +100)
┌────────────────────────────┐
│  Kalman Filter             │  ← Smooth noisy conviction with adaptive gain
│  Confidence Bands (±1.96σ) │  ← 95% interval on smoothed conviction
│  Regime Classification     │  ← STRONGLY OVERSOLD / OVERSOLD / NEUTRAL / OVERBOUGHT / STRONGLY OVERBOUGHT
└────────────────────────────┘
         │
         ▼ Diagnostics
┌────────────────────────────┐
│  OU Estimation             │  ← θ, μ, σ on residuals → half-life, forward projection
│  Hurst Exponent            │  ← H < 0.5 confirms mean-reversion (validates thesis)
│  Swing Divergences         │  ← Price vs residual divergence at swing extrema
│  Signal Performance        │  ← Forward return analysis at 5D/10D/20D horizons
└────────────────────────────┘
```

---

## v2.0 Changes from v1.1

| # | Change | Why It Matters |
|---|--------|---------------|
| 1 | **Walk-Forward Regression** | v1.1 fit on ALL data then analyzed in-sample residuals — the R² was overfitted. v2.0 uses expanding-window regression: at each time T, fit only on [0..T), predict T. Every residual is out-of-sample. The R² displayed is real. |
| 2 | **OU Estimation on Residuals** | The entire Aarambh thesis is "residuals mean-revert." That IS the Ornstein-Uhlenbeck process. Estimating θ gives: half-life (ln2/θ), equilibrium (μ), and a 90-day forward projection showing when the gap is expected to close. |
| 3 | **Kalman-Filtered Conviction** | Raw conviction jumped erratically because it's a vote count. Kalman filtering gives a smooth signal with 95% confidence bands. Wide band = uncertain conviction. |
| 4 | **Model Disagreement** | When Ridge says 22, Huber says 24, and OLS says 20, the fair value is uncertain. v2.0 tracks the standard deviation across model predictions as a "Model Spread" metric. High spread = low confidence in fair value. |
| 5 | **Residual Hurst Exponent** | H < 0.5 empirically confirms the residuals mean-revert. H > 0.5 means they trend — the Aarambh thesis may not hold. This is a built-in validity check. |
| 6 | **Apply Button for Predictors** | v1.1 recomputed the entire walk-forward regression on every multiselect click. v2.0 uses staging → commit: user plays freely, "Apply & Compute" button triggers one computation. |
| 7 | **Data Staleness Warning** | If the latest data point is >3 days old, a red banner warns the user. |
| 8 | **Swing-Based Divergences** | v1.1 checked point-to-point price/residual direction over 5 bars — caught trivial noise. v2.0 uses `argrelextrema` to find actual swing highs/lows, then compares consecutive swings for real divergences. |

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
| IN10Y, IN02Y, IN30Y | India sovereign yields | Predictor |
| INIRYY | India inflation rate | Predictor |
| REPO | RBI repo rate | Predictor |
| US02Y, US10Y, US30Y | US treasury yields | Predictor |
| NIFTY50_DY | Nifty 50 dividend yield | Predictor |
| NIFTY50_PB | Nifty 50 price-to-book | Predictor |

### Engine Output (ts_data)

| Column | Type | Description |
|--------|------|-------------|
| Actual | float | Target variable value |
| FairValue | float | Walk-forward ensemble prediction (OOS) |
| Residual | float | Actual − FairValue (OOS gap) |
| ModelSpread | float | Std across Ridge/Huber/OLS predictions |
| Z_{5,10,20,50,100} | float | Rolling z-score at each lookback |
| Zone_{5,10,20,50,100} | str | Classification per lookback |
| OversoldBreadth | float | % of lookbacks in undervalued zone |
| OverboughtBreadth | float | % of lookbacks in overvalued zone |
| ConvictionRaw | float | Raw breadth-weighted score |
| ConvictionScore | float | Kalman-filtered conviction |
| ConvictionUpper/Lower | float | 95% Kalman confidence bounds |
| Regime | str | Classification from filtered conviction |
| BullishDiv / BearishDiv | bool | Swing-based divergence flags |

---

## Mathematical Primitives

| Function | Purpose | Used In |
|----------|---------|---------|
| `ornstein_uhlenbeck_estimate` | Fit OU params (θ, μ, σ) via AR(1) regression | Residual half-life, projection |
| `kalman_filter_1d` | Adaptive noise filtering with variance tracking | Conviction smoothing |
| `hurst_rs` | Hurst exponent via Rescaled Range analysis | Mean-reversion validation |

These are the same primitives used in Arthagati v2.1 — shared mathematical foundation across Hemrek systems.

---

## Signal Logic

### Conviction Score

```
ConvictionRaw = OverboughtBreadth − OversoldBreadth + 0.5 × (ExtremeOB − ExtremeOS)
ConvictionScore = Kalman(ConvictionRaw)
```

| ConvictionScore | Signal | Strength |
|-----------------|--------|----------|
| < −60 | BUY | STRONG |
| −60 to −40 | BUY | MODERATE |
| −40 to −20 | BUY | WEAK |
| −20 to +20 | HOLD | NEUTRAL |
| +20 to +40 | SELL | WEAK |
| +40 to +60 | SELL | MODERATE |
| > +60 | SELL | STRONG |

### Confidence

Confidence = HIGH if the dominant breadth (oversold for BUY, overbought for SELL) is ≥80%, MEDIUM if ≥60%, LOW otherwise. Combined with Model Spread: high model disagreement warrants a downgrade.

### OU Forward Projection

The residual follows `dr = θ(μ − r)dt + σdW`. The expected future residual is:

```
E[r(t+n)] = μ + (r_current − μ) × exp(−θ × n)
```

This dotted line on the residual chart shows when the gap is expected to close.

---

## Diagnostics Dashboard

The top metrics row shows 4 diagnostic cards:

| Card | What It Means |
|------|---------------|
| OU Half-Life | Days for the current valuation gap to halve. Short = fast reversion expected. Long = structural shift possible. |
| Residual Hurst | H < 0.45 = mean-reverting (thesis confirmed). H > 0.55 = trending (thesis challenged). 0.45–0.55 = random walk. |
| OOS R² | Out-of-sample R² from walk-forward regression. How much variance the model explains on data it hasn't seen. |
| Model Spread | Average disagreement across ensemble models. Low = confident fair value. High = uncertain estimate. |

---

## Setup

### Local

```bash
pip install -r requirements.txt
streamlit run correl.py
```

### GitHub Codespaces

Open in Codespace — the `.devcontainer` configuration auto-installs dependencies and starts Streamlit on port 8501.

### Streamlit Cloud

1. Push to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set main file to `correl.py`

---

## Configuration

1. **Data Source**: Upload CSV/Excel or paste Google Sheets URL
2. **Target Variable**: The variable to compute fair value for (default: NIFTY50_PE)
3. **Predictors**: Macro factors that drive the target (multi-select)
4. **Date Column**: For time-axis display
5. **Apply & Compute**: Commits configuration and runs walk-forward engine (one-click)

The Apply button prevents recomputation during predictor exploration. The walk-forward regression is the most expensive operation (~1–3 seconds per 1000 observations depending on predictor count).

---

## Relationship to Other Hemrek Systems

| System | Thesis | Signal |
|--------|--------|--------|
| **Aarambh** | Valuation gaps mean-revert | "PE is 2σ below fair value across all timeframes" |
| **Arthagati** | Market mood is measurable | "Sentiment is at +52, OU half-life 38 days" |
| **Pragyam** | Strategy selection is regime-dependent | "96 strategies ranked by current market character" |
| **Nirnay** | Quantitative signals guide decisions | "Signal confluence from multiple systems" |

Aarambh + Arthagati together give a 2D view: when both systems agree (e.g., valuation says BUY and mood says extreme bearish with reversion expected), the signal is highest conviction.

---

## Version History

| Version | Changes |
|---------|---------|
| v1.0.0 | Initial release: in-sample regression, fixed z-scores |
| v1.1.0 | Hemrek Design System, Google Sheets integration, divergence detection |
| v2.0.0 | Walk-forward regression, OU estimation, Kalman conviction, model disagreement, Hurst validation, Apply button, data staleness, swing divergences |

---

## License

Proprietary — Hemrek Capital. All rights reserved.
