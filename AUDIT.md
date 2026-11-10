# Aarambh System Audit — v3.2.3 → v3.3.0

**Date:** 2026-11-10 · **Scope:** idea, implementation, outputs, deployment · **Data:** the live source sheet (5,042 rows, 08 Jun 2006 – 09 Nov 2026)

## 1. What the system is for

Aarambh answers: *is the target (NIFTY50 PE) above or below what its predictors justify, by an unusual amount, across several lookbacks at once, and is that the start of a reversal?*

For that answer to be trustworthy the system has to:

1. use only information available on each day (no look-ahead),
2. score "unusual" the same way in every lookback, so "breadth across lookbacks" means something,
3. state uncertainty that matches the maths it claims,
4. tell the user whether its signals have actually preceded reversals.

v3.2.3 fell short on all four. v3.3.0 fixes the defects and adds (4), so the system now checks its own claims.

## 2. Method

1. Read every line of `correl.py`, README, CHANGELOG, LICENSE, requirements and the devcontainer.
2. Built a headless harness that runs the engine on the live sheet and records every output, so each fix was measured before and after.
3. Checked each statistical claim against its definition, and against the data where possible: z calibration, sign of the predictive relationship (Newey-West t-stats), OU bias direction, DFA behaviour.
4. Wrote tests for the invariants, chief among them **no look-ahead**: refitting on a truncated sample must reproduce the past exactly.

## 3. Findings and resolutions

Severity reflects the effect on the signal the user acts on.

### Critical — the output was wrong

| # | Finding | Evidence | Resolution |
|---|---------|----------|------------|
| C1 | **Dates read month-first.** `pd.to_datetime` inferred MM/DD from `08/06/2006`. | 3,054 of 5,042 rows silently dropped; the rest scrambled; latest date read as 10 Dec 2026. | `parse_dates` picks one format for the column (fixed earlier today, now in `aarambh/data.py`). |
| C2 | **z-scores mis-scaled.** The 2.5–97.5 percentile range was divided by 1.35, the IQR constant for the 25–75 range. Scale differed by lookback. | z std by lookback 5/10/20/50/100: **1.61 / 0.98 / 0.71 / 0.56 / 0.49**. "Overvalued" meant ≈2.9σ; long lookbacks almost never fired, and the 5-day window dominated "breadth". | Robust gap score (median, IQR/1.349), then **conformal calibration** against its own past: z = Φ⁻¹(rank). Now **1.00 / 0.98 / 1.01 / 1.06 / 1.02**, with ≈32% of days at \|z\|>1 and ≈5% at \|z\|>2 in every lookback. |
| C3 | **Future values back-filled into the past** (`bfill`). | INIRYY starts 12 Jun 2013, so **34.5% of rows** carried a 2013 value. Yields for 2006–07 came from 2 Jan 2008. | Forward-fill only. Leading rows without every predictor are trimmed, and the app says which predictor set the start date. |
| C4 | **Ensemble "out-of-sample" weights were in-sample.** The validation slice sat inside the training window, which rewards overfitting. | `X[t_start−val:t_start]` ⊂ training set. | Each model's own true OOS errors over the previous 60 rows; equal weights until 10 exist; 5% floor on normalised weights. |
| C5 | **OU bias correction had the wrong sign.** OLS underestimates AR(1) persistence; the code subtracted (1+3a)/n, which doubled the bias. The headline used a 60-row window. σ was wrong for a > 0.98, and μ was inconsistent with the corrected a. | Displayed half-life **2 days**; the full-sample half-life was 15. | Kendall correction a + (1+3a)/n (test: lands within 0.01 of the true a; OLS does not). Exact σ for every a. Current = trailing 252 rows; full sample shown alongside. Displayed now: **8 d** current, **18 d** full sample. |
| C6 | **The headline conviction pointed the wrong way.** A consequence of C2: it followed the 5-day residual. | Conviction vs NIFTY over the next 5 days: corr +0.064, **HAC t +2.33**. "Overbought" days were followed by *rises*. | Fixed by C2–C4. The relationship is now measured and shown (§5). |

### High — the claim did not match the computation

| # | Finding | Resolution |
|---|---------|------------|
| H1 | **Bai-Perron break detection never ran.** It imported `statsmodels.tsa.regime_switching.bai_perron`, which does not exist. As a side effect the UI reported "statsmodels missing" even though it was installed. The design was also look-ahead: breaks found on the full sample were applied to earlier dates. | Removed. Regime adaptivity comes from model memory. The statsmodels flag now reflects ADF/KPSS only. |
| H2 | **Signal t-stats ignored overlapping windows.** 20-day forward changes at daily frequency share 19 days, so plain t overstates significance. The performance table also used a different rule (unbounded score > 40) from the live signal (bounded > 20). | Newey-West (HAC) regression of forward change on BUY/SELL dummies with lags = h, using the live-signal rule. The table reports episodes (the honest sample size), edge vs neutral days, and hit rate. |
| H3 | **The OU projection band reflected only θ uncertainty.** It left out diffusion, and grew exponentially when θ_std > θ. | Proper OU predictive band: ±1.96·√(σ²(1−e^{−2θh})/2θ). |
| H4 | **DFA "Hurst" was misapplied.** It was computed on residual *levels* over scales n/10…n/4, forced to 0.5 when ADF p > 0.05, and read as "< 0.5 = mean-reverting", which holds only for increments. | DFA-1 on the gap's daily changes, with log-spaced scales from 10 to n/4. Tested on white noise (≈0.5), mean-reverting increments (< 0.4) and trending series (> 0.6). |
| H5 | **The "SPRT DDM" didn't match its docstring.** The code had no λ on the drift term (gain 1.875), seeded its state from the next 20 observations (look-ahead), and used an ad-hoc variance with no derivation. | Unit-gain leaky accumulator s = (1−λ)s + λe starting at the first observation. Band = the accumulator's standard error √(λ/(2−λ)·v), where v is the EW innovation variance. |
| H6 | **The past changed when new data arrived.** The refit interval was `n // 200`, so it depended on the total sample length. | Fixed `REFIT_INTERVAL = 5`. |
| H7 | **Results weren't reproducible.** ElasticNet used random coordinate descent with no seed. | `random_state=0`. |
| H8 | **Interpretation text claimed history it never computed** ("Historical data suggests this is a buying opportunity", "historically correlates with favorable forward returns"). | Every signal sentence now quotes its measured track record and a verdict based on the HAC t-stat. |

### Medium

| # | Finding | Resolution |
|---|---------|------------|
| M1 | Three threshold vocabularies (signal 20/40/60, card 20/40 "strong", chart 20/40). | One function, `classify_conviction`, used everywhere. |
| M2 | Feature-impact "history" only ever had 1 entry, and came from the PCA-OLS model only. | The ensemble's effective standardized coefficient (Σ weight × coefficient), signed, recorded at every refit and charted over time. |
| M3 | The θ-stability check compared 10 near-identical overlapping windows, so it always said "stable". | Rolling half-life chart, plus a flag when the current half-life is more than 2× away from the full-sample value. |
| M4 | Divergence was flagged only if it fell on the very last row, and the expanding std was back-filled. | Flagged if confirmed within the last 5 rows; causal std. |
| M5 | Percentage-text columns (`NIFTY50_EY` = "6.48%") were unusable. | Numeric text (`%`, thousands separators) is converted on load. |
| M6 | "R² vs random walk" compared a same-day fair value with yesterday's actual: not a forecast test. | Replaced by **Gap edge** and **Signal edge** (−IC with the next 20 rows, HAC t). |
| M7 | Target and date changes could only be applied from inside the collapsed predictors expander. | The Apply button sits outside the expander and lists every pending change. |
| M8 | Reset kept the 1-hour sheet cache, so a refreshed sheet didn't show up. | Reset clears the cache. |
| M9 | Warm-up "fair values" (expanding mean, y₀ for row 0) were mixed into the gap. | NaN until a model exists. |

### Low — operations

| # | Finding | Resolution |
|---|---------|------------|
| L1 | The app crashed at startup without `secrets.toml`: `st.secrets` ran before `set_page_config`. | `set_page_config` comes first; env var first; secrets read only if a file exists. |
| L2 | The devcontainer ran a non-existent `aarambh.py` and was named "v2.0". CHANGELOG 3.2.2 says this was fixed; it wasn't. | Runs `correl.py`; named v3.3.0. |
| L3 | README pointed to a missing `.env.example`; there was no `.gitignore`. | `.streamlit/secrets.toml.example` and `.gitignore` added. |
| L4 | No theme config, so the page rendered white around dark cards. The menu is hidden, so users couldn't switch. | `.streamlit/config.toml`, dark theme. |
| L5 | The selected timeframe button wasn't highlighted. | Highlighted. |
| L6 | One 2,850-line file mixing UI and maths; no tests. | `aarambh/` package (config, data, stats, engine) plus a UI-only `correl.py`; 44 tests. |

## 4. Before and after on the live sheet

| | v3.2.3 (+ date fix) | v3.3.0 |
|---|---|---|
| Rows used | 5,041 (1,738 with fabricated INIRYY) | 3,303 (from 12 Jun 2013, set by INIRYY) |
| z std across lookbacks | 0.49 – 1.61 | 0.98 – 1.06 |
| Days per regime | 78% neutral, 10% OS, 11% OB | 64% neutral, 19% OS, 17% OB |
| Displayed gap half-life | 2 d | 8 d (full sample 18 d) |
| OOS R² | 0.923 | 0.892 (no back-filled rows) |
| Break detection | silently off, "statsmodels missing" | removed; ADF/KPSS reported |
| Feature-impact history | 1 point | 657 refits |
| Signal track record | plain t on overlapping windows | HAC t, episodes, edge vs neutral |

## 5. What the system now says about itself

With the defects fixed, the system measures whether its signals have worked. On the live sheet, **no configuration shows a statistically reliable edge** at 5, 10 or 20 rows:

| Predictors / memory | OOS R² | Gap half-life | Gap edge (t) | Signal edge (t) | BUY 20-row edge (t) | SELL 20-row edge (t) |
|---|---|---|---|---|---|---|
| Sheet's 14 / adaptive (**default**) | 0.892 | 18 d | −0.02 (−0.39) | +0.03 (+0.53) | +0.84 pts (+1.06) | −0.66 pts (−0.74) |
| Sheet's 14 / long | 0.634 | 126 d | −0.14 (**−2.06**) | +0.03 (+0.39) | +1.02 pts (+1.13) | −0.96 pts (−1.39) |
| Without PB, DY / adaptive | 0.825 | 17 d | −0.00 (−0.04) | +0.01 (+0.21) | −0.65 pts (−0.68) | +0.43 pts (+0.67) |
| Without PB, DY / long | 0.540 | 153 d | +0.04 (+0.50) | +0.04 (+0.67) | +1.58 pts (+2.01) | −0.70 pts (−0.95) |

Edge = move in the signal's favour; positive is good; |t| < 2 is indistinguishable from noise. One cell reaches t ≈ 2 out of the many tested, which is about what chance alone would produce. Tuning settings to maximise these numbers on this same history would be data-snooping, so this audit doesn't. The tests confirm the measurement itself works: on synthetic data it finds a planted mean-reverting gap (t > 3) and finds no edge in a random-walk gap.

## 6. Decisions for the owner

1. **Price-sharing predictors.** NIFTY50_PB and NIFTY50_DY contain the same index price as NIFTY50_PE. When price rises the fair value rises with it, so the gap measures earnings-versus-book noise rather than mispricing. PB is the single largest driver (+0.74 PE per σ). The defaults still mirror the sheet's own SELECTOR = 1 choices; the app explains the issue next to the drivers chart.
2. **Model memory.** *Adaptive* (the default) makes the gap a short-lived tracking error with an 18-day half-life. *Long* makes it a long-run valuation gap with a 126-day half-life. The sidebar offers both. Your sheet's own COR. PE / PE_DEV works like *Long*.
3. **Horizons.** Validation uses 5/10/20 rows, as before. A valuation thesis may deserve 60–120 rows; extending `FORWARD_HORIZONS` in `aarambh/config.py` is a one-line change.
4. **INIRYY** starts in June 2013 and costs 1,738 rows. Deselecting it starts the analysis in January 2008, when the yields begin.
5. **Sheet typo.** About row 4274 is dated 28/08/2023 but sits between 27/09 and 29/09/2023. Its NIFTY value (19306.05) is the 28 Aug close. The app keeps the first 28/08/2023 row and warns you.

## 7. Reproduce

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q          # 44 tests, ≈20 s
streamlit run correl.py
```
