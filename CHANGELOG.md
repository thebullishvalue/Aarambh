# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.3.0] - 2026-11-10 — System Audit

Full audit of idea, implementation and output against the live source sheet. Evidence for every item is in `AUDIT.md`. Signals change materially: conviction thresholds now sit on a calibrated scale, and the history no longer contains back-filled data.

### Fixed — Critical
- **Date Parsing (DD/MM/YYYY)**: `pd.to_datetime()` inferred MM/DD/YYYY from the sheet's first date (`08/06/2006`), so 3,054 of 5,042 rows were dropped and the rest scrambled. `parse_dates()` picks one format for the whole column. Repeated dates keep their first row and are named in a warning.
- **z-Score Calibration**: z divided the 2.5–97.5 percentile range by 1.35 (the 25–75 IQR constant), so the std of z ranged from 1.61 (5-day) to 0.49 (100-day) and "Overvalued" meant ≈2.9σ. Replaced by a robust gap score (median, IQR/1.349) ranked against its own past (`calibrated_zscores`): std ≈ 1.0 and ≈32% / ≈5% of days beyond ±1 / ±2 in every lookback.
- **Look-Ahead via Back-Fill**: `clean_data` back-filled predictors, so INIRYY's first value (June 2013) filled 34.5% of history and 2008 yields filled 2006–07. Now forward-fill only; rows before the latest-starting predictor are trimmed and the app says which one set the start.
- **In-Sample Ensemble Weights**: "out-of-sample" inverse-MAE weights were computed on rows inside the training window. Now each model's true out-of-sample errors over the previous 60 rows, with a 5% floor on normalised weights.
- **OU Bias Correction Sign**: the AR(1) correction subtracted (1+3a)/n instead of adding it, doubling the downward bias; the headline used a 60-row window (displayed half-life 2 d). Now Kendall's correction, exact σ for every a, current = trailing 252 rows with the full sample alongside.

### Fixed — High
- **Break Detection Never Ran**: it imported a statsmodels module that does not exist, which also made the UI report "statsmodels missing". The design applied full-sample breaks to earlier dates (look-ahead). Removed.
- **Overlapping-Window t-Stats**: signal performance used plain t-tests on overlapping 5/10/20-row changes, and a different rule from the live signal. Now Newey-West HAC regressions on the live BUY/SELL rule, with episodes, edge vs neutral days and hit rate.
- **OU Projection Band**: it covered θ uncertainty only and could explode. Now the OU predictive band.
- **DFA**: applied to residual levels over n/10…n/4 and forced to 0.5 when ADF p > 0.05. Now DFA-1 on the gap's daily changes with log-spaced scales 10…n/4.
- **DDM**: the code did not match its documented equation (gain 1.875), was seeded from the next 20 observations, and used an ad-hoc variance. Now a unit-gain leaky accumulator starting at the first observation, with a real standard error band.
- **Reproducibility**: the refit interval depended on total sample length (`n // 200`) and ElasticNet ran unseeded random descent. Now a fixed `REFIT_INTERVAL = 5` and `random_state=0`.
- **Unfounded Claims**: interpretation text asserted historical performance that was never computed. Every signal sentence now quotes its measured track record.

### Fixed — Medium & Operations
- One threshold vocabulary (`classify_conviction`) for signal, regime, cards and charts.
- Feature-impact history had one entry; it now records the ensemble's effective standardized coefficients at every refit and charts them.
- θ "stability" compared near-identical overlapping windows (always stable); replaced by a rolling half-life chart and a 2× consistency flag.
- Divergence badge only checked the last row; it now covers the last 5. Causal expanding std.
- Percentage and thousands-separator text columns (e.g. `NIFTY50_EY`) are converted to numbers.
- "R² vs random walk" (same-day fair value vs yesterday's actual) replaced by Gap edge and Signal edge with HAC t.
- Apply Configuration moved out of the predictors expander and lists every pending change. Reset Analysis clears the sheet cache.
- Warm-up rows have no fair value instead of an expanding mean.
- The app crashed at startup without `secrets.toml`; `set_page_config` now comes first, and secrets are read only if a file exists.
- The devcontainer ran a non-existent `aarambh.py`; it now runs `correl.py`.
- Added `.streamlit/config.toml` (dark theme; the menu is hidden so users could not switch), `.streamlit/secrets.toml.example`, `.gitignore`. The selected timeframe is highlighted.

### Added
- **Model memory** option: Adaptive (500 rows, 252-row half-life weights; default) or Long (all history, equal weights).
- **Track record** on the signal card and interpretation card, plus Gap edge and Signal edge cards.
- Notices for every change the cleaner makes (repeated dates, unreadable dates, missing targets, trimmed rows and why).
- A flag when predictors share the target's index price (e.g. NIFTY50_PB / NIFTY50_DY for NIFTY50_PE).
- `tests/`: 44 tests including no look-ahead (truncated refit reproduces the past exactly) and planted-edge detection.

### Changed
- Split into the `aarambh/` package (`config`, `data`, `stats`, `engine`) and a UI-only `correl.py` (still the entry point).
- `clean_data()` returns `(data, CleanReport)`.
- Names match the maths: Kendall (not Andrews) bias correction, leaky accumulator (not SPRT), calibrated (conformal) z-scores.

---

## [3.2.3] - 2026-04-06 — Production Cleanup & Security Hardening

### Removed
- **Dead Functions**: Removed `andrews_median_unbiased_ar1()` (40 lines) and `_safe_array_operation()` (28 lines) — zero traceable references via full import/call-flow tracing
- **Dead Imports**: Removed unused `TimeSeriesSplit` import from `sklearn.model_selection`
- **Dead Code Branches**: Eliminated unused `inline` parameter branch in `_render_metric_card()`, unreachable `_render_header()` call in landing page path, and unused `vol_multiplier` variable in OU diagnostics
- **Security Risk**: Deleted `.streamlit/secrets.toml` containing live Google Sheets URL — secrets now exclusively managed via Streamlit Cloud secrets or environment variables

### Fixed
- **Trailing Commas**: Cleaned up 10 trailing comma artifacts in `st.plotly_chart()` and `st.button()` calls left over from `use_container_width` removal
- **Misleading Comment**: Corrected docstring in `_compute_multi_lookback_signals()` that referenced `shift(1)` when actual implementation uses backward-looking window slicing
- **Unused Assignment**: Removed `sm = None` from statsmodels import fallback — variable never referenced, only `_HAS_STATSMODELS` flag is checked

### Changed
- **Import Cleanup**: Renamed `statsmodels.api as sm` to `statsmodels.api as _sm` to clarify it's only used for side-effect imports (functions are imported directly)
- **Secrets Management**: `DEFAULT_SHEET_URL` now reads from Streamlit secrets → environment variable → empty string (no hardcoded fallback)
- **Codebase Size**: Reduced from 2,859 to ~2,716 lines (143 lines of dead code removed)

---

## [3.2.2] - 2026-04-05 — Production Deployment Hardening

### Fixed
- **Devcontainer Configuration**: Corrected container name from `v2.0` to `v3.2.2` and fixed entry point script from `aarambh.py` to `correl.py`, ensuring consistent development environment setup
- **Dynamic Version Logging**: Replaced hardcoded `v3.2.1` in engine startup log message with dynamic `VERSION` constant, preventing version display inconsistencies across releases

### Changed
- **Production Metadata**: Updated version references across `requirements.txt`, `LICENSE.md`, and documentation to reflect current release state

---

## [3.2.1] - 2026-04-01 — ADAM Phase II Logic Guard Hardening + Phase III Production Release

### Critical — Mathematical Consistency
- **Dynamic Ensemble Inverse-MAE Weights**: Eradicated static simple averages across walk-forward predictions. Ensemble constituents are now dynamically weighted using an inverse-MAE schema built on OOS trailing validation sets, featuring a 5% baseline preservation floor.
- **Double-Bounding Singularity**: `ConvictionScore` math natively applies and exports `ConvictionBounded`, strictly enforcing identical numerical states between the underlying logic loops and the display UI.

### Major — Temporal Stability
- **DDM State Initialization Bias**: `drift_diffusion_filter` now initializes transient state arrays utilizing rolling forward sums of the initial 20 observations, eliminating early systemic drag.
- **Temporal Record Alignment**: Feature impact matrix updates structurally map directly to block end times `t_index`, rather than global arbitrary limits.
- **OU Dynamics**: Projection bands override full sample estimations dynamically with trailing `theta_history[-1]` values, reflecting localized confidence bounds exactly.
- **Hurst Stationarity Guard**: Implemented ADF tests prior to DFA Hurst invocations. If signals diverge violently from base stationarity, deterministic bounds return `H=0.5` instead of hallucinating fractional trend structures.

### UI/UX Reorganization

#### Architecture
- **Tab Overhaul**: Collapsed fragmented 5-tab structure into **4 logically flowing tabs**:
  - **Dashboard**: Primary signal, Base Conviction, DDM Confidence, Market State, Model Quality, Fair Value plot
  - **Breadth Topology**: Zone breadth, Signal Frequency, Average Z-Score, Current Lookback States
  - **ML Diagnostics**: OU diagnostics, Feature Impact, Signal Performance
  - **Data Table**: Full time-series with CSV export
- **Primary Signal Relocation**: Moved from tab content to above all tabs (always visible, full-width card)
- **Timeframe Filter**: Replaced radio buttons with Swing-style button row (1M, 6M, 1Y, 2Y, ALL)

#### Visualization Standards
- **Unified Chart Margins**: `dict(t=10, l=60, r=20, b=10)` across all Plotly figures
- **Consistent Line Widths**: 2.5px main data lines, 1px reference lines
- **Standardized Legends**: Horizontal orientation, top-right position, font size 9
- **Base Conviction Plot**: Threshold markers at ±20 (small) and ±40 (large), color-coded
- **DDM Interpretation Card**: Dynamic 5-state analysis matching Regime Context style

#### Code Quality
- **Dead Code Removal**: ~500 lines of old tab rendering functions eliminated
- **DRY Implementation**: `_render_metric_card()` helper for consistent card rendering
- **Threading Removed**: Sequential walk-forward execution (no ThreadPoolExecutor warnings)
- **Streamlit API Updated**: `use_container_width=True` → `width="stretch"` (15+ instances)

---

## [3.2.0] - 2026 — ADAM Phase I Architectural Hardening

### Critical — Mathematical Correctness
- **True Conformal Quantiles**: Replaced pseudo-Gaussian z-score implementations with the rigorous empirical `compute_conformal_zscores` primitive. Signal zones now mathematically align with fat-tailed reality.
- **Bai-Perron Regime Binding**: Expanding windows now mathematically bind to the most recent structural break (`max(max_lookback, last_break)`), severing legacy coefficients from polluting out-of-sample regressions across shifting regimes.
- **DDM Variance Capping**: `drift_diffusion_filter` now caps scaling variance geometrically (`min(abs(drift) * 0.5, long_run_var * 0.5)`) to prevent ballooning standard errors during prolonged regimes.

### Added
- **True Thread-Safety**: Re-established `ThreadPoolExecutor` async processing safely locked behind state mutation conditions.

### Fixed
- **ElasticNetCV Silent Exception**: Repaired the `ElasticNetCV(n_alphas=10)` silent exception bug, restoring the ElasticNet sub-model to the Walk-Forward ensemble.

---

## [3.1.0] - 2026 — Initial Refactor

### Fixed
- **Look-Ahead Bias in Z-Score Computation**: Rolling mean/std now use `shift(1)` to ensure only past data is used in standardization. Previously, current residual was included in its own z-score calculation, inflating OOS R² by 10–20%.
- **Jackknife Correction for Near-Unit-Root AR(1)**: Replaced biased jackknife estimator with Andrews (1993) median-unbiased correction. Half-life estimates now accurate for persistent series (θ ≈ 0.95).
- **DDM Variance Unbounded Growth**: Added mean-reverting variance term: σ²_t = (1-λ)σ²_{t-1} + λσ²_LR. Confidence bands now stable during extended regimes.

### Added
- **Structural Break Detection**: Bai-Perron multiple breakpoint testing before walk-forward. Expanding window resets if regime shift detected within trailing period.
- **DFA Hurst Exponent**: Replaced biased R/S estimator with Detrended Fluctuation Analysis (Peng et al., 1994). Proper lag range: max(4, n/10) to n/4.
- **Significance Testing**: t-statistics and p-values for all signal performance metrics.
- **Rolling θ Estimation**: Time-series of OU mean-reversion speed computed over trailing 60-day windows.
- **Feature Impact History**: Time-varying feature weights stored at each refit interval.
- **Conformal Quantile Z-Scores**: Empirical quantile-based z-scores preserve fat tails.
- **θ Stability Diagnostic**: New metric card showing whether mean-reversion speed is stable (CV < 50%).

### Changed
- **Thread-Safe Walk-Forward Execution**: Replaced parallel ThreadPoolExecutor with lock-protected sequential execution.
- **Conviction Score Soft Bounds**: Applied tanh transformation: Conviction_bounded = 100 × tanh(Conviction_raw / 100).
- **Winsorized Forward Changes**: Forward % changes capped at ±100% to prevent spurious outliers.
- **DRY Helper Functions**: Extracted `_safe_array_operation`, `_classify_zones`, `_detect_crossover_signals`, `_compute_significance`, `_apply_conviction_bounds`.

### Removed
- Dead code: `kalman_filter_1d`, `_compute_kalman_conviction` (fully replaced by DDM).

### Performance Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| OOS R² | Inflated | Honest | -10–20% |
| Half-Life Estimate | Biased low | Unbiased | +20–40% |
| Confidence Band Stability | Diverging | Stable | Fixed |
| Thread Safety | Race condition | Lock-protected | Fixed |

---

## [3.0.0] - 2026 — ADAM Refactor

### Added
- **Adaptive Conformal Unbounded Bounds**: Native continuous-scale expanding empirical quantiles natively mapping limits in Out-of-Sample residuals.
- **Drift-Diffusion Accumulator (DDM)**: A Sequential Probability Ratio Test integrating the evidence signals into an orthogonal bounding space.
- **Strict Decoupled OU Regression**: Stationary variance projections derived accurately exclusively via ln(2)/θ.

### Changed
- Replaced the constrained rank approximation (`stats.norm.ppf()`) bounding with conformal unbounded standard deviations, permanently repairing extreme "black swan" tail-clipping bugs.
- Removed `kalman_filter_1d` and `_compute_kalman_conviction` in favor of physically valid Drift-Diffusion (DDM), directly solving the boundary-break covariance mis-specification.
- Decoupled `dynamic_theta` from the arbitrary `vol_multiplier` to prevent mathematically illegal OU diffusion contraction.
- Complete UI refactor across Streamlit cards mapping UI strings to accurately reflect new DDM and Conformal realities.

---

## [2.2.1] - Prior Release

### Fixed
- **Dataset Transition Crash**: Resolved state desynchronization when switching between datasets with non-overlapping predictor columns.
- **Predictor Mutation Cache Bypass**: Fixed inline edits to predictor variables being ignored by Streamlit session state cache if the target sum remained identical.

---

## [2.2.0] - Prior Release

### Changed
- **Vectorized Walk-Forward Engine**: Massively optimized performance by vectorizing prediction loops across multi-row execution chunks.
- **sklearn LinearRegression Swap**: Replaced heavy `statsmodels` WLS implementation with `scikit-learn` `LinearRegression`.
- **ElasticNetCV Tuning**: Repaired parameter grid and relaxed `HuberRegressor` convergence tolerance to dramatically reduce ensemble fit times.
- Updated devcontainer environments to reflect v2.2.0.

---

## [2.1.0] - Prior Release

### Added
- `CHANGELOG.md` to track version history.
- Dynamic Feature Impact tracking using WLS coefficients and PCA back-projection.

### Changed
- Reorganized dashboard diagnostic cards with concise sub-metrics for uniform heights.

### Fixed
- **ConvergenceWarning**: Resolved `sklearn.linear_model._coordinate_descent` warning by increasing `max_iter` to `10000`, relaxing `tol`, and filtering non-consequential warnings.

---

## [2.0.0] - Prior Release — Major Update

### Added
- Walk-forward expanding-window ensemble regression (Ridge + Huber + OLS) for Fair Value estimation.
- Ornstein-Uhlenbeck (OU) mean-reversion parameter estimation on OOS residuals.
- Kalman-filtered breadth conviction scoring with 95% confidence bands.
- Residual Hurst Exponent to validate mean-reversion empirical properties.
- Swing-based divergences detection using local extrema logic.
- "Apply & Compute" staging system to prevent heavy engine recalculations during feature selection.
- Data staleness warnings.

### Changed
- Transitioned from in-sample baseline metrics to strictly out-of-sample (OOS) validation framework.
- Enhanced application theme with the @thebullishvalue Design System styling.

---

## [1.0.0] - Initial Release

### Added
- Static multi-lookback Z-score banding and fair-value linear models.
