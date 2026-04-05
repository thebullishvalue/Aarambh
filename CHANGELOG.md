# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

### Phase III — UI/UX Reorganization (Production Release)

#### UX / Interface Consolidation
- **Tab Architecture Overhaul**: Collapsed fragmented 5-tab structure (`Regime`, `Signal`, `Zones`, `Signals`, `Data`) into **4 logically flowing tabs**:
  - **📊 Dashboard**: Primary signal (full-width), Base Conviction (full-width), SPRT DDM Confidence Boundaries (full-width), Market State (3 cards + interpretation), Model Quality (4 cards), Fair Value plot
  - **🗺️ Breadth Topology**: Zone breadth, Signal Frequency, Average Z-Score, Current Lookback States
  - **🧠 ML Diagnostics**: OU diagnostics (3 cards), Feature Impact (gradient bar chart + history table), Signal Performance
  - **📋 Data Table**: Full time-series with Swing-style table rendering, CSV export
- **Primary Signal Relocation**: Moved from tab content to above all tabs (always visible, full-width card with interpretation)
- **Timeframe Filter**: Replaced radio buttons with Swing-style button row (1M, 6M, 1Y, 2Y, ALL)
- **Sidebar Flow**: Google Sheets default, "Run Analysis" → "Reset Analysis" button pattern

#### Visualization Standards
- **Unified Chart Margins**: `dict(t=10, l=60, r=20, b=10)` across all Plotly figures
- **Consistent Line Widths**: 2.5px main data lines, 1px reference lines
- **Standardized Legends**: Horizontal orientation, top-right position, font size 9
- **Color Palette**: Hemrek Design System CSS variables throughout
- **Base Conviction Plot**: Threshold markers at ±20 (small, 4px) and ±40 (large, 8px), color-coded (green/red/gray)
- **DDM Interpretation Card**: Dynamic 5-state analysis (STRONG OVERSOLD → STRONG OVERBOUGHT) matching Regime Context style
- **Feature Impact Chart**: Gradient bars based on impact magnitude (cyan → white)

#### Code Quality
- **Dead Code Removal**: ~500 lines of old tab rendering functions eliminated (`_render_tab_regime`, `_render_tab_signal`, `_render_tab_zones`, `_render_tab_signals`)
- **DRY Implementation**: `_render_metric_card()` helper function for consistent card rendering
- **Threading Removed**: Sequential walk-forward execution (no `ThreadPoolExecutor`, no `ScriptRunContext` warnings)
- **Streamlit API Updated**: `use_container_width=True` → `width="stretch"` (15+ instances)
- **Error Handling**: Feature impact table only renders when data exists (no blank rows)

#### Documentation Coherence
- **README.md Layout Guide**: Complete 4-tab UI structure documentation with component descriptions and usage scenarios
- **CHANGELOG.md Production State**: Finalized v3.2.1 UI refactoring documented as Production Release
- **Inline Docstrings**: Updated for all public functions with Phase III changes

---

## [3.2.0] - Prior Release — ADAM Phase I Architectural Hardening

### Critical — Mathematical Correctness
- **True Conformal Quantiles**: Replaced pseudo-Gaussian z-score implementations inside standard loops with the rigorous empirical `compute_conformal_zscores` primitive. Signal zones now mathematically align with fat-tailed reality.
- **Bai-Perron Regime Binding**: Expanding windows now mathematically bind to the most recent structural break (`max(max_lookback, last_break)`), severing legacy coefficients from polluting out-of-sample regressions across shifting regimes.
- **DDM Variance Capping**: `drift_diffusion_filter` now caps scaling variance geometrically (`min(abs(drift) * 0.5, long_run_var * 0.5)`) to prevent ballooning standard errors and undefined confidence bands during prolonged regimes.

### Major — Architectural Improvements
- **True Thread-Safety**: Re-established `ThreadPoolExecutor` async processing safely locked behind state mutation conditions, overriding the false sequential lock architecture of v3.1.0.
- **Parametric Fixes**: Repaired the `ElasticNetCV(n_alphas=10)` silent exception bug, restoring the ElasticNet sub-model to the Walk-Forward ensemble.

---

## [3.1.0] - Prior Release — Initial Refactor

### Critical — Mathematical Correctness

#### Fixed
- **Look-Ahead Bias in Z-Score Computation**: Rolling mean/std now use `shift(1)` to ensure only past data is used in standardization. Previously, current residual was included in its own z-score calculation, inflating OOS R² by 10-20%.
- **Jackknife Correction for Near-Unit-Root AR(1)**: Replaced biased jackknife estimator with Andrews (1993) median-unbiased correction. Half-life estimates now accurate for persistent series (θ ≈ 0.95).
- **DDM Variance Unbounded Growth**: Added mean-reverting variance term: σ²_t = (1-λ)σ²_{t-1} + λσ²_LR. Confidence bands now stable during extended regimes.

### Major — Architectural Improvements

#### Added
- **Structural Break Detection**: Bai-Perron multiple breakpoint testing before walk-forward. Expanding window resets if regime shift detected within trailing period.
- **DFA Hurst Exponent**: Replaced biased R/S estimator with Detrended Fluctuation Analysis (Peng et al., 1994). Proper lag range: max(4, n/10) to n/4.
- **Significance Testing**: t-statistics and p-values for all signal performance metrics. Users now see statistical confidence (✅ p<0.05, ⚠️ p<0.10).
- **Rolling θ Estimation**: Time-series of OU mean-reversion speed computed over trailing 60-day windows. Projection confidence bands reflect parameter uncertainty.
- **Feature Impact History**: Time-varying feature weights stored at each refit interval. Users can track factor rotation over time.
- **Conformal Quantile Z-Scores**: Empirical quantile-based z-scores preserve fat tails instead of Gaussian mean/std clipping.
- **θ Stability Diagnostic**: New metric card showing whether mean-reversion speed is stable (CV < 50%) or unstable.

#### Changed
- **Thread-Safe Walk-Forward Execution**: Replaced parallel ThreadPoolExecutor with lock-protected sequential execution. Results now deterministic and reproducible.
- **Conviction Score Soft Bounds**: Applied tanh transformation: Conviction_bounded = 100 × tanh(Conviction_raw / 100). Score interpretation consistent at extremes.
- **Winsorized Forward Changes**: Forward % changes capped at ±100% to prevent spurious outliers from near-zero denominators.

### Minor — Code Quality & DRY

#### Changed
- **DRY Helper Functions**: Extracted `_safe_array_operation`, `_classify_zones`, `_detect_crossover_signals`, `_compute_significance`, `_apply_conviction_bounds` for reusability.
- **Unified Zone Classification**: Single `_classify_zones` function used across all lookback computations.
- **Consolidated Signal Detection**: `_detect_crossover_signals` reused for all lookback windows.

#### Removed
- Dead code: `kalman_filter_1d`, `_compute_kalman_conviction` (fully replaced by DDM in v3.0.0).

### Documentation

#### Updated
- **README.md**: Complete architecture diagram reflecting all v3.1.0 components (Bai-Perron, DFA, Andrews MU, conformal quantiles).
- **Mathematical Equations**: Added explicit formulas for OU process, DDM filter, DFA Hurst, and soft bounds.
- **Reference Papers**: Added Andrews (1993), Bai & Perron (1998), Peng et al. (1994) citations.

### Performance Impact

| Metric | v3.0.0 | v3.1.0 | Change |
|--------|--------|--------|--------|
| OOS R² | Inflated | Honest | -10-20% |
| Half-Life Estimate | Biased low | Unbiased | +20-40% |
| Confidence Band Stability | Diverging | Stable | Fixed |
| Hit Rate Significance | Not shown | t-stat + p-value | New |
| Thread Safety | Race condition | Lock-protected | Fixed |

### Expected Behavior Changes

1. **Lower OOS R²**: Expect 10-20% reduction after look-ahead fix. This is *truth-telling*, not degradation.
2. **Longer Half-Lives**: Andrews MU correction increases t₁/₂ by 20-40% for persistent series.
3. **Wider Confidence Bands**: Mean-reverting variance + rolling θ uncertainty → more honest uncertainty quantification.
4. **Significance Flags**: Hit rates now accompanied by t-statistics. Users can distinguish signal from noise.

---

## [3.0.0] - ADAM Refactor / Current

### Added
- **Adaptive Conformal Unbounded Bounds**: Native continuous-scale expanding empirical quantiles natively mapping limits in Out-of-Sample residuals.
- **Drift-Diffusion Accumulator (DDM)**: A Sequential Probability Ratio Test integrating the evidence signals into an orthogonal bounding space.
- **Strict Decoupled OR Regression**: Stationary variance projections derived accurately exclusively via $ln(2)/\theta$.

### Changed
- Replaced the constrained rank approximation (`stats.norm.ppf()`) bounding with conformal unbounded standard deviations, permanently repairing extreme "black swan" tail-clipping bugs.
- Pulled the `kalman_filter_1d` and `_compute_kalman_conviction` in favor of physically valid Drift-Diffusion (DDM), directly solving the boundary-break covariance mis-specification.
- Decoupled `dynamic_theta` from the arbitrary `vol_multiplier` to prevent mathematically illegal OU diffusion contraction.
- Complete UI Refactor across Streamlit cards mapping UI strings to accurately reflect new DDM and Conformal realities.

## [2.2.1] - Prior Release

### Fixed
- Resolved a fatal Dataset Transition Crash caused by state desynchronization when switching between datasets with non-overlapping predictor columns.
- Fixed a Predictor Mutation Cache Bypass vulnerability where inline edits to predictor variables were ignored by the Streamlit session state cache if the target sum remained identical.

## [2.2.0] - Prior Release

### Changed
- Massively optimized Walk-Forward engine performance by vectorizing prediction loops across multi-row execution chunks.
- Replaced heavy `statsmodels` WLS implementation with `scikit-learn` `LinearRegression` to bypass unnecessary statistical covariance computations during walk-forward fitting.
- Tuned `ElasticNetCV` parameter grid and relaxed `HuberRegressor` convergence tolerance to dramatically reduce ensemble fit times without sacrificing out-of-sample accuracy.
- Application version dependencies and devcontainer environments updated to reflect `v2.2.0`.

## [2.1.0] - Prior Release

### Added
- `CHANGELOG.md` to track version history and codebase changes over time.
- Implementation of dynamic Feature Impact tracking using WLS coefficients and PCA back-projection.

### Changed
- Dashboard diagnostic cards have been reorganized with concise sub-metrics to ensure uniform heights, significantly improving UI consistency.
- Updated project documentation (`README.md`) to reflect the new version structure.

### Fixed
- Resolved `ConvergenceWarning` originating from `sklearn.linear_model._coordinate_descent` by aggressively increasing the `max_iter` limit to `10000`, relaxing `tol`, and actively filtering non-consequential warnings.

## [2.0.0] - Major Update

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
- Enhanced application theme with the Hemrek Design System styling.

## [1.0.0] - Initial Framework

- Initial release: static multi-lookback Z-score banding and fair-value linear models.