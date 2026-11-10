"""FairValueEngine: walk-forward fair value → valuation gap → multi-lookback breadth →
conviction → signal, plus diagnostics and an honest record of how the signal has done.

Causality contract: every column of ``ts_data`` except ``FwdChg_*`` uses only data up
to and including its own row. ``tests/test_engine.py`` checks this by changing the
future and asserting the past is unchanged. Whole-sample summaries (OOS R², full-sample
half-life, DFA α, validation statistics) are labelled as such.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNetCV, HuberRegressor, LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

from .config import (
    CALIBRATION_MIN, CONVICTION_MODERATE, CONVICTION_STRONG, CONVICTION_WEAK,
    DEFAULT_MODEL_MEMORY, ENSEMBLE_WEIGHT_FLOOR, ENSEMBLE_WEIGHT_MIN_OBS,
    ENSEMBLE_WEIGHT_WINDOW, FORWARD_HORIZONS, HEADLINE_HORIZON, HUBER_EPSILON,
    HUBER_MAX_ITER, LOOKBACK_WINDOWS, MIN_TRAIN_SIZE, MODEL_MEMORY, OU_HISTORY_STEP,
    OU_PROJECTION_DAYS, OU_WINDOW, REFIT_INTERVAL, RIDGE_ALPHAS, VERSION, Z_EXTREME,
    Z_THRESHOLD,
)
from .stats import (
    calibrated_zscores, classify_zones, count_episodes, detect_crossover_signals,
    drift_diffusion_filter, half_life, hurst_dfa, information_coefficient,
    newey_west_ols, ornstein_uhlenbeck_estimate, ou_forecast, soft_bound,
)

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:  # optional: only the ADF/KPSS diagnostics need it
    HAS_STATSMODELS = False

log = logging.getLogger("aarambh")

MODEL_NAMES = ("ridge", "huber", "elasticnet", "pca_ols")


def classify_conviction(score: float) -> tuple[str, str, str]:
    """One vocabulary for the whole app: bounded conviction → (signal, strength, regime)."""
    if not np.isfinite(score):
        return "HOLD", "NEUTRAL", "N/A"
    a = abs(score)
    side_sig, side_reg = ("BUY", "OVERSOLD") if score < 0 else ("SELL", "OVERBOUGHT")
    if a <= CONVICTION_WEAK:
        return "HOLD", "NEUTRAL", "NEUTRAL"
    strength = "STRONG" if a > CONVICTION_STRONG else "MODERATE" if a > CONVICTION_MODERATE else "WEAK"
    regime = f"STRONGLY {side_reg}" if a > CONVICTION_STRONG else side_reg
    return side_sig, strength, regime


@dataclass
class _FittedEnsemble:
    predict: dict[str, Callable[[np.ndarray], np.ndarray]]
    coefs: dict[str, np.ndarray]   # standardized coefficients, full feature length


class FairValueEngine:
    """Walk-forward fair value engine with multi-lookback breadth analytics."""

    def __init__(self, memory: str = DEFAULT_MODEL_MEMORY, lookbacks: tuple[int, ...] = LOOKBACK_WINDOWS) -> None:
        if memory not in MODEL_MEMORY:
            raise ValueError(f"memory must be one of {sorted(MODEL_MEMORY)}")
        self.memory = memory
        self.max_train = MODEL_MEMORY[memory]["max_train"]
        self.decay_half_life = MODEL_MEMORY[memory]["decay_half_life"]
        self.lookbacks = tuple(lookbacks)
        self.active_lookbacks: tuple[int, ...] = ()
        self.ts_data = pd.DataFrame()
        self.model_stats: dict = {}
        self.ou_params: dict = {}
        self.ou_projection = np.array([])
        self.ou_projection_upper = np.array([])
        self.ou_projection_lower = np.array([])
        self.theta_history: list[dict] = []
        self.hurst = float("nan")
        self.latest_feature_impacts: dict[str, float] = {}
        self.latest_feature_coefs: dict[str, float] = {}
        self.feature_impact_history: list[dict] = []
        self.ensemble_weight_history: list[dict] = []
        self.validation: dict = {}
        self.residual_stats: dict = {}

    # ── Public API ────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None,
            progress_callback: Callable[[float, str], None] | None = None) -> FairValueEngine:
        t_start = time.time()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X must be 2-D with one row per target value")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("X and y must be finite; clean the data first (aarambh.data.clean_data)")
        self.feature_names = list(feature_names) if feature_names else [f"X{i}" for i in range(X.shape[1])]
        self.n_samples = len(y)
        self.y = y.copy()
        progress = progress_callback or (lambda *_: None)

        log.info("AARAMBH v%s — %d observations, %d features, memory=%s",
                 VERSION, self.n_samples, len(self.feature_names), self.memory)
        self._walk_forward_regression(X, y, progress)
        self.residuals = y - self.predictions

        progress(0.97, "Computing valuation-gap signals…")
        self.ts_data = pd.DataFrame({
            "Actual": self.y, "FairValue": self.predictions,
            "Residual": self.residuals, "ModelSpread": self.model_spread,
        })
        self._compute_model_stats()
        self._compute_multi_lookback_signals()
        self._compute_breadth_metrics()
        self._compute_conviction()
        self._find_pivots()
        self._compute_divergences()
        self._compute_forward_changes()
        self._compute_ou_diagnostics()
        self._compute_hurst()
        self._compute_validation()
        progress(1.0, "Done.")
        log.info("  ✓ Engine ready in %.1fs | OOS R² %.3f | half-life %.0fd | DFA α %.2f",
                 time.time() - t_start, self.model_stats.get("r2_oos", float("nan")),
                 self.ou_params.get("half_life", float("nan")), self.hurst)
        return self

    def get_current_signal(self) -> dict:
        """The latest observation's signal, with the measured record of that signal type."""
        if self.ts_data.empty:
            return {"signal": "HOLD", "strength": "NEUTRAL", "confidence": "N/A", "regime": "N/A",
                    "conviction_score": float("nan"), "warming_up": True}
        cur = self.ts_data.iloc[-1]
        score = float(cur["ConvictionBounded"])
        signal, strength, regime = classify_conviction(score)
        os_b, ob_b = float(cur["OversoldBreadth"]), float(cur["OverboughtBreadth"])
        if signal == "BUY":
            confidence = "HIGH" if os_b >= 80 else "MEDIUM" if os_b >= 60 else "LOW"
        elif signal == "SELL":
            confidence = "HIGH" if ob_b >= 80 else "MEDIUM" if ob_b >= 60 else "LOW"
        elif np.isfinite(score):
            confidence = "HIGH" if abs(score) < 10 else "MEDIUM"
        else:
            confidence = "N/A"

        recent = self.ts_data.iloc[-5:]
        record = self.validation.get("by_horizon", {}).get(HEADLINE_HORIZON, {}).get(signal.lower())
        return {
            "signal": signal, "strength": strength, "confidence": confidence, "regime": regime,
            "warming_up": not np.isfinite(score),
            "conviction_score": score,
            "conviction_upper": float(cur["ConvictionUpper"]),
            "conviction_lower": float(cur["ConvictionLower"]),
            "oversold_breadth": os_b, "overbought_breadth": ob_b,
            "residual": float(cur["Residual"]), "fair_value": float(cur["FairValue"]),
            "actual": float(cur["Actual"]), "avg_z": float(cur["AvgZ"]),
            "model_spread": float(cur["ModelSpread"]),
            "has_bullish_div": bool(recent["BullishDiv"].any()),
            "has_bearish_div": bool(recent["BearishDiv"].any()),
            "ou_half_life": self.ou_params.get("half_life", float("nan")),
            "adf_pvalue": self.ou_params.get("adf_pvalue", float("nan")),
            "kpss_pvalue": self.ou_params.get("kpss_pvalue", float("nan")),
            "hurst": self.hurst,
            "theta_consistent": self.ou_params.get("theta_consistent", True),
            "record": record,
        }

    def get_model_stats(self) -> dict:
        return self.model_stats

    def get_regime_stats(self) -> dict:
        reg = self.ts_data["Regime"]
        counts = reg[reg != "N/A"].value_counts()
        return {
            "strongly_oversold": int(counts.get("STRONGLY OVERSOLD", 0)),
            "oversold": int(counts.get("OVERSOLD", 0)),
            "neutral": int(counts.get("NEUTRAL", 0)),
            "overbought": int(counts.get("OVERBOUGHT", 0)),
            "strongly_overbought": int(counts.get("STRONGLY OVERBOUGHT", 0)),
            "n_valid": int(counts.sum()),
            "current_regime": reg.iloc[-1],
        }

    def get_signal_performance(self) -> dict:
        return self.validation.get("by_horizon", {})

    def get_feature_impact_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.feature_impact_history)

    # ── Walk-forward regression ───────────────────────────────────────────

    def _training_weights(self, n_train: int) -> np.ndarray:
        if self.decay_half_life is None:
            return np.ones(n_train)
        age = np.arange(n_train - 1, -1, -1, dtype=float)
        return np.exp(-np.log(2.0) / self.decay_half_life * age)

    def _walk_forward_regression(self, X: np.ndarray, y: np.ndarray, progress) -> None:
        """Refit every REFIT_INTERVAL rows on data strictly before the block; predict the block.

        The first MIN_TRAIN_SIZE rows have no model yet, so their fair value is NaN.
        """
        n, k = X.shape
        self.predictions = np.full(n, np.nan)
        self.model_spread = np.full(n, np.nan)
        self.model_oos = {m: np.full(n, np.nan) for m in MODEL_NAMES}
        self.feature_impact_history, self.ensemble_weight_history = [], []
        starts = list(range(MIN_TRAIN_SIZE, n, REFIT_INTERVAL))
        coef = np.zeros(k)

        for i, t0 in enumerate(starts):
            t1 = min(t0 + REFIT_INTERVAL, n)
            lo = 0 if self.max_train is None else max(0, t0 - self.max_train)
            fitted = self._fit_ensemble(X[lo:t0], y[lo:t0])
            preds = {}
            for name, fn in fitted.predict.items():
                p = fn(X[t0:t1])
                if np.all(np.isfinite(p)) and np.all(np.abs(p) < 1e10):
                    preds[name] = p
                    self.model_oos[name][t0:t1] = p
            if preds:
                w = self._ensemble_weights(y, t0, list(preds))
                P = np.vstack([preds[m] for m in preds])
                self.predictions[t0:t1] = np.array([w[m] for m in preds]) @ P
                self.model_spread[t0:t1] = P.std(axis=0) if len(preds) > 1 else 0.0
                coef = sum(w[m] * fitted.coefs[m] for m in preds)
                self.ensemble_weight_history.append({"index": t0, **w})
            else:
                self.predictions[t0:t1] = float(np.mean(y[lo:t0]))
                self.model_spread[t0:t1] = 0.0
            self._record_feature_impacts(coef, t0)
            if (i + 1) % max(1, len(starts) // 40) == 0 or i + 1 == len(starts):
                progress(0.95 * (i + 1) / len(starts), f"Walking forward… ({t1}/{n} observations)")

    def _fit_ensemble(self, X_train: np.ndarray, y_train: np.ndarray) -> _FittedEnsemble:
        """Ridge + Huber + ElasticNet + PCA→WLS on standardized, recency-weighted data."""
        k = X_train.shape[1]
        valid = np.std(X_train, axis=0) > 1e-8
        if not valid.any():
            return _FittedEnsemble({}, {})
        scaler = StandardScaler().fit(X_train[:, valid])
        Xs = scaler.transform(X_train[:, valid])
        sw = self._training_weights(len(y_train))

        def full(c: np.ndarray) -> np.ndarray:
            out = np.zeros(k)
            out[valid] = c
            return out

        def wrap(model, transform=None):
            def predict(X_new: np.ndarray) -> np.ndarray:
                Z = scaler.transform(X_new[:, valid])
                return model.predict(transform(Z) if transform else Z)
            return predict

        predict, coefs = {}, {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            try:
                m = RidgeCV(alphas=list(RIDGE_ALPHAS)).fit(Xs, y_train, sample_weight=sw)
                predict["ridge"], coefs["ridge"] = wrap(m), full(m.coef_)
            except Exception as exc:
                log.debug("ridge fit failed: %s", exc)
            try:
                m = HuberRegressor(epsilon=HUBER_EPSILON, max_iter=HUBER_MAX_ITER, tol=1e-3).fit(Xs, y_train, sample_weight=sw)
                predict["huber"], coefs["huber"] = wrap(m), full(m.coef_)
            except Exception as exc:
                log.debug("huber fit failed: %s", exc)
            try:
                m = ElasticNetCV(l1_ratio=[0.5, 0.9, 1.0], n_alphas=10, cv=2, max_iter=2000, tol=1e-2,
                                 selection="random", random_state=0, n_jobs=1).fit(Xs, y_train, sample_weight=sw)
                predict["elasticnet"], coefs["elasticnet"] = wrap(m), full(m.coef_)
            except Exception as exc:
                log.debug("elasticnet fit failed: %s", exc)
            try:
                pca = PCA(n_components=0.95, svd_solver="full").fit(Xs)
                ols = LinearRegression().fit(pca.transform(Xs), y_train, sample_weight=sw)
                predict["pca_ols"], coefs["pca_ols"] = wrap(ols, pca.transform), full(ols.coef_ @ pca.components_)
            except Exception as exc:
                log.debug("pca/ols fit failed: %s", exc)
        return _FittedEnsemble(predict, coefs)

    def _ensemble_weights(self, y: np.ndarray, t0: int, names: list[str]) -> dict[str, float]:
        """Inverse-MAE weights from each model's own *out-of-sample* errors before t0.

        Each weight is at least ENSEMBLE_WEIGHT_FLOOR; equal weights until every model
        has ENSEMBLE_WEIGHT_MIN_OBS past out-of-sample predictions.
        """
        lo = max(0, t0 - ENSEMBLE_WEIGHT_WINDOW)
        inv = {}
        for m in names:
            err = np.abs(y[lo:t0] - self.model_oos[m][lo:t0])
            err = err[np.isfinite(err)]
            if len(err) < ENSEMBLE_WEIGHT_MIN_OBS:
                return {name: 1.0 / len(names) for name in names}
            inv[m] = 1.0 / max(float(err.mean()), 1e-9)
        total = sum(inv.values())
        floor = min(ENSEMBLE_WEIGHT_FLOOR, 1.0 / len(names))
        return {m: floor + (1.0 - floor * len(names)) * inv[m] / total for m in names}

    def _record_feature_impacts(self, coef: np.ndarray, t_index: int) -> None:
        """Ensemble's effective standardized coefficient per feature, as % of the total."""
        total = float(np.sum(np.abs(coef)))
        if total < 1e-12:
            return
        pct = np.abs(coef) / total * 100.0
        self.feature_impact_history.append({"index": t_index, **dict(zip(self.feature_names, pct.tolist()))})
        order = np.argsort(-pct)
        self.latest_feature_impacts = {self.feature_names[j]: float(pct[j]) for j in order}
        self.latest_feature_coefs = {self.feature_names[j]: float(coef[j]) for j in order}

    # ── Analytics ─────────────────────────────────────────────────────────

    def _compute_model_stats(self) -> None:
        """Whole-sample out-of-sample fit of the fair value to the target."""
        m = np.isfinite(self.predictions)
        y_v, p_v = self.y[m], self.predictions[m]
        ss_res = float(np.sum((y_v - p_v) ** 2))
        ss_tot = float(np.sum((y_v - y_v.mean()) ** 2)) if len(y_v) else 0.0
        self.model_stats = {
            "r2_oos": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
            "rmse_oos": float(np.sqrt(np.mean((y_v - p_v) ** 2))) if len(y_v) else float("nan"),
            "mae_oos": float(np.mean(np.abs(y_v - p_v))) if len(y_v) else float("nan"),
            "n_obs": int(m.sum()),
            "n_features": len(self.feature_names),
            "avg_model_spread": float(np.nanmean(self.model_spread[m])) if m.any() else float("nan"),
            "memory": self.memory,
        }

    def _compute_multi_lookback_signals(self) -> None:
        n_oos = int(np.isfinite(self.residuals).sum())
        self.active_lookbacks = tuple(lb for lb in self.lookbacks if n_oos > lb + CALIBRATION_MIN)
        for lb in self.active_lookbacks:
            z = calibrated_zscores(self.residuals, lb, CALIBRATION_MIN)
            buy, sell = detect_crossover_signals(z)
            self.ts_data[f"Z_{lb}"] = z
            self.ts_data[f"Zone_{lb}"] = classify_zones(z)
            self.ts_data[f"Buy_{lb}"] = buy
            self.ts_data[f"Sell_{lb}"] = sell

    def _compute_breadth_metrics(self) -> None:
        """Share of lookbacks flagging each zone, and raw conviction.

        Conviction_raw = (OB% − OS%) + 0.5 × (ExtremeOB% − ExtremeOS%), in [−150, 150].
        Undefined until every active lookback has produced its first z-score.
        """
        n = len(self.ts_data)
        if not self.active_lookbacks:
            for col in ("OversoldBreadth", "OverboughtBreadth", "ExtremeOversold", "ExtremeOverbought",
                        "AvgZ", "ConvictionRaw"):
                self.ts_data[col] = np.full(n, np.nan)
            self.ts_data["BuySignalBreadth"] = 0
            self.ts_data["SellSignalBreadth"] = 0
            return
        Z = np.vstack([self.ts_data[f"Z_{lb}"].to_numpy() for lb in self.active_lookbacks])
        valid = np.isfinite(Z)
        n_valid = valid.sum(axis=0).astype(float)
        first_full = np.argmax(valid.all(axis=0)) if valid.all(axis=0).any() else n
        denom = np.where((n_valid > 0) & (np.arange(n) >= first_full), n_valid, np.nan)
        with np.errstate(invalid="ignore"):
            os_, ob = (Z < -Z_THRESHOLD).sum(0), (Z > Z_THRESHOLD).sum(0)
            xos, xob = (Z < -Z_EXTREME).sum(0), (Z > Z_EXTREME).sum(0)
        self.ts_data["OversoldBreadth"] = os_ / denom * 100
        self.ts_data["OverboughtBreadth"] = ob / denom * 100
        self.ts_data["ExtremeOversold"] = xos / denom * 100
        self.ts_data["ExtremeOverbought"] = xob / denom * 100
        self.ts_data["BuySignalBreadth"] = sum(self.ts_data[f"Buy_{lb}"].astype(int) for lb in self.active_lookbacks)
        self.ts_data["SellSignalBreadth"] = sum(self.ts_data[f"Sell_{lb}"].astype(int) for lb in self.active_lookbacks)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg_z = np.nanmean(Z, axis=0)
        self.ts_data["AvgZ"] = np.where(np.isfinite(denom), avg_z, np.nan)
        self.ts_data["ConvictionRaw"] = (ob - os_) / denom * 100 + 0.5 * (xob - xos) / denom * 100

    def _compute_conviction(self) -> None:
        """Leaky accumulation of raw conviction, soft-bounded to ±100, with a 95% band."""
        state, se = drift_diffusion_filter(self.ts_data["ConvictionRaw"].to_numpy())
        self.ts_data["ConvictionScore"] = state
        self.ts_data["ConvictionBounded"] = soft_bound(state)
        self.ts_data["ConvictionUpper"] = soft_bound(state + 1.96 * se)
        self.ts_data["ConvictionLower"] = soft_bound(state - 1.96 * se)
        self.ts_data["Regime"] = [classify_conviction(c)[2] for c in self.ts_data["ConvictionBounded"]]

    @staticmethod
    def _swing_points(values: np.ndarray, order: int) -> tuple[list[int], list[int]]:
        """Swing highs/lows confirmed ``order`` rows later (index = confirmation row)."""
        tops, bottoms = [], []
        for i in range(2 * order, len(values)):
            w = values[i - 2 * order: i + 1]
            if not np.isfinite(w).all():
                continue
            if np.argmax(w) == order:
                tops.append(i)
            if np.argmin(w) == order:
                bottoms.append(i)
        return tops, bottoms

    def _find_pivots(self, order: int = 5) -> None:
        r = self.residuals
        tops, bottoms = self._swing_points(r, order)
        self.ts_data["IsPivotTop"] = False
        self.ts_data["IsPivotBottom"] = False
        self.ts_data.loc[tops, "IsPivotTop"] = True
        self.ts_data.loc[bottoms, "IsPivotBottom"] = True
        rv = r[np.isfinite(r)]
        self.residual_stats = {
            "mean": float(rv.mean()), "std": float(rv.std()), "current": float(r[-1]),
            "percentile": float((rv < r[-1]).mean() * 100),
        } if len(rv) else {}

    def _compute_divergences(self, order: int = 5) -> None:
        """Target makes a lower low while the gap makes a higher low (bullish), or the reverse.

        Flagged on the row that confirms the swing, so it is known at that time.
        """
        n = len(self.ts_data)
        bull, bear = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        price, resid = self.y, self.residuals
        spread = pd.Series(resid).expanding(min_periods=20).std().to_numpy()
        tops, bottoms = self._swing_points(np.where(np.isfinite(resid), price, np.nan), order)
        last = None
        for i in bottoms:
            c = i - order
            if last is not None and price[c] < price[last] and resid[c] > resid[last] and resid[c] < -0.5 * spread[c]:
                bull[i] = True
            last = c
        last = None
        for i in tops:
            c = i - order
            if last is not None and price[c] > price[last] and resid[c] < resid[last] and resid[c] > 0.5 * spread[c]:
                bear[i] = True
            last = c
        self.ts_data["BullishDiv"] = bull
        self.ts_data["BearishDiv"] = bear

    def _compute_forward_changes(self) -> None:
        """% change of the target over the next h rows (the outcome the signal is judged on)."""
        y = self.y
        for h in FORWARD_HORIZONS:
            fwd = np.full(len(y), np.nan)
            if len(y) > h:
                base = y[:-h]
                with np.errstate(divide="ignore", invalid="ignore"):
                    fwd[:-h] = np.where(np.abs(base) > 1e-10, (y[h:] - base) / np.abs(base) * 100, np.nan)
            self.ts_data[f"FwdChg_{h}"] = np.clip(fwd, -100, 100)

    def _compute_ou_diagnostics(self) -> None:
        """OU fit of the gap: full sample (reference) and trailing OU_WINDOW (current)."""
        r = self.residuals[np.isfinite(self.residuals)]
        if len(r) < 30:
            self.ou_params = {"half_life": float("nan"), "adf_pvalue": float("nan"),
                              "kpss_pvalue": float("nan"), "theta_consistent": True}
            return
        th_full, _, _ = ornstein_uhlenbeck_estimate(r)
        window = min(OU_WINDOW, len(r))
        theta, mu, sigma = ornstein_uhlenbeck_estimate(r[-window:])

        adf_p = kpss_p = float("nan")
        if HAS_STATSMODELS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    adf_p = float(adfuller(r, autolag="AIC")[1])
                except Exception as exc:
                    log.debug("ADF failed: %s", exc)
                try:
                    kpss_p = float(kpss(r, regression="c", nlags="auto")[1])
                except Exception as exc:
                    log.debug("KPSS failed: %s", exc)

        offset = len(self.residuals) - len(r)
        self.theta_history = []
        for end in list(range(window, len(r), OU_HISTORY_STEP)) + [len(r)]:
            th, _, _ = ornstein_uhlenbeck_estimate(r[end - window:end])
            self.theta_history.append({"index": offset + end - 1, "theta": th, "half_life": half_life(th)})

        hl, hl_full = half_life(theta), half_life(th_full)
        self.ou_params = {
            "theta": theta, "mu": mu, "sigma": sigma, "window": window,
            "half_life": hl, "theta_full": th_full, "half_life_full": hl_full,
            "stationary_std": sigma / np.sqrt(2.0 * theta),
            "adf_pvalue": adf_p, "kpss_pvalue": kpss_p,
            "theta_consistent": bool(0.5 <= hl / hl_full <= 2.0),
        }
        mean, std = ou_forecast(float(r[-1]), theta, mu, sigma, OU_PROJECTION_DAYS)
        self.ou_projection = mean
        self.ou_projection_upper = mean + 1.96 * std
        self.ou_projection_lower = mean - 1.96 * std

    def _compute_hurst(self) -> None:
        """DFA α of the gap's daily changes (whole sample): < 0.5 means the gap mean-reverts."""
        r = self.residuals[np.isfinite(self.residuals)]
        self.hurst = hurst_dfa(np.diff(r)) if len(r) > 60 else float("nan")

    def _compute_validation(self) -> None:
        """How the system's own outputs have related to what happened next (whole sample).

        For each horizon h: regress the target's forward h-row % change on BUY and SELL
        dummies (live-signal definition) with Newey-West(h) errors. ``edge`` is the move
        in the signal's favour relative to neutral days; t > 2 suggests it is not noise.
        Also reports information coefficients of the gap and of conviction.
        """
        ts = self.ts_data
        c = ts["ConvictionBounded"].to_numpy()
        valid = np.isfinite(c)
        buy, sell = valid & (c < -CONVICTION_WEAK), valid & (c > CONVICTION_WEAK)
        by_h = {}
        for h in FORWARD_HORIZONS:
            f = ts[f"FwdChg_{h}"].to_numpy()
            m = valid & np.isfinite(f)
            if m.sum() < 60:
                continue
            Xd = np.column_stack([np.ones(m.sum()), buy[m], sell[m]]).astype(float)
            beta, se = newey_west_ols(f[m], Xd, lags=h)
            res = {"n_days": int(m.sum()), "neutral_avg": float(beta[0])}
            for j, (name, mask, direction) in enumerate((("buy", buy, 1.0), ("sell", sell, -1.0)), start=1):
                sel = mask & m
                cnt = int(sel.sum())
                ok = cnt >= 5 and se[j] > 0
                res[name] = {
                    "days": cnt, "episodes": count_episodes(sel),
                    "avg_fwd": float(f[sel].mean()) if cnt else float("nan"),
                    "edge": float(direction * beta[j]) if ok else float("nan"),
                    "t_stat": float(direction * beta[j] / se[j]) if ok else float("nan"),
                    "hit_rate": float(np.mean(direction * f[sel] > 0)) if cnt else float("nan"),
                }
            by_h[h] = res

        h = HEADLINE_HORIZON
        fwd = ts[f"FwdChg_{h}"].to_numpy() if f"FwdChg_{h}" in ts else np.full(len(ts), np.nan)
        gap = information_coefficient(ts["Residual"].to_numpy(), fwd, lags=h)
        sig = information_coefficient(c, fwd, lags=h)
        # Both should be negative if the system works (overvalued → falls). Report as edge = −IC.
        self.validation = {
            "by_horizon": by_h, "horizon": h,
            "gap_edge": -gap["ic"], "gap_edge_t": -gap["t_stat"],
            "signal_edge": -sig["ic"], "signal_edge_t": -sig["t_stat"], "n": sig["n"],
        }
        self.model_stats.update({k: self.validation[k] for k in ("gap_edge", "gap_edge_t", "signal_edge", "signal_edge_t")})
