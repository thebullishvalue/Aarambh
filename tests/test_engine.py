import numpy as np
import pandas as pd
import pytest

from aarambh.config import CONVICTION_STRONG, CONVICTION_WEAK, ENSEMBLE_WEIGHT_FLOOR, LOOKBACK_WINDOWS
from aarambh.engine import FairValueEngine, classify_conviction


def synthetic(n, gap_ar, seed, gap_sd=1.0):
    """y = linear fair value of 3 random-walk drivers + a gap with AR(1) coefficient gap_ar."""
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(0, 0.05, size=(n, 3)), axis=0)
    gap = np.zeros(n)
    for t in range(1, n):
        gap[t] = gap_ar * gap[t - 1] + rng.normal(0, gap_sd)
    y = 50 + X @ np.array([2.0, -1.0, 0.5]) + gap
    return X, y


@pytest.fixture(scope="module")
def small_fit():
    X, y = synthetic(400, 0.95, seed=0)
    return X, y, FairValueEngine().fit(X, y, feature_names=["a", "b", "c"])


def test_no_look_ahead(small_fit):
    """The past must not depend on the future: refit on a truncated sample, compare."""
    X, y, full = small_fit
    cut = 300
    part = FairValueEngine().fit(X[:cut], y[:cut], feature_names=["a", "b", "c"])
    cols = [c for c in full.ts_data.columns if not c.startswith("FwdChg_")]
    pd.testing.assert_frame_equal(full.ts_data.loc[: cut - 1, cols], part.ts_data[cols], check_exact=True)


def test_outputs_complete(small_fit):
    _, _, eng = small_fit
    ts = eng.ts_data
    assert eng.active_lookbacks == LOOKBACK_WINDOWS
    for col in ("FairValue", "Residual", "ConvictionBounded", "Regime", "AvgZ", "OversoldBreadth"):
        assert col in ts
    assert ts["FairValue"].iloc[:20].isna().all() and ts["FairValue"].iloc[20:].notna().all()
    assert np.isfinite(ts["ConvictionBounded"].iloc[-1])
    assert ts["ConvictionBounded"].abs().max() < 100
    sig = eng.get_current_signal()
    assert sig["signal"] in {"BUY", "SELL", "HOLD"} and not sig["warming_up"]
    assert len(eng.feature_impact_history) > 10


def test_ensemble_weights_are_a_distribution(small_fit):
    _, _, eng = small_fit
    for row in eng.ensemble_weight_history:
        w = np.array([v for k, v in row.items() if k != "index"])
        assert w.sum() == pytest.approx(1.0)
        assert (w >= ENSEMBLE_WEIGHT_FLOOR - 1e-12).all()


def test_regime_matches_signal_vocabulary(small_fit):
    _, _, eng = small_fit
    for c, regime in zip(eng.ts_data["ConvictionBounded"], eng.ts_data["Regime"]):
        assert classify_conviction(c)[2] == regime


@pytest.mark.parametrize("score,expected", [
    (0, ("HOLD", "NEUTRAL", "NEUTRAL")),
    (CONVICTION_WEAK, ("HOLD", "NEUTRAL", "NEUTRAL")),
    (-30, ("BUY", "WEAK", "OVERSOLD")),
    (50, ("SELL", "MODERATE", "OVERBOUGHT")),
    (-(CONVICTION_STRONG + 1), ("BUY", "STRONG", "STRONGLY OVERSOLD")),
    (float("nan"), ("HOLD", "NEUTRAL", "N/A")),
])
def test_classify_conviction(score, expected):
    assert classify_conviction(score) == expected


def test_validation_finds_a_planted_edge_and_not_a_fake_one():
    """A mean-reverting gap must show a positive edge; a random-walk gap must not."""
    X, y = synthetic(1200, 0.97, seed=1)
    real = FairValueEngine().fit(X, y, feature_names=["a", "b", "c"]).validation
    assert real["gap_edge"] > 0.2 and real["gap_edge_t"] > 3

    X, y = synthetic(1200, 1.0, seed=2, gap_sd=0.3)
    fake = FairValueEngine().fit(X, y, feature_names=["a", "b", "c"]).validation
    assert abs(fake["signal_edge_t"]) < 2.5


def test_short_sample_degrades_gracefully():
    X, y = synthetic(100, 0.9, seed=3)
    eng = FairValueEngine().fit(X, y, feature_names=["a", "b", "c"])
    assert all(lb + 60 < 80 for lb in eng.active_lookbacks)
    assert "signal" in eng.get_current_signal()


def test_long_memory_runs():
    X, y = synthetic(300, 0.95, seed=4)
    eng = FairValueEngine(memory="long").fit(X, y, feature_names=["a", "b", "c"])
    assert eng.model_stats["memory"] == "long"


def test_rejects_unclean_input():
    X, y = synthetic(100, 0.9, seed=5)
    X[10, 0] = np.nan
    with pytest.raises(ValueError):
        FairValueEngine().fit(X, y)
