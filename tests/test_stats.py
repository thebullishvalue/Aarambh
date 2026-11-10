import numpy as np
import pytest

from aarambh.stats import (
    calibrated_zscores, classify_zones, count_episodes, drift_diffusion_filter, hurst_dfa,
    newey_west_ols, ornstein_uhlenbeck_estimate, ou_forecast,
)


def ar1(n, a, seed, sd=1.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = a * x[t - 1] + rng.normal(0, sd)
    return x


class TestCalibratedZ:
    @pytest.mark.parametrize("window", [5, 20, 100])
    def test_calibrated_for_every_window(self, window):
        z = calibrated_zscores(ar1(4000, 0.95, seed=window), window)
        z = z[np.isfinite(z)][500:]
        assert abs(np.std(z) - 1.0) < 0.1
        assert 0.27 < np.mean(np.abs(z) > 1) < 0.37
        assert 0.03 < np.mean(np.abs(z) > 2) < 0.075

    def test_causal(self):
        x = ar1(800, 0.9, seed=1)
        x2 = x.copy()
        x2[600:] += 50.0
        z1, z2 = calibrated_zscores(x, 20), calibrated_zscores(x2, 20)
        np.testing.assert_array_equal(z1[:600], z2[:600])

    def test_zones(self):
        z = np.array([-3, -1.5, 0, 1.5, 3, np.nan])
        assert classify_zones(z).tolist() == ["Extreme Under", "Undervalued", "Fair Value",
                                              "Overvalued", "Extreme Over", "N/A"]


class TestOU:
    def test_recovers_half_life_and_corrects_bias_upward(self):
        a_true = 0.95                                  # half-life ≈ 13.5 rows
        estimates, naive = [], []
        for seed in range(40):
            x = ar1(250, a_true, seed)
            theta, _, _ = ornstein_uhlenbeck_estimate(x)
            estimates.append(np.exp(-theta))
            naive.append(np.corrcoef(x[:-1], x[1:])[0, 1])
        # OLS is biased low; the corrected estimate must sit closer to the truth.
        assert abs(np.mean(estimates) - a_true) < abs(np.mean(naive) - a_true)
        assert abs(np.mean(estimates) - a_true) < 0.01

    def test_sigma_consistent_with_stationary_std(self):
        x = ar1(20000, 0.9, seed=3)
        theta, mu, sigma = ornstein_uhlenbeck_estimate(x)
        assert abs(sigma / np.sqrt(2 * theta) - np.std(x)) / np.std(x) < 0.05

    def test_forecast_converges_to_stationary(self):
        mean, std = ou_forecast(5.0, theta=0.1, mu=1.0, sigma=0.5, horizon=200)
        assert mean[0] == pytest.approx(1 + 4 * np.exp(-0.1))
        assert mean[-1] == pytest.approx(1.0, abs=1e-6)
        assert std[-1] == pytest.approx(0.5 / np.sqrt(0.2), rel=1e-6)
        assert np.all(np.diff(std) >= 0) and std[0] < std[10]


class TestAccumulator:
    def test_unit_gain_and_start_at_first_observation(self):
        state, se = drift_diffusion_filter(np.r_[np.nan, np.full(300, 40.0)])
        assert np.isnan(state[0]) and state[1] == 40.0
        assert state[-1] == pytest.approx(40.0)
        assert np.all(np.isfinite(se[1:]))

    def test_causal(self):
        x = np.random.default_rng(0).normal(size=300)
        x2 = x.copy()
        x2[200:] = 100
        np.testing.assert_array_equal(drift_diffusion_filter(x)[0][:200], drift_diffusion_filter(x2)[0][:200])


class TestDFA:
    def test_white_noise_half(self):
        assert abs(hurst_dfa(np.random.default_rng(0).normal(size=4000)) - 0.5) < 0.08

    def test_mean_reverting_increments_below_half(self):
        assert hurst_dfa(np.diff(ar1(4000, 0.9, seed=2))) < 0.4

    def test_trending_increments_above_half(self):
        assert hurst_dfa(ar1(4000, 0.9, seed=4)) > 0.6


def test_newey_west_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 0.3 * x + ar1(500, 0.7, seed=9)
    X = np.column_stack([np.ones(500), x])
    beta, se = newey_west_ols(y, X, lags=10)
    ref = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 10, "use_correction": True})
    np.testing.assert_allclose(beta, ref.params, rtol=1e-10)
    np.testing.assert_allclose(se, ref.bse, rtol=1e-6)


def test_count_episodes():
    assert count_episodes(np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=bool)) == 3
    assert count_episodes(np.array([1, 1], dtype=bool)) == 1
    assert count_episodes(np.zeros(4, dtype=bool)) == 0
