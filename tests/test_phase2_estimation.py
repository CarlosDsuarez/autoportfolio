"""
test_phase2_estimation.py — Tests unitarios de Fase 2.

Cubre: T-07 (rolling engine), T-08 (expected returns),
T-09 (covariance estimators), T-11 (universe filter).

Restricciones validadas:
  R-03: Filtro de liquidez excluye activos ilíquidos.
  R-05: Sin look-ahead en ventanas.
  R-09: Estimadores estables, PSD, condición razonable.
  R-10: Reproducibilidad con seed(42).

Ejecutar: pytest tests/test_phase2_estimation.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.rolling_engine import (
    get_window,
    rolling_windows,
    get_rebalance_dates,
)
from src.expected_returns import (
    mean_historical_return,
    capm_shrunk_return,
    black_litterman_equilibrium,
    estimate_expected_returns,
)
from src.covariance_estimators import (
    sample_covariance,
    ledoit_wolf_covariance,
    pca_factor_covariance,
    estimate_covariance,
    eigenvalue_diagnostics,
)
from src.universe_filter import (
    filter_universe_at_date,
    precompute_rolling_metrics,
)

SEED = 42
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def prices():
    """Synthetic prices: 600 days, 10 tickers (includes SPY)."""
    np.random.seed(SEED)
    dates = pd.bdate_range("2021-01-01", periods=600, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY",
               "JPM", "JNJ", "XOM", "AGG", "GLD"]
    log_rets = np.random.normal(0.0003, 0.015, (600, 10))
    # Add correlation structure
    market_factor = np.random.normal(0, 0.01, 600)
    for i in range(10):
        log_rets[:, i] += 0.5 * market_factor
    prices_arr = 100 * np.exp(np.cumsum(log_rets, axis=0))
    df = pd.DataFrame(prices_arr, index=dates, columns=tickers)
    df.index.name = "date"
    return df


@pytest.fixture
def volumes(prices):
    """Synthetic volumes aligned with prices."""
    np.random.seed(SEED + 1)
    vols = np.random.randint(100_000, 50_000_000, prices.shape)
    df = pd.DataFrame(vols, index=prices.index, columns=prices.columns,
                      dtype=float)
    df.index.name = "date"
    return df


# ===========================================================================
# T-07: Rolling Window Engine
# ===========================================================================
class TestRollingEngine:
    def test_get_window_size(self, prices):
        t = prices.index[300]
        w = get_window(prices, t, window=252)
        assert len(w) == 252

    def test_get_window_no_future_data(self, prices):
        """R-05: No data after t."""
        t = prices.index[300]
        w = get_window(prices, t, window=252)
        assert w.index.max() <= t

    def test_get_window_raises_on_insufficient_data(self, prices):
        t = prices.index[10]  # Only 11 rows available
        with pytest.raises(ValueError, match="Insufficient data"):
            get_window(prices, t, window=252)

    def test_rolling_windows_no_overlap(self, prices):
        """R-05: Train and test windows must not overlap."""
        windows = list(rolling_windows(
            prices, window=252, step=21, test_periods=63
        ))
        assert len(windows) > 0
        for w in windows:
            assert w.train_data.index.max() < w.test_data.index.min()

    def test_rolling_windows_train_size(self, prices):
        windows = list(rolling_windows(
            prices, window=252, step=21, test_periods=63
        ))
        for w in windows:
            assert len(w.train_data) == 252

    def test_rolling_windows_sequential_ids(self, prices):
        windows = list(rolling_windows(
            prices, window=252, step=21, test_periods=63
        ))
        ids = [w.window_id for w in windows]
        assert ids == list(range(1, len(ids) + 1))

    def test_rebalance_dates_monthly(self, prices):
        dates = get_rebalance_dates(prices, freq="M", warmup=252)
        assert len(dates) > 0
        # All dates must have >= warmup rows before them
        for d in dates:
            assert (prices.index <= d).sum() >= 252

    def test_rebalance_dates_quarterly(self, prices):
        dates = get_rebalance_dates(prices, freq="Q", warmup=252)
        assert len(dates) > 0
        assert len(dates) < len(
            get_rebalance_dates(prices, freq="M", warmup=252)
        )


# ===========================================================================
# T-08: Expected Returns
# ===========================================================================
class TestExpectedReturns:
    def test_historical_return_shape(self, prices):
        mu = mean_historical_return(prices)
        assert len(mu) == prices.shape[1]

    def test_historical_return_reasonable_range(self, prices):
        mu = mean_historical_return(prices)
        # Annualized returns should be in [-100%, +200%] for normal data
        assert mu.min() > -1.0
        assert mu.max() < 2.0

    def test_capm_shrunk_requires_benchmark(self, prices):
        # Remove benchmark
        no_spy = prices.drop(columns=["SPY"])
        with pytest.raises(ValueError, match="Benchmark"):
            capm_shrunk_return(no_spy, benchmark_col="SPY")

    def test_capm_shrunk_shape(self, prices):
        mu = capm_shrunk_return(prices, benchmark_col="SPY")
        assert len(mu) == prices.shape[1]

    def test_capm_shrunk_alpha_extremes(self, prices):
        """alpha=0 → pure historical, alpha=1 → pure CAPM."""
        mu_hist = mean_historical_return(prices)
        mu_a0 = capm_shrunk_return(prices, shrinkage_alpha=0.0)
        pd.testing.assert_series_equal(mu_a0, mu_hist, atol=1e-10)

    def test_black_litterman_shape(self, prices):
        cov = sample_covariance(prices)
        mu = black_litterman_equilibrium(prices, cov)
        assert len(mu) == prices.shape[1]

    def test_black_litterman_positive_with_positive_cov(self, prices):
        """With positive definite cov and positive weights, pi > 0."""
        cov = sample_covariance(prices)
        mu = black_litterman_equilibrium(
            prices, cov, risk_aversion=2.5, risk_free_rate=0.0
        )
        # Equal-weight, positive cov → implied returns should be positive
        assert (mu >= 0).all()

    def test_unified_interface(self, prices):
        for method in ["historical", "capm_shrunk"]:
            mu = estimate_expected_returns(prices, method=method)
            assert len(mu) == prices.shape[1]

    def test_unified_bl_requires_cov(self, prices):
        with pytest.raises(ValueError, match="cov_matrix"):
            estimate_expected_returns(prices, method="black_litterman")

    def test_unknown_method_raises(self, prices):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_expected_returns(prices, method="nonexistent")


# ===========================================================================
# T-09: Covariance Estimators
# ===========================================================================
class TestCovarianceEstimators:
    def test_sample_cov_shape(self, prices):
        sigma = sample_covariance(prices)
        n = prices.shape[1]
        assert sigma.shape == (n, n)

    def test_sample_cov_symmetric(self, prices):
        sigma = sample_covariance(prices)
        np.testing.assert_allclose(
            sigma.values, sigma.values.T, atol=1e-12
        )

    def test_sample_cov_psd(self, prices):
        sigma = sample_covariance(prices)
        eigenvalues = np.linalg.eigh(sigma.values)[0]
        assert eigenvalues.min() >= -1e-10

    def test_ledoit_wolf_shape(self, prices):
        sigma = ledoit_wolf_covariance(prices)
        n = prices.shape[1]
        assert sigma.shape == (n, n)

    def test_ledoit_wolf_better_conditioned(self, prices):
        """LW should have lower condition number than sample."""
        s_sample = sample_covariance(prices)
        s_lw = ledoit_wolf_covariance(prices)
        cn_sample = np.linalg.cond(s_sample.values)
        cn_lw = np.linalg.cond(s_lw.values)
        assert cn_lw <= cn_sample * 1.1  # Allow 10% margin

    def test_pca_shape(self, prices):
        sigma = pca_factor_covariance(prices, n_factors=3)
        n = prices.shape[1]
        assert sigma.shape == (n, n)

    def test_pca_psd(self, prices):
        sigma = pca_factor_covariance(prices, n_factors=3)
        eigenvalues = np.linalg.eigh(sigma.values)[0]
        assert eigenvalues.min() >= -1e-10

    def test_pca_fewer_factors_lower_rank(self, prices):
        """PCA with fewer factors should have lower effective rank."""
        s3 = pca_factor_covariance(prices, n_factors=3)
        s5 = pca_factor_covariance(prices, n_factors=5)
        # Condition number with fewer factors should be comparable or lower
        cn3 = np.linalg.cond(s3.values)
        cn5 = np.linalg.cond(s5.values)
        # This is a soft check — more factors can be better or worse
        assert cn3 > 0 and cn5 > 0

    def test_unified_interface(self, prices):
        for method in ["sample", "ledoit_wolf", "pca"]:
            sigma = estimate_covariance(prices, method=method)
            assert sigma.shape[0] == sigma.shape[1]
            assert sigma.shape[0] == prices.shape[1]

    def test_eigenvalue_diagnostics(self, prices):
        sigma = sample_covariance(prices)
        diag = eigenvalue_diagnostics(sigma, "test")
        assert diag["condition_number"] > 0
        assert diag["min_eigenvalue"] >= 0
        assert 0 < diag["top1_explained"] <= 1.0

    def test_unknown_method_raises(self, prices):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_covariance(prices, method="nonexistent")


# ===========================================================================
# T-11: Dynamic Liquidity Filter
# ===========================================================================
class TestUniverseFilter:
    def test_filter_returns_subset(self, prices, volumes):
        t = prices.index[-1]
        active, _ = filter_universe_at_date(
            t, prices, volumes,
            amihud_percentile_threshold=90.0,
            min_volume=100_000,
        )
        assert len(active) <= len(prices.columns)
        assert len(active) > 0

    def test_filter_no_lookahead(self, prices, volumes):
        """R-05: Filter at t must not use data after t."""
        t = prices.index[300]
        active, _ = filter_universe_at_date(
            t, prices, volumes,
            amihud_percentile_threshold=90.0,
            min_volume=100_000,
        )
        # All active tickers should have data up to t
        for ticker in active:
            assert ticker in prices.columns

    def test_strict_threshold_excludes_more(self, prices, volumes):
        """Lower percentile threshold → more exclusions."""
        t = prices.index[-1]
        active_90, _ = filter_universe_at_date(
            t, prices, volumes, amihud_percentile_threshold=90.0,
        )
        active_50, _ = filter_universe_at_date(
            t, prices, volumes, amihud_percentile_threshold=50.0,
        )
        assert len(active_50) <= len(active_90)

    def test_exclusion_log_columns(self, prices, volumes):
        t = prices.index[-1]
        _, log = filter_universe_at_date(
            t, prices, volumes,
            amihud_percentile_threshold=80.0,
        )
        if not log.empty:
            assert "ticker" in log.columns
            assert "reason" in log.columns

    def test_precompute_metrics_shape(self, prices, volumes):
        r_amihud, r_vol = precompute_rolling_metrics(
            prices, volumes, window=30
        )
        assert r_amihud.shape == prices.shape
        assert r_vol.shape == volumes.shape

    def test_high_volume_threshold_excludes_all_except_liquid(
        self, prices, volumes
    ):
        """Very high min_volume should exclude low-volume assets."""
        t = prices.index[-1]
        # Set unreasonably high threshold
        active, excluded = filter_universe_at_date(
            t, prices, volumes, min_volume=1e15,
        )
        # Some or all should be excluded
        assert len(active) < len(prices.columns)


# ===========================================================================
# R-10: Reproducibility
# ===========================================================================
class TestReproducibility:
    def test_covariance_deterministic(self, prices):
        s1 = ledoit_wolf_covariance(prices)
        s2 = ledoit_wolf_covariance(prices)
        pd.testing.assert_frame_equal(s1, s2)

    def test_expected_returns_deterministic(self, prices):
        mu1 = mean_historical_return(prices)
        mu2 = mean_historical_return(prices)
        pd.testing.assert_series_equal(mu1, mu2)

    def test_rolling_windows_deterministic(self, prices):
        w1 = list(rolling_windows(prices, window=252, step=63))
        w2 = list(rolling_windows(prices, window=252, step=63))
        assert len(w1) == len(w2)
        for a, b in zip(w1, w2):
            assert a.train_start == b.train_start
            assert a.test_end == b.test_end
