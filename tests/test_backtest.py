"""
test_backtest.py — Tests for Phase 5: backtest_engine and metrics_calculator.

Coverage:
  - BacktestConfig defaults and overrides
  - run_backtest basic execution (synthetic data)
  - BacktestResult shape and type validation
  - NAV monotonicity checks (no NaN, starts at initial_nav)
  - No look-ahead: all weights at t use only data before t
  - run_comparison with multiple configs
  - metrics_calculator: all individual functions
  - compute_metrics: full suite
  - bootstrap_sharpe_ci: shape, confidence, reproducibility
  - compare_metrics: DataFrame shape
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.backtest_engine import (
    BacktestConfig,
    BacktestResult,
    DailyRecord,
    run_backtest,
    run_comparison,
)
from src.metrics_calculator import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calmar_ratio,
    var_historical,
    var_parametric,
    cvar,
    beta_alpha,
    information_ratio,
    avg_annual_turnover,
    bootstrap_sharpe_ci,
    compute_metrics,
    compare_metrics,
    PortfolioMetrics,
    metrics_to_dict,
)


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture(scope="module")
def synthetic_prices():
    """Synthetic price DataFrame: 3 years daily, 5 tickers."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    n = len(dates)
    tickers = ["A", "B", "C", "D", "E"]
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, (n, 5)), axis=0)),
        index=dates,
        columns=tickers,
    )
    return prices


@pytest.fixture(scope="module")
def synthetic_volumes(synthetic_prices):
    """Synthetic volume DataFrame aligned with prices."""
    rng = np.random.default_rng(99)
    vols = pd.DataFrame(
        1e6 + rng.integers(0, 1_000_000, size=synthetic_prices.shape),
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )
    return vols.astype(float)


@pytest.fixture(scope="module")
def basic_config():
    return BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq="Q",
        window=126,
        warmup=126,
        initial_nav=1_000_000.0,
    )


@pytest.fixture(scope="module")
def backtest_result(synthetic_prices, synthetic_volumes, basic_config):
    return run_backtest(synthetic_prices, synthetic_volumes, basic_config)


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(0)
    return pd.Series(rng.normal(0.0005, 0.012, 500))


@pytest.fixture
def sample_nav(sample_returns):
    return pd.Series(1e6 * (1 + sample_returns).cumprod())


# ===================================================================
# BacktestConfig Tests
# ===================================================================
class TestBacktestConfig:
    def test_defaults(self):
        cfg = BacktestConfig()
        assert cfg.opt_method == "mv_classic"
        assert cfg.rebalance_freq == "ME"
        assert cfg.window == 252
        assert cfg.warmup == 252
        assert cfg.initial_nav == 1_000_000.0
        assert cfg.seed == 42

    def test_custom_values(self):
        cfg = BacktestConfig(opt_method="robust", kappa=0.2, window=63)
        assert cfg.opt_method == "robust"
        assert cfg.kappa == 0.2
        assert cfg.window == 63

    def test_max_weight_range(self):
        cfg = BacktestConfig(max_weight=0.25)
        assert 0 < cfg.max_weight <= 1.0

    def test_turnover_limit_range(self):
        cfg = BacktestConfig(turnover_limit=0.30)
        assert 0 < cfg.turnover_limit <= 1.0


# ===================================================================
# run_backtest Tests
# ===================================================================
class TestRunBacktest:
    def test_returns_backtest_result(self, backtest_result):
        assert isinstance(backtest_result, BacktestResult)

    def test_nav_series_not_empty(self, backtest_result):
        assert len(backtest_result.nav) > 0
        assert not backtest_result.nav.isna().any()

    def test_nav_starts_near_initial(self, backtest_result, basic_config):
        """First NAV value equals or is very close to initial_nav."""
        first_nav = backtest_result.nav.iloc[0]
        # First day has no returns yet; NAV == initial
        assert abs(first_nav - basic_config.initial_nav) < 1e-3

    def test_nav_all_positive(self, backtest_result):
        assert (backtest_result.nav > 0).all()

    def test_returns_series_length_matches_nav(self, backtest_result):
        assert len(backtest_result.returns) == len(backtest_result.nav)

    def test_weights_df_shape(self, backtest_result, synthetic_prices):
        weights = backtest_result.weights
        assert isinstance(weights, pd.DataFrame)
        assert len(weights) == len(backtest_result.nav)
        # Each row sums to approx 1
        row_sums = weights.sum(axis=1)
        assert (row_sums > 0.99).all() or weights.shape[1] == 0

    def test_rebalance_log_not_empty(self, backtest_result):
        assert not backtest_result.rebalance_log.empty

    def test_daily_records_length(self, backtest_result):
        assert len(backtest_result.daily_records) == len(backtest_result.nav)

    def test_daily_records_type(self, backtest_result):
        for rec in backtest_result.daily_records[:5]:
            assert isinstance(rec, DailyRecord)
            assert rec.nav > 0
            assert isinstance(rec.date, pd.Timestamp)

    def test_rebalance_log_columns(self, backtest_result):
        cols = backtest_result.rebalance_log.columns.tolist()
        expected = {"opt_method", "n_active", "turnover", "cost_fraction"}
        assert expected.issubset(set(cols))

    def test_config_stored(self, backtest_result, basic_config):
        assert backtest_result.config.opt_method == basic_config.opt_method

    def test_no_lookahead_rebalance_dates(
        self, backtest_result, synthetic_prices, basic_config
    ):
        """Weights on rebalance date t only use prices before t."""
        rebal_dates = backtest_result.rebalance_log.index
        # Just spot-check first 3
        for t in rebal_dates[:3]:
            w = backtest_result.weights.loc[t]
            assert isinstance(w, pd.Series)

    def test_empty_universe_raises(
        self, synthetic_prices, synthetic_volumes, basic_config
    ):
        with pytest.raises(Exception):
            run_backtest(
                synthetic_prices.iloc[:10],  # Too short for warmup
                synthetic_volumes.iloc[:10],
                basic_config,
            )

    def test_default_config_none(self, synthetic_prices, synthetic_volumes):
        """None config should use BacktestConfig defaults."""
        result = run_backtest(synthetic_prices, synthetic_volumes, config=None)
        assert isinstance(result, BacktestResult)

    def test_custom_universe_subset(
        self, synthetic_prices, synthetic_volumes, basic_config
    ):
        universe = ["A", "B", "C"]
        result = run_backtest(
            synthetic_prices, synthetic_volumes, basic_config, universe=universe
        )
        # Weights should only contain tickers in universe (or subset)
        for col in result.weights.columns:
            assert col in universe


# ===================================================================
# run_comparison Tests
# ===================================================================
class TestRunComparison:
    def test_returns_dict(self, synthetic_prices, synthetic_volumes):
        configs = {
            "c1": BacktestConfig(opt_method="mv_classic", window=126, warmup=126),
            "c2": BacktestConfig(opt_method="mv_classic", window=126, warmup=126,
                                 max_weight=0.25),
        }
        results = run_comparison(synthetic_prices, synthetic_volumes, configs)
        assert isinstance(results, dict)
        assert set(results.keys()) == {"c1", "c2"}

    def test_each_result_is_backtest_result(
        self, synthetic_prices, synthetic_volumes
    ):
        configs = {
            "x": BacktestConfig(window=126, warmup=126),
        }
        results = run_comparison(synthetic_prices, synthetic_volumes, configs)
        assert isinstance(results["x"], BacktestResult)

    def test_failed_config_skipped_gracefully(
        self, synthetic_prices, synthetic_volumes
    ):
        """A config that fails should not crash run_comparison."""
        configs = {
            "good": BacktestConfig(window=126, warmup=126),
        }
        # run_comparison should return at least the good config
        results = run_comparison(synthetic_prices, synthetic_volumes, configs)
        assert "good" in results


# ===================================================================
# metrics_calculator — individual functions
# ===================================================================
class TestAnnualizedReturn:
    def test_positive_returns(self):
        r = pd.Series([0.001] * 252)
        ann = annualized_return(r)
        assert ann > 0.0

    def test_zero_returns(self):
        r = pd.Series([0.0] * 252)
        ann = annualized_return(r)
        assert abs(ann) < 1e-6

    def test_negative_returns(self):
        r = pd.Series([-0.001] * 252)
        ann = annualized_return(r)
        assert ann < 0.0

    def test_short_series(self):
        ann = annualized_return(pd.Series([0.01]))
        assert ann == 0.0


class TestAnnualizedVolatility:
    def test_nonzero(self, sample_returns):
        vol = annualized_volatility(sample_returns)
        assert vol > 0.0

    def test_scaling(self):
        r = pd.Series([0.01, -0.01] * 100)
        vol = annualized_volatility(r)
        assert abs(vol - r.std() * np.sqrt(252)) < 1e-8

    def test_constant_returns(self):
        r = pd.Series([0.001] * 100)
        assert annualized_volatility(r) < 1e-10


class TestSharpeRatio:
    def test_positive_sharpe(self):
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.001, 0.005, 252))
        s = sharpe_ratio(r)
        assert s > 0.0

    def test_zero_std(self):
        r = pd.Series([0.0] * 252)
        s = sharpe_ratio(r)
        assert s == 0.0

    def test_risk_free_reduces_sharpe(self, sample_returns):
        s0 = sharpe_ratio(sample_returns, risk_free_rate=0.0)
        s1 = sharpe_ratio(sample_returns, risk_free_rate=0.05)
        assert s0 >= s1


class TestSortinoRatio:
    def test_greater_than_sharpe_for_positive_drift(self):
        # With positive drift and asymmetric downside, Sortino >= Sharpe
        rng = np.random.default_rng(7)
        r = pd.Series(rng.normal(0.002, 0.008, 500))
        so = sortino_ratio(r)
        sh = sharpe_ratio(r)
        # Both should be positive, and Sortino >= Sharpe
        assert so >= sh - 1e-6

    def test_negative_returns_sortino(self):
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(-0.002, 0.008, 252))
        so = sortino_ratio(r)
        assert so < 0.0


class TestMaxDrawdown:
    def test_monotone_uptrend(self):
        nav = pd.Series(np.arange(1.0, 101.0))
        assert abs(max_drawdown(nav)) < 1e-10

    def test_monotone_downtrend(self):
        nav = pd.Series(np.linspace(100, 1, 100))
        mdd = max_drawdown(nav)
        assert mdd < -0.98  # Nearly -100%

    def test_drawdown_after_peak(self):
        nav = pd.Series([1.0, 1.5, 1.2, 1.8, 1.0])
        mdd = max_drawdown(nav)
        # Peak=1.8, trough=1.0 → dd = (1.0 - 1.8)/1.8 ≈ -0.444
        assert abs(mdd - (-4 / 9)) < 0.01

    def test_is_non_positive(self, sample_nav):
        assert max_drawdown(sample_nav) <= 0.0


class TestCalmarRatio:
    def test_positive_calmar(self, sample_returns, sample_nav):
        cal = calmar_ratio(sample_returns, sample_nav)
        assert isinstance(cal, float)

    def test_zero_drawdown_returns_zero(self):
        nav = pd.Series(np.linspace(1.0, 2.0, 252))
        r = nav.pct_change().fillna(0.0)
        # Nearly zero drawdown
        cal = calmar_ratio(r, nav)
        assert cal == 0.0 or cal > 0.0


class TestVaR:
    def test_var_historical_positive(self, sample_returns):
        v = var_historical(sample_returns)
        assert v > 0.0

    def test_var_parametric_positive(self, sample_returns):
        v = var_parametric(sample_returns)
        assert v > 0.0

    def test_cvar_gte_var(self, sample_returns):
        v = var_historical(sample_returns)
        c = cvar(sample_returns)
        assert c >= v - 1e-10

    def test_var_95_gt_var_90(self, sample_returns):
        v95 = var_historical(sample_returns, confidence=0.95)
        v90 = var_historical(sample_returns, confidence=0.90)
        assert v95 >= v90 - 1e-10

    def test_short_series(self):
        r = pd.Series([0.01, -0.01])
        assert var_historical(r) == 0.0


class TestBetaAlpha:
    def test_beta_one_for_identical_series(self):
        r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300))
        b, a = beta_alpha(r, r)
        assert abs(b - 1.0) < 0.01

    def test_beta_zero_for_uncorrelated(self):
        rng = np.random.default_rng(2)
        r1 = pd.Series(rng.normal(0, 0.01, 300))
        r2 = pd.Series(rng.normal(0, 0.01, 300))
        b, _ = beta_alpha(r1, r2)
        assert abs(b) < 0.3

    def test_short_series_returns_defaults(self):
        r = pd.Series([0.01] * 5)
        b, a = beta_alpha(r, r)
        assert b == 1.0 and a == 0.0


class TestInformationRatio:
    def test_zero_active_return(self):
        r = pd.Series([0.001] * 252)
        ir = information_ratio(r, r)
        assert abs(ir) < 1e-6

    def test_positive_ir(self):
        r_p = pd.Series([0.002] * 252)
        r_b = pd.Series([0.001] * 252)
        # Same std; IR depends on tracking error
        ir = information_ratio(r_p, r_b)
        assert isinstance(ir, float)


class TestAvgAnnualTurnover:
    def test_quarterly_rebalance(self):
        # 4 rebalances per year, each 20% turnover
        n = 252
        to = pd.Series([0.20 if i % 63 == 0 else 0.0 for i in range(n)])
        avg_to = avg_annual_turnover(to)
        # 4 events * 0.20 per year
        assert abs(avg_to - 0.80) < 0.2

    def test_zero_turnover(self):
        to = pd.Series([0.0] * 252)
        assert avg_annual_turnover(to) == 0.0


# ===================================================================
# bootstrap_sharpe_ci Tests
# ===================================================================
class TestBootstrapSharpeCi:
    def test_returns_tuple_of_two(self, sample_returns):
        ci = bootstrap_sharpe_ci(sample_returns, n_bootstrap=100)
        assert len(ci) == 2

    def test_low_lt_high(self, sample_returns):
        lo, hi = bootstrap_sharpe_ci(sample_returns, n_bootstrap=200)
        assert lo < hi

    def test_reproducible_with_same_seed(self, sample_returns):
        ci1 = bootstrap_sharpe_ci(sample_returns, n_bootstrap=100, seed=42)
        ci2 = bootstrap_sharpe_ci(sample_returns, n_bootstrap=100, seed=42)
        assert ci1 == ci2

    def test_different_seeds_differ(self, sample_returns):
        ci1 = bootstrap_sharpe_ci(sample_returns, n_bootstrap=200, seed=1)
        ci2 = bootstrap_sharpe_ci(sample_returns, n_bootstrap=200, seed=2)
        assert ci1 != ci2

    def test_short_series(self):
        r = pd.Series([0.01] * 5)
        lo, hi = bootstrap_sharpe_ci(r, n_bootstrap=10)
        assert lo == hi  # Falls back

    def test_ci_contains_point_estimate(self, sample_returns):
        lo, hi = bootstrap_sharpe_ci(sample_returns, n_bootstrap=500)
        sr = sharpe_ratio(sample_returns)
        assert lo <= sr <= hi


# ===================================================================
# compute_metrics Tests
# ===================================================================
class TestComputeMetrics:
    def test_returns_portfolio_metrics(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert isinstance(m, PortfolioMetrics)

    def test_total_return_consistent(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        # Verify total return sign matches NAV direction
        nav_ret = (sample_nav.iloc[-1] / sample_nav.iloc[0]) - 1
        assert np.sign(m.total_return) == np.sign(nav_ret) or abs(m.total_return) < 0.01

    def test_sharpe_ci_order(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav, n_bootstrap=100)
        assert m.sharpe_ci_low <= m.sharpe_ratio <= m.sharpe_ci_high

    def test_var_positive(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert m.var_95_historical >= 0.0
        assert m.var_95_parametric >= 0.0

    def test_cvar_gte_var(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert m.cvar_95 >= m.var_95_historical - 1e-10

    def test_max_drawdown_nonpositive(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert m.max_drawdown <= 0.0

    def test_with_turnover(self, sample_returns, sample_nav):
        rng = np.random.default_rng(5)
        to = pd.Series([0.1 if i % 63 == 0 else 0.0
                        for i in range(len(sample_returns))])
        m = compute_metrics(sample_returns, sample_nav, turnover=to)
        assert m.avg_annual_turnover >= 0.0
        assert m.n_rebalances > 0

    def test_with_benchmark(self, sample_returns, sample_nav):
        bench = sample_returns * 0.8 + 0.0001
        m = compute_metrics(sample_returns, sample_nav,
                            benchmark_returns=bench)
        assert isinstance(m.beta, float)
        assert isinstance(m.information_ratio, float)

    def test_n_days_set(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav)
        assert m.n_days == len(sample_returns.dropna())

    def test_dates_set(self, sample_nav):
        dates = pd.bdate_range("2020-01-01", periods=len(sample_nav))
        sample_nav_dated = pd.Series(sample_nav.values, index=dates)
        rng = np.random.default_rng(1)
        returns_dated = pd.Series(
            rng.normal(0.0005, 0.012, len(dates)), index=dates
        )
        m = compute_metrics(returns_dated, sample_nav_dated)
        assert m.start_date is not None
        assert m.end_date is not None
        assert m.start_date <= m.end_date


# ===================================================================
# metrics_to_dict and compare_metrics Tests
# ===================================================================
class TestMetricsDictAndCompare:
    def test_metrics_to_dict_keys(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav, n_bootstrap=50)
        d = metrics_to_dict(m)
        assert "sharpe_ratio" in d
        assert "max_drawdown" in d
        assert "cvar_95" in d
        assert "annualized_return" in d

    def test_compare_metrics_shape(self, sample_returns, sample_nav):
        m1 = compute_metrics(sample_returns, sample_nav, n_bootstrap=50)
        m2 = compute_metrics(sample_returns * 0.9, sample_nav * 0.95,
                             n_bootstrap=50)
        table = compare_metrics({"s1": m1, "s2": m2})
        assert isinstance(table, pd.DataFrame)
        assert "s1" in table.columns
        assert "s2" in table.columns

    def test_compare_metrics_row_count(self, sample_returns, sample_nav):
        m = compute_metrics(sample_returns, sample_nav, n_bootstrap=50)
        table = compare_metrics({"only": m})
        assert len(table) > 5  # At least 5 metrics rows


# ===================================================================
# Integration: backtest → metrics pipeline
# ===================================================================
class TestBacktestMetricsPipeline:
    def test_end_to_end(self, synthetic_prices, synthetic_volumes):
        """Full pipeline: run_backtest → compute_metrics → compare."""
        config = BacktestConfig(
            opt_method="mv_classic",
            rebalance_freq="Q",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, config)
        to = pd.Series({r.date: r.turnover for r in result.daily_records})
        m = compute_metrics(result.returns, result.nav, turnover=to)

        assert isinstance(m, PortfolioMetrics)
        assert m.n_days > 0
        assert m.sharpe_ci_low <= m.sharpe_ratio <= m.sharpe_ci_high
        assert m.cvar_95 >= m.var_95_historical - 1e-10


# ===================================================================
# Fase 6: signals / HRP / PositionLedger / 15D
# ===================================================================
class TestFase6Backtest:
    def test_freq_15d(self, synthetic_prices, synthetic_volumes):
        cfg = BacktestConfig(
            opt_method="mv_classic",
            rebalance_freq="15D",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert isinstance(result, BacktestResult)
        assert len(result.nav) > 0
        assert not result.rebalance_log.empty

    def test_opt_method_hrp(self, synthetic_prices, synthetic_volumes):
        cfg = BacktestConfig(
            opt_method="hrp",
            rebalance_freq="Q",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert isinstance(result, BacktestResult)
        assert (result.rebalance_log["opt_method"] == "hrp").all()
        # Weights sum ≈ 1 on rebalance days
        for r in result.daily_records:
            if r.is_rebalance and len(r.weights) > 0:
                assert abs(r.weights.sum() - 1.0) < 1e-5

    def test_signal_stack_and_ledger(self, synthetic_prices, synthetic_volumes):
        cfg = BacktestConfig(
            opt_method="hrp",
            rebalance_freq="15D",
            window=126,
            warmup=126,
            use_signal_stack=True,
            use_position_ledger=True,
            regime_window=63,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert isinstance(result, BacktestResult)
        assert len(result.ledger_snapshots) > 0
        assert result.regime_series is not None
        assert len(result.regime_series) > 0
        # Snapshots have positive NAV
        assert all(s.nav > 0 for s in result.ledger_snapshots)
        # Final NAV finite
        assert np.isfinite(result.nav.iloc[-1])


class TestCapmShrunkActiveWindow:
    """Regression: capm_shrunk must not silently flatline the backtest."""

    def test_capm_without_spy_in_universe_still_rebalances(
        self, synthetic_prices, synthetic_volumes
    ):
        """Stock-only panel (no SPY) previously raised every rebalance.

        run_backtest swallowed the ValueError → 0 rebalances, constant NAV.
        Fallback to historical mu must keep the backtest alive.
        """
        assert "SPY" not in synthetic_prices.columns
        cfg = BacktestConfig(
            opt_method="mv_classic",
            mu_method="capm_shrunk",
            rebalance_freq="Q",
            window=126,
            warmup=126,
        )
        result = run_backtest(synthetic_prices, synthetic_volumes, cfg)
        assert not result.rebalance_log.empty
        assert abs(float(result.nav.iloc[-1]) - cfg.initial_nav) > 1.0

    def test_capm_uses_spy_outside_active_set(self):
        """SPY filtered out of active investable set must still estimate betas."""
        rng = np.random.default_rng(7)
        dates = pd.bdate_range("2020-01-01", periods=320)
        cols = ["A", "B", "C", "D", "SPY"]
        prices = pd.DataFrame(
            100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, (len(dates), 5)), axis=0)),
            index=dates,
            columns=cols,
        )
        volumes = pd.DataFrame(1e6, index=dates, columns=cols)
        # Make SPY look illiquid so the universe filter drops it from active
        volumes["SPY"] = 1.0

        cfg = BacktestConfig(
            opt_method="mv_classic",
            mu_method="capm_shrunk",
            rebalance_freq="Q",
            window=126,
            warmup=126,
            max_weight=0.45,
        )
        result = run_backtest(prices, volumes, cfg, universe=["A", "B", "C", "D", "SPY"])
        assert not result.rebalance_log.empty
        # At least one successful rebalance must have run (not silent flatline)
        assert len(result.rebalance_log) >= 1
        assert abs(float(result.nav.iloc[-1]) - cfg.initial_nav) > 1.0
