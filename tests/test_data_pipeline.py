"""
test_data_pipeline.py — T-06: Tests unitarios del pipeline de datos.

Restricciones validadas:
  R-03: Liquidity scores existen y son positivos.
  R-05: Sin look-ahead bias; rango de fechas correcto.
  R-10: Reproducibilidad con seed(42); pipeline modular.

Ejecutar: pytest tests/test_data_pipeline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_client import (
    calc_liquidity_scores,
    calc_log_returns,
    clean_prices,
    load_universe,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
TOLERANCE = 1e-6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_universe_path(tmp_path):
    """Create a minimal universe CSV for testing."""
    csv = tmp_path / "universe.csv"
    csv.write_text(
        "ticker,asset_type,sector,name,inclusion_reason\n"
        "AAPL,stock,Technology,Apple,Test\n"
        "MSFT,stock,Technology,Microsoft,Test\n"
        "SPY,etf,Broad US Equity,SPY ETF,Test\n"
    )
    return str(csv)


@pytest.fixture
def sample_prices():
    """Generate synthetic price data for testing."""
    np.random.seed(SEED)
    dates = pd.bdate_range("2020-01-01", periods=504, freq="B")
    tickers = ["AAPL", "MSFT", "SPY"]
    # Random walk prices starting at 100
    log_rets = np.random.normal(0.0003, 0.015, (504, 3))
    prices_arr = 100 * np.exp(np.cumsum(log_rets, axis=0))
    df = pd.DataFrame(prices_arr, index=dates, columns=tickers)
    df.index.name = "date"
    return df


@pytest.fixture
def sample_volumes():
    """Generate synthetic volume data for testing."""
    np.random.seed(SEED + 1)
    dates = pd.bdate_range("2020-01-01", periods=504, freq="B")
    tickers = ["AAPL", "MSFT", "SPY"]
    vols = np.random.randint(500_000, 10_000_000, (504, 3))
    df = pd.DataFrame(vols, index=dates, columns=tickers, dtype=float)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# T-01: Universe loading tests
# ---------------------------------------------------------------------------
class TestLoadUniverse:
    def test_load_valid_universe(self, sample_universe_path):
        df = load_universe(sample_universe_path)
        assert len(df) == 3
        assert set(df["asset_type"].unique()) == {"stock", "etf"}

    def test_required_columns_present(self, sample_universe_path):
        df = load_universe(sample_universe_path)
        for col in ["ticker", "asset_type", "sector", "name"]:
            assert col in df.columns

    def test_missing_columns_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("ticker,name\nAAPL,Apple\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_universe(str(bad))


# ---------------------------------------------------------------------------
# T-04: Data cleaning tests
# ---------------------------------------------------------------------------
class TestCleanPrices:
    def test_no_nan_in_output(self, sample_prices):
        clean, _ = clean_prices(sample_prices)
        assert clean.isna().sum().sum() == 0

    def test_all_tickers_retained_if_clean(self, sample_prices):
        clean, report = clean_prices(sample_prices)
        # Synthetic data has no missing → all retained
        assert set(clean.columns) == set(sample_prices.columns)

    def test_outlier_detection(self, sample_prices):
        """Inject an extreme outlier and verify it gets flagged."""
        corrupted = sample_prices.copy()
        corrupted.iloc[100, 0] = corrupted.iloc[99, 0] * 5.0  # 400% jump
        _, report = clean_prices(corrupted)
        aapl_row = report[report["ticker"] == "AAPL"].iloc[0]
        assert aapl_row["outliers_flagged"] > 0

    def test_ffill_limited(self, sample_prices):
        """Insert NaN gap > ffill_max and verify ticker handling."""
        gapped = sample_prices.copy()
        # 5 consecutive NaN (ffill_max=2 will leave NaN)
        gapped.iloc[50:55, 0] = np.nan
        clean, _ = clean_prices(gapped, ffill_max=2)
        # Should still work (interpolation covers remaining)
        assert not clean.empty

    def test_midpanel_halt_preserves_calendar(self, sample_prices):
        """Trading halt > ffill_max must not stitch non-consecutive sessions.

        Previously clean_prices used dropna() on rows, removing halt dates for
        *all* tickers. Downstream pct_change then treated a multi-session gap
        as a single-day return (silent return / vol corruption).
        """
        gapped = sample_prices.copy()
        halt = gapped.index[100:104]  # 4 sessions; ffill_max=2 leaves residual NaNs
        gapped.loc[halt, "AAPL"] = np.nan
        clean, report = clean_prices(gapped, ffill_max=2)

        assert not clean.empty
        assert "AAPL" not in clean.columns  # incomplete ticker dropped
        assert set(clean.columns) == {"MSFT", "SPY"}
        # Full calendar retained for surviving tickers (no row stitching)
        assert len(clean) == len(sample_prices)
        assert list(clean.index) == list(sample_prices.index)
        aapl_row = report[report["ticker"] == "AAPL"].iloc[0]
        assert not bool(aapl_row["in_clean_set"])

    def test_late_ipo_does_not_truncate_panel(self, sample_prices):
        """Late-listed ticker under 5% missing must not truncate all history."""
        gapped = sample_prices.copy()
        # ~2% leading NaNs — below the 5% ticker-drop threshold
        gapped.iloc[:10, 0] = np.nan
        clean, _ = clean_prices(gapped, ffill_max=2)
        assert clean.index[0] == sample_prices.index[0]
        assert "AAPL" not in clean.columns
        assert len(clean) == len(sample_prices)

    def test_quality_report_columns(self, sample_prices):
        _, report = clean_prices(sample_prices)
        expected = {
            "ticker", "total_rows", "missing_count", "missing_pct",
            "outliers_flagged", "first_date", "last_date", "in_clean_set",
        }
        assert expected.issubset(set(report.columns))


# ---------------------------------------------------------------------------
# T-04b: Log-returns tests
# ---------------------------------------------------------------------------
class TestLogReturns:
    def test_shape(self, sample_prices):
        rets = calc_log_returns(sample_prices)
        # One row less due to diff
        assert rets.shape == (sample_prices.shape[0] - 1, sample_prices.shape[1])

    def test_no_nan(self, sample_prices):
        rets = calc_log_returns(sample_prices)
        assert rets.isna().sum().sum() == 0

    def test_returns_magnitude(self, sample_prices):
        """Daily log-returns should be small (< 50% absolute)."""
        rets = calc_log_returns(sample_prices)
        assert rets.abs().max().max() < 0.5


# ---------------------------------------------------------------------------
# T-05: Liquidity scores tests (R-03)
# ---------------------------------------------------------------------------
class TestLiquidityScores:
    def test_all_tickers_scored(self, sample_prices, sample_volumes):
        scores = calc_liquidity_scores(sample_prices, sample_volumes, window=30)
        assert len(scores) == sample_prices.shape[1]

    def test_amihud_positive(self, sample_prices, sample_volumes):
        scores = calc_liquidity_scores(sample_prices, sample_volumes, window=30)
        assert (scores["amihud_illiquidity"] >= 0).all()

    def test_volume_positive(self, sample_prices, sample_volumes):
        scores = calc_liquidity_scores(sample_prices, sample_volumes, window=30)
        assert (scores["avg_volume_30d"] > 0).all()

    def test_rank_is_sequential(self, sample_prices, sample_volumes):
        scores = calc_liquidity_scores(sample_prices, sample_volumes, window=30)
        assert scores["liquidity_rank"].max() <= len(scores)


# ---------------------------------------------------------------------------
# R-05: No look-ahead bias validation
# ---------------------------------------------------------------------------
class TestNoLookAhead:
    def test_date_range_respected(self, sample_prices):
        """Ensure clean prices stay within original date range."""
        clean, _ = clean_prices(sample_prices)
        assert clean.index.min() >= sample_prices.index.min()
        assert clean.index.max() <= sample_prices.index.max()

    def test_returns_use_only_past_data(self, sample_prices):
        """Log-returns at t should only use prices at t and t-1."""
        rets = calc_log_returns(sample_prices)
        # Verify first return date is after first price date
        assert rets.index[0] > sample_prices.index[0]


# ---------------------------------------------------------------------------
# R-10: Reproducibility with seed(42)
# ---------------------------------------------------------------------------
class TestReproducibility:
    def test_deterministic_output(self, sample_prices):
        """Same input → same output across runs."""
        clean1, _ = clean_prices(sample_prices)
        clean2, _ = clean_prices(sample_prices)
        pd.testing.assert_frame_equal(clean1, clean2)

    def test_log_returns_deterministic(self, sample_prices):
        rets1 = calc_log_returns(sample_prices)
        rets2 = calc_log_returns(sample_prices)
        pd.testing.assert_frame_equal(rets1, rets2)


# ---------------------------------------------------------------------------
# R-04: Weight constraints preconditions
# ---------------------------------------------------------------------------
class TestWeightPreconditions:
    def test_sufficient_tickers_for_diversification(self, sample_prices):
        """Need at least 2 tickers for meaningful optimization."""
        clean, _ = clean_prices(sample_prices)
        assert clean.shape[1] >= 2

    def test_sufficient_rows_for_estimation(self, sample_prices):
        """Need at least 252 rows for 1-year rolling window."""
        clean, _ = clean_prices(sample_prices)
        assert clean.shape[0] >= 252
