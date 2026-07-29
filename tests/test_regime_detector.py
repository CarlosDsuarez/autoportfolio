"""
test_regime_detector.py — Tests de T-33: clasificador de régimen.

Verifica:
  (a) high_vol_* en ventana sintética de alta volatilidad
  (b) sin look-ahead: clasificación en t no cambia con datos futuros
"""

import numpy as np
import pandas as pd
import pytest

from src.regime_detector import (
    HIGH_VOL_RANGING,
    HIGH_VOL_TRENDING,
    REGIME_STATES,
    RegimeReport,
    classify_regime,
    rolling_regime_series,
)

SEED = 42


def _make_prices(n: int = 400, n_assets: int = 5, vol: float = 0.01, drift: float = 0.0):
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(drift, vol, size=(n, n_assets))
    arr = 100.0 * np.exp(np.cumsum(rets, axis=0))
    cols = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(arr, index=dates, columns=cols)


class TestRegimeDetector:
    def test_high_vol_detected(self):
        """Synthetic high-vol window should classify as high_vol_*."""
        # Low vol warmup then high vol shock
        rng = np.random.default_rng(SEED)
        n = 400
        n_assets = 5
        dates = pd.bdate_range("2020-01-01", periods=n, freq="B")
        rets = rng.normal(0.0003, 0.005, size=(n, n_assets))
        # Last 80 days: 5x volatility
        rets[-80:] = rng.normal(0.0, 0.04, size=(80, n_assets))
        arr = 100.0 * np.exp(np.cumsum(rets, axis=0))
        prices = pd.DataFrame(arr, index=dates, columns=[f"A{i}" for i in range(n_assets)])

        t = prices.index[-1]
        report = classify_regime(prices, t, window=63)
        assert isinstance(report, RegimeReport)
        assert report.state in REGIME_STATES
        assert report.state.startswith("high_vol_"), (
            f"Expected high_vol_* got {report.state} "
            f"(vol_pct={report.volatility_percentile:.1f})"
        )
        assert report.volatility_percentile >= 70.0

    def test_no_lookahead(self):
        """Classification at t must be invariant to appending future data."""
        prices = _make_prices(n=350, vol=0.012)
        t = prices.index[300]
        report_a = classify_regime(prices.iloc[:301], t, window=63)

        # Append noisy future
        rng = np.random.default_rng(SEED + 7)
        extra_n = 50
        last = prices.iloc[300].values
        extra_rets = rng.normal(0, 0.05, size=(extra_n, prices.shape[1]))
        extra_arr = last * np.exp(np.cumsum(extra_rets, axis=0))
        extra_dates = pd.bdate_range(prices.index[300] + pd.Timedelta(days=1), periods=extra_n, freq="B")
        # Align: start after t
        future = pd.DataFrame(extra_arr, index=extra_dates[:extra_n], columns=prices.columns)
        # Ensure future index strictly after t
        future = future[future.index > t]
        extended = pd.concat([prices.iloc[:301], future])

        report_b = classify_regime(extended, t, window=63)
        assert report_a.state == report_b.state
        assert abs(report_a.volatility_percentile - report_b.volatility_percentile) < 1e-9
        assert abs(report_a.trend_score - report_b.trend_score) < 1e-9

    def test_rolling_series_values_are_valid_states(self):
        prices = _make_prices(n=320, vol=0.01)
        series = rolling_regime_series(prices, window=63, min_history=252, step=5)
        assert len(series) > 0
        assert set(series.unique()).issubset(set(REGIME_STATES))
