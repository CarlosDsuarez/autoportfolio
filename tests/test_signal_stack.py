"""
test_signal_stack.py — Tests de T-34: motor multi-factor.
"""

import numpy as np
import pandas as pd
import pytest

from src.signal_stack import SignalStackConfig, compute_signal_scores

SEED = 42


@pytest.fixture
def panel():
    rng = np.random.default_rng(SEED)
    n, m = 400, 8
    dates = pd.bdate_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(0.0004, 0.015, size=(n, m))
    # Give asset 0 strong momentum
    rets[-252:-21, 0] += 0.003
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rets, axis=0)),
        index=dates,
        columns=[f"T{i}" for i in range(m)],
    )
    volumes = pd.DataFrame(
        rng.integers(1e5, 5e6, size=(n, m)).astype(float),
        index=dates,
        columns=prices.columns,
    )
    return prices, volumes


class TestSignalStack:
    def test_scores_shape_and_finite(self, panel):
        prices, volumes = panel
        t = prices.index[-1]
        scores = compute_signal_scores(prices, volumes, t)
        assert len(scores) == prices.shape[1]
        assert scores.notna().all()
        assert np.isfinite(scores).all()

    def test_config_weights_change_output(self, panel):
        prices, volumes = panel
        t = prices.index[-1]
        s1 = compute_signal_scores(prices, volumes, t, SignalStackConfig(w_momentum=1.0, w_mean_reversion=0.0, w_volatility=0.0, w_liquidity=0.0))
        s2 = compute_signal_scores(prices, volumes, t, SignalStackConfig(w_momentum=0.0, w_mean_reversion=0.0, w_volatility=1.0, w_liquidity=0.0))
        # Not identical under different factor emphasis
        assert not np.allclose(s1.values, s2.values)

    def test_no_lookahead(self, panel):
        prices, volumes = panel
        t = prices.index[300]
        a = compute_signal_scores(prices.iloc[:301], volumes.iloc[:301], t)
        # Append future
        ext_p = prices.copy()
        ext_v = volumes.copy()
        b = compute_signal_scores(ext_p, ext_v, t)
        pd.testing.assert_series_equal(a.sort_index(), b.sort_index())
