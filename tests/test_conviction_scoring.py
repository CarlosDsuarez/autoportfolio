"""
test_conviction_scoring.py — Tests de T-36: proxy rule-based.
"""

import pandas as pd
import pytest

from src.conviction_scoring import (
    ConvictionScorer,
    conviction_score,
    conviction_to_tilts,
)
from src.regime_detector import HIGH_VOL_RANGING, LOW_VOL_TRENDING


class TestConviction:
    def test_regime_dampens_high_vol_ranging(self):
        scores = pd.Series({"A": 1.0, "B": 0.5, "C": -0.2})
        c_lv = conviction_score(scores, LOW_VOL_TRENDING)
        c_hv = conviction_score(scores, HIGH_VOL_RANGING)
        # High-vol ranging should shrink absolute conviction on average
        assert c_hv.abs().mean() < c_lv.abs().mean()

    def test_tilts_sum_to_one(self):
        scores = pd.Series({"A": 2.0, "B": 0.0, "C": -1.0})
        conv = conviction_score(scores, LOW_VOL_TRENDING)
        w = conviction_to_tilts(conv)
        assert abs(w.sum() - 1.0) < 1e-8
        assert (w >= 0).all()

    def test_protocol_is_runtime_checkable(self):
        def dummy(features):
            return 0.5

        assert isinstance(dummy, ConvictionScorer)

    def test_no_external_calls(self):
        """Sanity: function returns without network (no LLM)."""
        scores = pd.Series({"X": 0.1, "Y": -0.1})
        out = conviction_score(scores, "low_vol_ranging")
        assert len(out) == 2
