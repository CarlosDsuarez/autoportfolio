"""
test_hrp_sizing.py — Tests de T-35: Hierarchical Risk Parity.
"""

import numpy as np
import pandas as pd
import pytest

from src.hrp_sizing import hrp_weights

SEED = 42


class TestHRP:
    def test_weights_sum_to_one_and_long_only(self):
        rng = np.random.default_rng(SEED)
        n, m = 300, 6
        rets = pd.DataFrame(
            rng.normal(0.0003, 0.01, size=(n, m)),
            columns=[f"A{i}" for i in range(m)],
        )
        w = hrp_weights(rets)
        assert abs(w.sum() - 1.0) < 1e-8
        assert (w >= -1e-12).all()
        assert len(w) == m

    def test_correlated_pair_cluster_treatment(self):
        """Two nearly identical assets should share cluster weight vs a third.

        Construct: A and B almost perfectly correlated (same noise),
        C independent with similar vol. HRP should allocate roughly
        half to {A,B} combined and half to C (risk parity across clusters).
        """
        rng = np.random.default_rng(SEED)
        n = 500
        common = rng.normal(0, 0.01, size=n)
        noise = rng.normal(0, 0.001, size=n)
        c_indep = rng.normal(0, 0.01, size=n)
        rets = pd.DataFrame({
            "A": common + noise,
            "B": common - noise,   # ~corr 1 with A
            "C": c_indep,
        })
        w = hrp_weights(rets)
        assert abs(w.sum() - 1.0) < 1e-8
        assert (w >= 0).all()
        pair = w["A"] + w["B"]
        # Combined pair weight should be close to C's weight (cluster RP)
        assert abs(pair - w["C"]) < 0.15, (
            f"pair={pair:.3f} C={w['C']:.3f} — expected similar cluster weights"
        )
        # Individual A and B each get less than C (split inside cluster)
        assert w["A"] < w["C"] + 0.05
        assert w["B"] < w["C"] + 0.05
