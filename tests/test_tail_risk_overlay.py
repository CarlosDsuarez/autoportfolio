"""
test_tail_risk_overlay.py — Tests de T-37: solo interfaz (sin opciones).
"""

import pandas as pd
import pytest

from src.tail_risk_overlay import TailRiskOverlay


class TestTailRiskOverlay:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            TailRiskOverlay()  # type: ignore[abstract]

    def test_subclass_must_implement(self):
        class Incomplete(TailRiskOverlay):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_stub_contract(self):
        class Passthrough(TailRiskOverlay):
            def hedge_adjustment(self, portfolio_state, regime):
                return pd.Series(0.0, index=portfolio_state.index)

        overlay = Passthrough()
        w = pd.Series({"A": 0.5, "B": 0.5})
        adj = overlay.hedge_adjustment(w, "high_vol_ranging")
        assert (adj == 0.0).all()

    def test_module_docstring_mentions_options_gap(self):
        import src.tail_risk_overlay as mod
        doc = mod.__doc__ or ""
        assert "IV" in doc or "opciones" in doc.lower() or "options" in doc.lower()
