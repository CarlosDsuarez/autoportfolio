"""
tail_risk_overlay.py — T-37: Interfaz de Tail Risk Hedge Overlay.

BLOQUEADO POR DATOS: una implementación concreta requiere superficie de
volatilidad implícita (IV surface) / datos de opciones que data_client.py
NO provee (solo precios y volúmenes de equity/ETF vía yfinance).

NO se implementa con datos sintéticos ni mockeados de opciones.
Este módulo expone únicamente la interfaz abstracta para una futura
integración con un proveedor tipo CBOE / ORATS / Polygon Options.

Cuando exista una fuente de IV:
  1. Subclass TailRiskOverlay
  2. Implementar hedge_adjustment(portfolio_state, regime) -> Series
  3. Inyectar en el loop de backtest/live como overlay post-sizing

Restricciones cubiertas: R-05 (no inventar datos futuros/sintéticos de IV).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger("tail_risk_overlay")

SEED = 42


class TailRiskOverlay(ABC):
    """Abstract overlay that adjusts portfolio weights for tail-risk hedging.

    Concrete implementations MUST consume an options/IV data source that is
    not available in the current autoportfolio data pipeline. Do NOT subclass
    with synthetic IV mocks for production backtests — that would invent
    information content the historical simulation does not have.

    Required future data (examples of providers):
      - CBOE LiveVol / DataShop (IV surface, skew, term structure)
      - ORATS (options analytics)
      - Polygon Options / similar vendor APIs

    Method contract:
      hedge_adjustment(portfolio_state, regime) returns a Series of weight
      *deltas* (same index as portfolio_state) that should be added to the
      target weights (e.g. buy put-proxy ETF, reduce cyclical beta). The
      caller is responsible for re-normalizing after applying the overlay.
    """

    @abstractmethod
    def hedge_adjustment(
        self,
        portfolio_state: pd.Series,
        regime: str,
    ) -> pd.Series:
        """Compute weight adjustments for tail-risk hedging.

        Args:
            portfolio_state: Current or target portfolio weights
                (ticker-indexed Series).
            regime: Regime state string from regime_detector
                (e.g. 'high_vol_ranging').

        Returns:
            Series of weight deltas (same index). Sum may be non-zero
            (e.g. allocating to a hedge sleeve financed by cash/trim).

        Raises:
            NotImplementedError: Always on the abstract base; concrete
                subclasses raise if their options data feed is unavailable.
        """
        raise NotImplementedError(
            "TailRiskOverlay requires an options/IV data source "
            "(e.g. CBOE/ORATS/Polygon). No implementation is available "
            "in the current autoportfolio data pipeline."
        )
