"""
position_ledger.py — T-32: Libro de posiciones a nivel de lote (FIFO).

Traduce pesos objetivo del optimizador a cantidades de acciones reales
sin conexión a broker externo. Costeo FIFO para P&L realizado.

Métodos clave:
  - trades_from_target_weights(): pesos → deltas de shares (float)
  - apply_trades(): ejecuta trades con costeo FIFO
  - positions(), nav(), snapshot(), transactions_df()

Usa shares flotantes (no enteros) para evitar ruido de redondeo y
preservar determinismo (R-10) en tests y backtests.

Restricciones cubiertas: R-01 (costos), R-05 (precios en t, sin look-ahead
cuando el caller pasa precios correctos), R-10 (determinismo).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("position_ledger")

SEED = 42


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class Lot:
    """Single purchase lot for FIFO cost basis tracking."""

    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp


@dataclass
class LedgerSnapshot:
    """Point-in-time ledger state for auditability."""

    t: pd.Timestamp
    positions: pd.Series          # shares by ticker
    cash: float
    nav: float
    avg_entry: pd.Series          # average entry price by ticker
    cost_basis: pd.Series         # total cost basis by ticker
    unrealized_pnl: float
    realized_pnl: float


@dataclass
class _TxnRecord:
    """Internal transaction log row."""

    t: pd.Timestamp
    ticker: str
    shares: float          # signed: +buy / -sell
    price: float
    proceeds: float        # signed cash impact before costs
    realized_pnl: float
    cost: float


# ===================================================================
# PositionLedger
# ===================================================================
class PositionLedger:
    """Lot-level position book with FIFO costing (no external broker).

    Args:
        initial_cash: Starting cash balance.
    """

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        self.cash: float = float(initial_cash)
        self._lots: Dict[str, List[Lot]] = {}
        self.realized_pnl: float = 0.0
        self._txns: List[_TxnRecord] = []
        logger.info("PositionLedger init cash=%.2f", self.cash)

    # -----------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------
    def positions(self) -> pd.Series:
        """Return current share quantities by ticker.

        Returns:
            Series of shares indexed by ticker (empty Series if flat).
        """
        data = {
            tk: sum(lot.shares for lot in lots)
            for tk, lots in self._lots.items()
            if lots and sum(lot.shares for lot in lots) > 1e-12
        }
        if not data:
            return pd.Series(dtype=float)
        return pd.Series(data, dtype=float)

    def nav(self, prices: pd.Series) -> float:
        """Mark-to-market NAV = cash + Σ shares_i * price_i.

        Args:
            prices: Series of prices indexed by ticker.

        Returns:
            Total NAV as float.
        """
        pos = self.positions()
        if pos.empty:
            return float(self.cash)
        aligned = pos.reindex(prices.index).fillna(0.0)
        px = prices.reindex(pos.index).fillna(0.0)
        return float(self.cash + float((pos * px).sum()))

    def _avg_entry(self) -> pd.Series:
        """Volume-weighted average entry price per ticker."""
        result: Dict[str, float] = {}
        for tk, lots in self._lots.items():
            total_shares = sum(l.shares for l in lots)
            if total_shares <= 1e-12:
                continue
            cost = sum(l.shares * l.entry_price for l in lots)
            result[tk] = cost / total_shares
        return pd.Series(result, dtype=float) if result else pd.Series(dtype=float)

    def _cost_basis(self) -> pd.Series:
        """Total cost basis (shares * entry) per ticker."""
        result: Dict[str, float] = {}
        for tk, lots in self._lots.items():
            cb = sum(l.shares * l.entry_price for l in lots)
            if abs(cb) > 1e-12:
                result[tk] = cb
        return pd.Series(result, dtype=float) if result else pd.Series(dtype=float)

    def snapshot(self, t: pd.Timestamp, prices: pd.Series) -> LedgerSnapshot:
        """Capture ledger state at date t for audit trail.

        Args:
            t: Snapshot timestamp.
            prices: Mark-to-market prices by ticker.

        Returns:
            LedgerSnapshot with positions, cash, NAV and PnL.
        """
        pos = self.positions()
        avg = self._avg_entry()
        cb = self._cost_basis()
        nav_val = self.nav(prices)
        unrealized = 0.0
        if not pos.empty:
            px = prices.reindex(pos.index).fillna(0.0)
            mkt = (pos * px).sum()
            unrealized = float(mkt - cb.reindex(pos.index).fillna(0.0).sum())
        return LedgerSnapshot(
            t=pd.Timestamp(t),
            positions=pos.copy(),
            cash=float(self.cash),
            nav=float(nav_val),
            avg_entry=avg.copy(),
            cost_basis=cb.copy(),
            unrealized_pnl=unrealized,
            realized_pnl=float(self.realized_pnl),
        )

    def transactions_df(self) -> pd.DataFrame:
        """Return full transaction history as a DataFrame.

        Returns:
            DataFrame with columns: t, ticker, shares, price, proceeds,
            realized_pnl, cost. Empty DataFrame if no trades yet.
        """
        if not self._txns:
            return pd.DataFrame(
                columns=[
                    "t", "ticker", "shares", "price",
                    "proceeds", "realized_pnl", "cost",
                ]
            )
        rows = [
            {
                "t": r.t,
                "ticker": r.ticker,
                "shares": r.shares,
                "price": r.price,
                "proceeds": r.proceeds,
                "realized_pnl": r.realized_pnl,
                "cost": r.cost,
            }
            for r in self._txns
        ]
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Weight → trade translation
    # -----------------------------------------------------------------
    def trades_from_target_weights(
        self,
        target_weights: pd.Series,
        prices_at_t: pd.Series,
        nav: Optional[float] = None,
    ) -> pd.DataFrame:
        """Translate target portfolio weights into share deltas.

        Uses floating-point shares (not integers) for determinism (R-10).

        Args:
            target_weights: Desired weights (should sum ≈ 1). Index = tickers.
            prices_at_t: Prices at rebalance time for valuation.
            nav: Portfolio NAV to size against. None → self.nav(prices_at_t).

        Returns:
            DataFrame with columns [ticker, shares] where shares is the
            signed delta (positive = buy, negative = sell). Only non-zero
            trades are included.
        """
        if nav is None:
            nav = self.nav(prices_at_t)

        tw = target_weights.fillna(0.0).astype(float)
        # Include current holdings that may need to be exited
        current = self.positions()
        all_tickers = sorted(set(tw.index) | set(current.index))

        rows: List[dict] = []
        for tk in all_tickers:
            if tk not in prices_at_t.index:
                continue
            px = float(prices_at_t[tk])
            if not np.isfinite(px) or px <= 0:
                continue
            w_tgt = float(tw[tk]) if tk in tw.index else 0.0
            target_shares = (w_tgt * nav) / px
            curr_shares = float(current[tk]) if tk in current.index else 0.0
            delta = target_shares - curr_shares
            if abs(delta) > 1e-10:
                rows.append({"ticker": tk, "shares": delta})

        if not rows:
            return pd.DataFrame(columns=["ticker", "shares"])
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Trade application (FIFO)
    # -----------------------------------------------------------------
    def apply_trades(
        self,
        trades: pd.DataFrame,
        prices_at_t: pd.Series,
        costs: float = 0.0,
        t: Optional[pd.Timestamp] = None,
    ) -> float:
        """Apply trades with FIFO lot costing.

        Buys create new lots. Sells consume oldest lots first and
        accumulate realized PnL. Transaction costs are deducted from cash.

        Args:
            trades: DataFrame with columns [ticker, shares] (signed deltas).
            prices_at_t: Execution prices by ticker.
            costs: Total transaction cost in currency units (R-01).
            t: Trade timestamp for the log. Defaults to NaT.

        Returns:
            Realized PnL generated by this batch of trades.
        """
        if t is None:
            t = pd.NaT
        else:
            t = pd.Timestamp(t)

        batch_realized = 0.0
        if trades is None or len(trades) == 0:
            if costs > 0:
                self.cash -= float(costs)
            return 0.0

        for _, row in trades.iterrows():
            tk = str(row["ticker"])
            delta = float(row["shares"])
            if abs(delta) < 1e-12:
                continue
            if tk not in prices_at_t.index:
                logger.warning("No price for %s — skipping trade.", tk)
                continue
            px = float(prices_at_t[tk])
            if not np.isfinite(px) or px <= 0:
                logger.warning("Invalid price for %s — skipping.", tk)
                continue

            if delta > 0:
                # Buy: new lot, spend cash
                self._lots.setdefault(tk, []).append(
                    Lot(ticker=tk, shares=delta, entry_price=px, entry_date=t)
                )
                cash_impact = -(delta * px)
                self.cash += cash_impact
                self._txns.append(_TxnRecord(
                    t=t, ticker=tk, shares=delta, price=px,
                    proceeds=cash_impact, realized_pnl=0.0, cost=0.0,
                ))
            else:
                # Sell: FIFO consume lots
                sell_shares = -delta
                pnl = self._fifo_sell(tk, sell_shares, px)
                batch_realized += pnl
                self.realized_pnl += pnl
                cash_impact = sell_shares * px
                self.cash += cash_impact
                self._txns.append(_TxnRecord(
                    t=t, ticker=tk, shares=delta, price=px,
                    proceeds=cash_impact, realized_pnl=pnl, cost=0.0,
                ))

        if costs > 0:
            self.cash -= float(costs)
            # Attribute cost to a synthetic log row if needed
            self._txns.append(_TxnRecord(
                t=t, ticker="__COST__", shares=0.0, price=0.0,
                proceeds=-float(costs), realized_pnl=0.0, cost=float(costs),
            ))

        logger.debug(
            "apply_trades: %d trades, realized_pnl=%.4f, cash=%.2f",
            len(trades), batch_realized, self.cash,
        )
        return batch_realized

    def _fifo_sell(self, ticker: str, shares: float, sell_price: float) -> float:
        """Consume lots FIFO; return realized PnL.

        Args:
            ticker: Asset to sell.
            shares: Number of shares to sell (positive).
            sell_price: Execution price.

        Returns:
            Realized PnL for this sale.

        Raises:
            ValueError: If trying to sell more shares than held.
        """
        lots = self._lots.get(ticker, [])
        remaining = shares
        pnl = 0.0
        while remaining > 1e-12 and lots:
            lot = lots[0]
            take = min(lot.shares, remaining)
            pnl += take * (sell_price - lot.entry_price)
            lot.shares -= take
            remaining -= take
            if lot.shares <= 1e-12:
                lots.pop(0)

        if remaining > 1e-8:
            raise ValueError(
                f"Cannot sell {shares} of {ticker}: "
                f"insufficient shares (short by {remaining:.6f})."
            )
        # Clean empty
        if not lots:
            self._lots.pop(ticker, None)
        return float(pnl)

    def weights_from_positions(self, prices: pd.Series) -> pd.Series:
        """Convert current share holdings to portfolio weights.

        Args:
            prices: Mark-to-market prices.

        Returns:
            Weight Series summing to ≤ 1 (cash is the residual).
            If NAV ≈ 0, returns empty Series.
        """
        pos = self.positions()
        nav_val = self.nav(prices)
        if nav_val < 1e-10 or pos.empty:
            return pd.Series(dtype=float)
        px = prices.reindex(pos.index).fillna(0.0)
        values = pos * px
        return (values / nav_val).astype(float)
