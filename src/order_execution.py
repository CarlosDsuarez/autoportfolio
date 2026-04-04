"""
order_execution.py — T-23: Simulación de ejecución TWAP.

Modela la ejecución de trades con:
  1. TWAP: fragmenta cada orden en N sub-órdenes equidistantes.
  2. Market impact: modelo lineal proporcional a order_size / ADV.
  3. Slippage simulado: spread bid-ask estimado.
  4. Costo total = comisión + market impact + slippage.

Reduce costo vs. orden en bloque cuando el orden es grande relativo al ADV.

Restricciones cubiertas: R-01 (costos de transacción realistas).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("order_execution")

SEED = 42
_DEFAULT_N_SLICES = 10          # TWAP slices
_DEFAULT_COMMISSION_BPS = 5.0   # 5 bps per side
_DEFAULT_SPREAD_BPS = 2.0       # 2 bps half-spread
_DEFAULT_IMPACT_COEF = 0.1      # linear market impact coefficient
_DEFAULT_ADV_WINDOW = 30        # days for ADV computation


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class OrderResult:
    """Result of executing a single asset order."""

    ticker: str
    order_value: float          # Notional value of order (+ = buy, - = sell)
    n_slices: int
    commission_cost: float      # Commission (bps * |order|)
    spread_cost: float          # Bid-ask spread cost
    market_impact_cost: float   # Price impact cost
    total_cost: float           # Sum of all costs
    avg_fill_price: float       # Average execution price (with slippage)
    cost_bps: float             # Total cost in basis points of |order|


@dataclass
class ExecutionResult:
    """Result of executing a full portfolio rebalance."""

    t: pd.Timestamp
    orders: Dict[str, OrderResult]  # ticker → OrderResult
    total_notional: float
    total_cost: float
    total_cost_bps: float           # Total cost / total notional * 10000
    effective_turnover: float       # sum(|order_value|) / portfolio_value
    order_df: pd.DataFrame          # Summary DataFrame


# ===================================================================
# ADV computation
# ===================================================================
def compute_adv(
    volumes: pd.DataFrame,
    prices: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_ADV_WINDOW,
) -> pd.Series:
    """Compute Average Daily Value (ADV) strictly before t.

    ADV_i = mean(volume_i * price_i) over the lookback window.
    CAUTION: Uses only data before t (R-05, no look-ahead).

    Args:
        volumes: Volume DataFrame.
        prices: Price DataFrame.
        t: Reference date (exclusive upper bound).
        window: Lookback window in trading days.

    Returns:
        Series of ADV values indexed by ticker.
    """
    v = volumes[volumes.index < t].iloc[-window:]
    p = prices[prices.index < t].iloc[-window:]
    common = v.index.intersection(p.index)
    dollar_vol = (v.loc[common] * p.loc[common]).mean()
    return dollar_vol


# ===================================================================
# Single order execution (TWAP + market impact)
# ===================================================================
def execute_order_twap(
    ticker: str,
    order_value: float,         # Notional: positive=buy, negative=sell
    adv: float,                 # Average daily value for this ticker
    price: float,               # Current market price
    n_slices: int = _DEFAULT_N_SLICES,
    commission_bps: float = _DEFAULT_COMMISSION_BPS,
    spread_bps: float = _DEFAULT_SPREAD_BPS,
    impact_coef: float = _DEFAULT_IMPACT_COEF,
) -> OrderResult:
    """Execute a single order via TWAP with market impact model.

    TWAP splits the order into `n_slices` equal parts executed over
    the trading session. Market impact per slice is lower than the
    full order, reducing total cost.

    Market impact (per slice) = impact_coef * (slice_value / adv)
    Total impact = sum over slices (linear in total order size / ADV).

    Args:
        ticker: Asset ticker.
        order_value: Signed notional value to execute.
        adv: Average daily value of the asset (for impact scaling).
        price: Current market price.
        n_slices: Number of TWAP sub-orders.
        commission_bps: Commission in basis points (per side).
        spread_bps: Half-spread in basis points.
        impact_coef: Linear market impact coefficient.

    Returns:
        OrderResult with cost breakdown.
    """
    abs_order = abs(order_value)
    if abs_order < 1e-6:
        return OrderResult(
            ticker=ticker,
            order_value=order_value,
            n_slices=n_slices,
            commission_cost=0.0,
            spread_cost=0.0,
            market_impact_cost=0.0,
            total_cost=0.0,
            avg_fill_price=price,
            cost_bps=0.0,
        )

    slice_value = abs_order / n_slices

    # Commission: flat bps on notional
    commission = abs_order * commission_bps / 10_000.0

    # Spread cost: half-spread on each side
    spread = abs_order * spread_bps / 10_000.0

    # Market impact: linear per slice, lower than block execution
    # Block impact would be: impact_coef * (abs_order / adv)
    # TWAP reduces impact by factor of 1/sqrt(n_slices) (square-root law)
    if adv > 1e-6:
        block_impact_frac = impact_coef * (abs_order / adv)
        twap_impact_frac = block_impact_frac / np.sqrt(n_slices)
    else:
        twap_impact_frac = 0.0
    market_impact = abs_order * twap_impact_frac

    total_cost = commission + spread + market_impact
    cost_bps = (total_cost / abs_order * 10_000.0) if abs_order > 0 else 0.0

    # Average fill price: adjusted for impact
    direction = 1.0 if order_value > 0 else -1.0
    fill_adj = twap_impact_frac * direction
    avg_fill_price = price * (1.0 + fill_adj)

    return OrderResult(
        ticker=ticker,
        order_value=order_value,
        n_slices=n_slices,
        commission_cost=round(commission, 8),
        spread_cost=round(spread, 8),
        market_impact_cost=round(market_impact, 8),
        total_cost=round(total_cost, 8),
        avg_fill_price=round(avg_fill_price, 6),
        cost_bps=round(cost_bps, 4),
    )


# ===================================================================
# Portfolio-level execution
# ===================================================================
def execute_rebalance(
    t: pd.Timestamp,
    w_current: pd.Series,
    w_target: pd.Series,
    portfolio_value: float,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    n_slices: int = _DEFAULT_N_SLICES,
    commission_bps: float = _DEFAULT_COMMISSION_BPS,
    spread_bps: float = _DEFAULT_SPREAD_BPS,
    impact_coef: float = _DEFAULT_IMPACT_COEF,
    adv_window: int = _DEFAULT_ADV_WINDOW,
) -> ExecutionResult:
    """Execute a full portfolio rebalance with TWAP and cost modeling.

    For each asset, computes the trade needed to move from w_current
    to w_target and estimates execution costs via TWAP.

    Args:
        t: Execution date.
        w_current: Current portfolio weights.
        w_target: Target portfolio weights.
        portfolio_value: Total portfolio value in dollars.
        prices: Full price DataFrame.
        volumes: Full volume DataFrame.
        n_slices: TWAP slice count.
        commission_bps: Commission rate per side.
        spread_bps: Half bid-ask spread.
        impact_coef: Market impact coefficient.
        adv_window: ADV lookback window.

    Returns:
        ExecutionResult with per-asset cost breakdown.
    """
    tickers = w_target.index.tolist()
    wc = w_current.reindex(tickers).fillna(0.0)
    wt = w_target.reindex(tickers).fillna(0.0)

    # Trade sizes (positive = buy, negative = sell)
    delta_w = wt - wc
    order_values = delta_w * portfolio_value  # Notional per trade

    # Current prices (strictly before t, R-05)
    p_current = prices[prices.index < t].iloc[-1].reindex(tickers).fillna(100.0)

    # ADV
    adv = compute_adv(volumes, prices, t, window=adv_window)
    adv = adv.reindex(tickers).fillna(portfolio_value * 0.001)

    orders: Dict[str, OrderResult] = {}
    for tk in tickers:
        ov = float(order_values.get(tk, 0.0))
        if abs(ov) < portfolio_value * 1e-5:
            continue  # Skip negligible trades
        res = execute_order_twap(
            ticker=tk,
            order_value=ov,
            adv=float(adv.get(tk, 1.0)),
            price=float(p_current.get(tk, 100.0)),
            n_slices=n_slices,
            commission_bps=commission_bps,
            spread_bps=spread_bps,
            impact_coef=impact_coef,
        )
        orders[tk] = res

    total_notional = sum(abs(o.order_value) for o in orders.values())
    total_cost = sum(o.total_cost for o in orders.values())
    total_cost_bps = (
        total_cost / total_notional * 10_000.0
        if total_notional > 1e-6 else 0.0
    )
    turnover = total_notional / portfolio_value if portfolio_value > 1e-6 else 0.0

    # Summary DataFrame
    rows = [
        {
            "ticker": tk,
            "order_value": o.order_value,
            "commission": o.commission_cost,
            "spread": o.spread_cost,
            "market_impact": o.market_impact_cost,
            "total_cost": o.total_cost,
            "cost_bps": o.cost_bps,
            "avg_fill_price": o.avg_fill_price,
        }
        for tk, o in orders.items()
    ]
    order_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    logger.info(
        "execute_rebalance(%s): %d trades, total_cost=%.4f "
        "(%.2f bps), turnover=%.4f",
        t.date(), len(orders), total_cost, total_cost_bps, turnover,
    )
    return ExecutionResult(
        t=t,
        orders=orders,
        total_notional=round(total_notional, 4),
        total_cost=round(total_cost, 6),
        total_cost_bps=round(total_cost_bps, 4),
        effective_turnover=round(turnover, 6),
        order_df=order_df,
    )
