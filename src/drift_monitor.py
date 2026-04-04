"""
drift_monitor.py — T-19: Detector de deriva del portafolio.

Calcula el drift entre pesos actuales (post-mercado) y pesos objetivo,
y determina si se debe reequilibrar.

Triggers de reequilibrio:
  1. Periodico: rebalanceo mensual o trimestral (por calendario).
  2. Umbral de drift: |w_actual_i - w_target_i| > drift_threshold
     en al menos un activo (default 5%).

Restricciones cubiertas: R-07 (reglas de reequilibrio), R-05.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("drift_monitor")

SEED = 42
_DEFAULT_DRIFT_THRESHOLD = 0.05   # 5% per-asset drift trigger
_DEFAULT_TOTAL_DRIFT_THRESHOLD = 0.20  # 20% total portfolio drift


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class DriftReport:
    """Result of a drift check at a given date."""

    t: pd.Timestamp
    w_current: pd.Series       # Drift-affected current weights
    w_target: pd.Series        # Target weights
    drift_vector: pd.Series    # |w_current - w_target| per asset
    max_drift: float           # Maximum per-asset drift
    total_drift: float         # sum(|w_current - w_target|) = L1
    should_rebalance: bool
    trigger_reason: str        # "periodic" | "threshold" | "none"
    breached_tickers: List[str] = field(default_factory=list)


# ===================================================================
# Core functions
# ===================================================================
def compute_drift_weights(
    w_target: pd.Series,
    returns: pd.Series,
) -> pd.Series:
    """Compute portfolio weights after one period of returns.

    Simulates how target weights evolve naturally due to price changes
    without rebalancing.

    Args:
        w_target: Target weights at start of period.
        returns: Simple (arithmetic) returns for the period per asset.

    Returns:
        Updated weights after price drift (normalized to sum to 1).
    """
    tickers = w_target.index
    ret = returns.reindex(tickers).fillna(0.0)
    w_drifted = w_target * (1.0 + ret)
    total = w_drifted.sum()
    if total > 1e-10:
        w_drifted /= total
    else:
        w_drifted = pd.Series(np.ones(len(tickers)) / len(tickers),
                              index=tickers)
    return w_drifted


def check_drift(
    t: pd.Timestamp,
    w_current: pd.Series,
    w_target: pd.Series,
    drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD,
    total_drift_threshold: float = _DEFAULT_TOTAL_DRIFT_THRESHOLD,
    is_rebalance_date: bool = False,
) -> DriftReport:
    """Check whether the portfolio has drifted beyond thresholds.

    Args:
        t: Current date.
        w_current: Current portfolio weights (post-market drift).
        w_target: Target portfolio weights from last optimization.
        drift_threshold: Per-asset drift trigger (R-07, default 5%).
        total_drift_threshold: L1 total drift trigger (default 20%).
        is_rebalance_date: True if this is a scheduled rebalance date.

    Returns:
        DriftReport with drift metrics and rebalance decision.
    """
    tickers = w_target.index.tolist()
    wc = w_current.reindex(tickers).fillna(0.0)
    wt = w_target.reindex(tickers).fillna(0.0)

    drift = (wc - wt).abs()
    max_drift = float(drift.max())
    total_drift = float(drift.sum())
    breached = drift[drift > drift_threshold].index.tolist()

    # Determine trigger
    if is_rebalance_date:
        should_rebalance = True
        reason = "periodic"
    elif max_drift > drift_threshold:
        should_rebalance = True
        reason = "threshold"
    elif total_drift > total_drift_threshold:
        should_rebalance = True
        reason = "threshold"
    else:
        should_rebalance = False
        reason = "none"

    logger.debug(
        "Drift check at %s: max_drift=%.4f, total_drift=%.4f, "
        "should_rebalance=%s (%s)",
        t.date(), max_drift, total_drift, should_rebalance, reason,
    )
    return DriftReport(
        t=t,
        w_current=wc,
        w_target=wt,
        drift_vector=drift,
        max_drift=max_drift,
        total_drift=total_drift,
        should_rebalance=should_rebalance,
        trigger_reason=reason,
        breached_tickers=breached,
    )


def rolling_drift_monitor(
    prices: pd.DataFrame,
    rebalance_dates: List[pd.Timestamp],
    target_weights: Dict[pd.Timestamp, pd.Series],
    drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD,
) -> pd.DataFrame:
    """Monitor drift across the full backtest period.

    At each date, computes current drift-adjusted weights and checks
    whether rebalancing is needed.

    Args:
        prices: Full price DataFrame.
        rebalance_dates: Scheduled rebalance dates (periodic trigger).
        target_weights: Dict {date → target weights} from optimizer.
        drift_threshold: Per-asset drift trigger.

    Returns:
        DataFrame with columns: date, max_drift, total_drift,
        should_rebalance, trigger_reason.
    """
    records = []
    rebal_set = set(rebalance_dates)
    sorted_dates = sorted(target_weights.keys())

    for i, t in enumerate(sorted_dates):
        wt = target_weights[t]
        if i == 0:
            wc = wt.copy()
        else:
            # Evolve previous target weights with subsequent returns
            t_prev = sorted_dates[i - 1]
            try:
                mask = (prices.index > t_prev) & (prices.index <= t)
                p_slice = prices.loc[mask]
                if len(p_slice) >= 2:
                    p_start = prices.loc[prices.index <= t_prev].iloc[-1]
                    p_end = p_slice.iloc[-1]
                    cum_ret = (p_end / p_start - 1.0).reindex(wt.index).fillna(0.0)
                    wc = compute_drift_weights(wt, cum_ret)
                else:
                    wc = wt.copy()
            except Exception:  # noqa: BLE001
                wc = wt.copy()

        report = check_drift(
            t=t,
            w_current=wc,
            w_target=wt,
            drift_threshold=drift_threshold,
            is_rebalance_date=(t in rebal_set),
        )
        records.append({
            "date": t,
            "max_drift": report.max_drift,
            "total_drift": report.total_drift,
            "should_rebalance": report.should_rebalance,
            "trigger_reason": report.trigger_reason,
            "n_breached": len(report.breached_tickers),
        })

    df = pd.DataFrame(records).set_index("date")
    logger.info(
        "Drift monitor: %d periods, %d rebalances triggered",
        len(df), df["should_rebalance"].sum(),
    )
    return df
