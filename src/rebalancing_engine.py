"""
rebalancing_engine.py — T-24: Motor de reequilibrio integrado.

Función central:
  rebalance(t, w_current, w_target, rule, ...) →
    {w_new, trades, cost, actual_turnover}

Integra:
  - T-19: drift_monitor (detección de deriva)
  - T-20: momentum_replacement
  - T-21: random_replacement
  - T-22: min_variance_replacement
  - T-23: order_execution (TWAP + market impact)

Restricciones cubiertas: R-01 (costos), R-02 (turnover), R-07 (reglas).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from src.drift_monitor import check_drift, DriftReport
from src.momentum_replacement import momentum_replace
from src.random_replacement import random_replace
from src.min_variance_replacement import min_variance_replace
from src.order_execution import execute_rebalance, ExecutionResult

logger = logging.getLogger("rebalancing_engine")

SEED = 42
TRADING_DAYS = 252
_DEFAULT_DRIFT_THRESHOLD = 0.05
_DEFAULT_TURNOVER_LIMIT = 0.20
_DEFAULT_PORTFOLIO_VALUE = 1_000_000.0
_DEFAULT_REPLACE_PCT = 0.10


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class RebalanceResult:
    """Output of a full rebalance operation."""

    t: pd.Timestamp
    w_before: pd.Series            # Weights before rebalance
    w_target: pd.Series            # Optimizer target weights
    w_new: pd.Series               # Final post-rebalance weights
    trades: pd.DataFrame           # Per-asset trade details
    cost: float                    # Total execution cost (dollars)
    cost_bps: float                # Cost in basis points
    actual_turnover: float         # sum(|w_new - w_before|)
    rule_applied: str              # Replacement rule used
    universe_changed: bool         # Whether universe was updated
    removed_tickers: List[str]
    added_tickers: List[str]
    drift_report: Optional[DriftReport]
    execution_result: Optional[ExecutionResult]


# ===================================================================
# Main rebalancing function
# ===================================================================
def rebalance(
    t: pd.Timestamp,
    w_current: pd.Series,
    w_target: pd.Series,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    rule: str = "none",
    universe: Optional[List[str]] = None,
    active_tickers: Optional[List[str]] = None,
    sigma: Optional[pd.DataFrame] = None,
    portfolio_value: float = _DEFAULT_PORTFOLIO_VALUE,
    drift_threshold: float = _DEFAULT_DRIFT_THRESHOLD,
    turnover_limit: float = _DEFAULT_TURNOVER_LIMIT,
    replace_pct: float = _DEFAULT_REPLACE_PCT,
    is_rebalance_date: bool = True,
    # Order execution params
    n_slices: int = 10,
    commission_bps: float = 5.0,
    spread_bps: float = 2.0,
    impact_coef: float = 0.1,
) -> RebalanceResult:
    """Execute a full rebalance with replacement rule and cost modeling.

    Steps:
      1. Check drift (T-19) — determine if rebalance is needed.
      2. Apply replacement rule if triggered (T-20/21/22).
      3. Cap w_new to enforce turnover limit (R-02).
      4. Execute orders via TWAP model (T-23) and record costs.

    Args:
        t: Rebalance date.
        w_current: Current portfolio weights (post-drift).
        w_target: Optimizer-suggested target weights.
        prices: Full clean price DataFrame.
        volumes: Full volume DataFrame.
        rule: Replacement rule: 'momentum', 'random', 'min_variance',
              or 'none'.
        universe: Full ticker universe (for replacement candidates).
        active_tickers: Liquid tickers at t (R-03).
        sigma: Covariance matrix (for min_variance rule).
        portfolio_value: Total portfolio value in dollars.
        drift_threshold: Per-asset drift trigger (R-07).
        turnover_limit: Maximum allowed turnover (R-02, default 20%).
        replace_pct: Fraction of portfolio to replace.
        is_rebalance_date: True if this is a scheduled rebalance.
        n_slices: TWAP slices for order execution.
        commission_bps: Commission rate in bps.
        spread_bps: Half bid-ask spread in bps.
        impact_coef: Linear market impact coefficient.

    Returns:
        RebalanceResult with full audit trail.

    Raises:
        ValueError: If rule is not recognized.
    """
    valid_rules = ("none", "momentum", "random", "min_variance")
    if rule not in valid_rules:
        raise ValueError(
            f"rule must be one of {valid_rules}, got '{rule}'"
        )

    tickers = w_target.index.tolist()
    wc = w_current.reindex(tickers).fillna(0.0)
    wt = w_target.copy()

    # --- Step 1: Drift check (T-19) ---
    drift_report = check_drift(
        t=t,
        w_current=wc,
        w_target=wt,
        drift_threshold=drift_threshold,
        is_rebalance_date=is_rebalance_date,
    )

    removed_tickers: List[str] = []
    added_tickers: List[str] = []
    universe_changed = False

    # --- Step 2: Apply replacement rule (T-20/21/22) ---
    current_portfolio = tickers

    if drift_report.should_rebalance and rule != "none":
        if universe is None:
            universe = tickers

        if rule == "momentum":
            res = momentum_replace(
                t=t,
                current_portfolio=current_portfolio,
                universe=universe,
                prices=prices,
                active_tickers=active_tickers,
                replace_pct=replace_pct,
            )
        elif rule == "random":
            res = random_replace(
                t=t,
                current_portfolio=current_portfolio,
                universe=universe,
                active_tickers=active_tickers,
                replace_pct=replace_pct,
            )
        else:  # min_variance
            res = min_variance_replace(
                t=t,
                current_portfolio=current_portfolio,
                universe=universe,
                prices=prices,
                w_current=wc,
                sigma=sigma,
                active_tickers=active_tickers,
                replace_pct=replace_pct,
            )

        removed_tickers = res.removed_tickers
        added_tickers = res.added_tickers
        universe_changed = len(removed_tickers) > 0 or len(added_tickers) > 0

        # Adjust target weights for new universe
        if universe_changed:
            new_tickers = res.new_universe
            # Re-normalize target weights over new universe
            wt_new = wt.reindex(new_tickers).fillna(0.0)
            # Removed tickers get 0, added tickers get equal share of removed
            total_removed_w = sum(wt.get(tk, 0.0) for tk in removed_tickers)
            n_added = len(added_tickers)
            for tk in added_tickers:
                wt_new[tk] = (
                    total_removed_w / n_added if n_added > 0 else 0.0
                )
            s = wt_new.sum()
            wt = wt_new / s if s > 1e-10 else pd.Series(
                np.ones(len(new_tickers)) / len(new_tickers),
                index=new_tickers,
            )
            # Expand wc to new tickers
            wc = wc.reindex(new_tickers).fillna(0.0)
            s = wc.sum()
            if s > 1e-10:
                wc /= s

    # --- Step 3: Apply turnover cap (R-02) ---
    if drift_report.should_rebalance:
        w_new = _apply_turnover_cap(wc, wt, turnover_limit)
    else:
        w_new = wc.copy()

    # Normalize
    s = w_new.sum()
    w_new = w_new / s if s > 1e-10 else wt.copy()

    # --- Step 4: Execute orders (T-23) ---
    execution: Optional[ExecutionResult] = None
    cost = 0.0
    cost_bps = 0.0
    if drift_report.should_rebalance:
        try:
            # Align both series to new ticker set
            all_tickers = w_new.index.tolist()
            wc_exec = wc.reindex(all_tickers).fillna(0.0)
            s = wc_exec.sum()
            if s > 1e-10:
                wc_exec /= s
            execution = execute_rebalance(
                t=t,
                w_current=wc_exec,
                w_target=w_new,
                portfolio_value=portfolio_value,
                prices=prices,
                volumes=volumes,
                n_slices=n_slices,
                commission_bps=commission_bps,
                spread_bps=spread_bps,
                impact_coef=impact_coef,
            )
            cost = execution.total_cost
            cost_bps = execution.total_cost_bps
            trades_df = execution.order_df
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Order execution failed at %s: %s", t.date(), exc
            )
            trades_df = pd.DataFrame()
    else:
        trades_df = pd.DataFrame()

    # Actual turnover
    actual_turnover = float(
        np.sum(np.abs(
            w_new.values
            - wc.reindex(w_new.index).fillna(0.0).values
        ))
    )

    logger.info(
        "rebalance(%s, rule=%s): turnover=%.4f, cost=%.4f "
        "(%.2f bps), universe_changed=%s",
        t.date(), rule, actual_turnover, cost, cost_bps, universe_changed,
    )
    return RebalanceResult(
        t=t,
        w_before=wc,
        w_target=wt,
        w_new=w_new,
        trades=trades_df,
        cost=round(cost, 6),
        cost_bps=round(cost_bps, 4),
        actual_turnover=round(actual_turnover, 6),
        rule_applied=rule,
        universe_changed=universe_changed,
        removed_tickers=removed_tickers,
        added_tickers=added_tickers,
        drift_report=drift_report,
        execution_result=execution,
    )


# ===================================================================
# Helper: turnover cap
# ===================================================================
def _apply_turnover_cap(
    w_current: pd.Series,
    w_target: pd.Series,
    turnover_limit: float,
) -> pd.Series:
    """Blend w_current and w_target to respect turnover cap (R-02).

    If |w_target - w_current|_1 > turnover_limit, interpolate:
      w_new = w_current + alpha * (w_target - w_current)
    where alpha = turnover_limit / actual_turnover.

    Args:
        w_current: Current weights.
        w_target: Target weights.
        turnover_limit: Maximum L1 turnover.

    Returns:
        Blended weight vector respecting the turnover limit.
    """
    tickers = w_target.index
    wc = w_current.reindex(tickers).fillna(0.0)
    delta = w_target - wc
    l1 = float(delta.abs().sum())

    if l1 <= turnover_limit + 1e-8:
        return w_target.copy()

    alpha = turnover_limit / l1
    w_blended = wc + alpha * delta
    w_blended = w_blended.clip(lower=0.0)
    s = w_blended.sum()
    return w_blended / s if s > 1e-10 else w_target.copy()
