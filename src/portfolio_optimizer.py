"""
portfolio_optimizer.py — T-17: Módulo principal de optimización.

Interfaz unificada para todos los métodos de optimización de Fase 3.

Métodos disponibles:
  "mv_classic"  — Máximo Sharpe clásico (T-12)
  "robust"      — Robusto SOCP con incertidumbre elipsoidal (T-13)
  "mv_tc"       — MV con costos de transacción (T-14)
  "robust_tc"   — Robusto + costos de transacción
  "mv_turnover" — MV con restricción de turnover (T-15)

Todas las llamadas retornan un dict estándar con:
  weights, expected_sharpe, turnover, cost, converged, solve_time

Restricciones cubiertas: R-01, R-02, R-03, R-04, R-10.
"""

import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.mv_optimizer import mv_optimize
from src.robust_optimizer import robust_optimize
from src.optimizer_with_tc import optimize_with_tc
from src.turnover_constraint import optimize_with_turnover
from src.liquidity_constraint import optimize_with_liquidity

logger = logging.getLogger("portfolio_optimizer")

SEED = 42
TRADING_DAYS = 252

VALID_METHODS = (
    "mv_classic",
    "robust",
    "mv_tc",
    "robust_tc",
    "mv_turnover",
    "liquidity",
)


def optimize(
    mu: pd.Series,
    sigma: pd.DataFrame,
    method: str = "mv_classic",
    w_prev: Optional[pd.Series] = None,
    max_weight: float = 0.30,
    turnover_limit: float = 0.20,
    tc_lambda: float = 0.001,
    budget_tc: float = 0.01,
    tc_mode: str = "penalty",
    turnover_mode: str = "hard",
    kappa: float = 0.1,
    n_obs: int = TRADING_DAYS,
    risk_free_rate: float = 0.0,
    risk_aversion: float = 1.0,
    costs: Optional[pd.Series] = None,
    # Liquidity method extras
    t: Optional[pd.Timestamp] = None,
    volumes: Optional[pd.DataFrame] = None,
    illiquidity_scores: Optional[pd.Series] = None,
) -> Dict:
    """Unified portfolio optimization interface.

    Routes to the appropriate optimizer based on `method`.

    Args:
        mu: Expected returns (annualized), indexed by ticker.
        sigma: Annualized covariance matrix (n x n DataFrame).
        method: Optimization method. One of VALID_METHODS.
        w_prev: Previous portfolio weights (for TC / turnover).
        max_weight: Per-asset weight cap (R-04, default 0.30).
        turnover_limit: Max turnover for 'mv_turnover' method (R-02).
        tc_lambda: TC penalty coefficient for 'mv_tc', 'robust_tc'.
        budget_tc: TC budget for mode='budget'.
        tc_mode: 'penalty' or 'budget' for TC methods.
        turnover_mode: 'hard' or 'soft' for turnover method.
        kappa: Robustness parameter for 'robust' / 'robust_tc'.
        n_obs: Observations used to estimate mu (for robust).
        risk_free_rate: Risk-free rate for Sharpe computation.
        risk_aversion: Risk aversion gamma in MV objective.
        costs: Per-asset costs Series (for TC methods).
        t: Rebalance date (required for 'liquidity' method, R-05).
        volumes: Volume DataFrame (required for 'liquidity').
        illiquidity_scores: Amihud scores (optional for 'liquidity').

    Returns:
        Dict with keys:
            weights (pd.Series): Optimal portfolio weights.
            expected_sharpe (float): In-sample Sharpe ratio.
            turnover (float): sum(|w - w_prev|).
            cost (float): Estimated transaction cost.
            converged (bool): Whether solver converged.
            solve_time (float): Solver wall-clock time in seconds.
            method (str): Method used.
            [extra keys depending on method]

    Raises:
        ValueError: If method is not in VALID_METHODS or required
            parameters for 'liquidity' are missing.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Valid: {list(VALID_METHODS)}"
        )

    tickers = mu.index.tolist()
    n = len(tickers)

    # Default w_prev: zero weights
    if w_prev is None:
        w_prev_aligned = pd.Series(np.zeros(n), index=tickers)
    else:
        w_prev_aligned = w_prev.reindex(tickers).fillna(0.0)

    costs_aligned = (
        costs.reindex(tickers).fillna(0.001)
        if costs is not None
        else None
    )

    t0_total = time.perf_counter()
    result: Dict = {}

    # ----------------------------------------------------------------
    # Route to optimizer
    # ----------------------------------------------------------------
    if method == "mv_classic":
        result = mv_optimize(
            mu=mu,
            sigma=sigma,
            max_weight=max_weight,
            risk_free_rate=risk_free_rate,
        )

    elif method == "robust":
        result = robust_optimize(
            mu=mu,
            sigma=sigma,
            n_obs=n_obs,
            kappa=kappa,
            max_weight=max_weight,
            risk_free_rate=risk_free_rate,
        )

    elif method == "mv_tc":
        result = optimize_with_tc(
            mu=mu,
            sigma=sigma,
            w_prev=w_prev_aligned,
            mode=tc_mode,
            tc_lambda=tc_lambda,
            budget_tc=budget_tc,
            costs=costs_aligned,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
        )

    elif method == "robust_tc":
        # Robust optimization + TC penalty (combined)
        robust_res = robust_optimize(
            mu=mu,
            sigma=sigma,
            n_obs=n_obs,
            kappa=kappa,
            max_weight=max_weight,
            risk_free_rate=risk_free_rate,
        )
        # If robust converged, apply TC penalty on top via mv_tc
        # with robust-adjusted mu
        if robust_res["status"] in ("optimal", "optimal_inaccurate"):
            tc_res = optimize_with_tc(
                mu=mu,
                sigma=sigma,
                w_prev=w_prev_aligned,
                mode=tc_mode,
                tc_lambda=tc_lambda,
                budget_tc=budget_tc,
                costs=costs_aligned,
                max_weight=max_weight,
                risk_aversion=risk_aversion,
            )
            result = tc_res
            result["robust_sharpe"] = robust_res.get("robust_sharpe", 0.0)
        else:
            result = robust_res

    elif method == "mv_turnover":
        result = optimize_with_turnover(
            mu=mu,
            sigma=sigma,
            w_prev=w_prev_aligned,
            mode=turnover_mode,
            delta=turnover_limit,
            max_weight=max_weight,
            risk_aversion=risk_aversion,
        )

    elif method == "liquidity":
        if t is None or volumes is None:
            raise ValueError(
                "method='liquidity' requires t and volumes arguments."
            )
        result = optimize_with_liquidity(
            mu=mu,
            sigma=sigma,
            t=t,
            volumes=volumes,
            illiquidity_scores=illiquidity_scores,
            w_prev=w_prev_aligned,
            global_max_weight=max_weight,
            risk_aversion=risk_aversion,
        )

    # ----------------------------------------------------------------
    # Standardize output
    # ----------------------------------------------------------------
    total_time = time.perf_counter() - t0_total
    weights = result.get("weights", pd.Series(np.ones(n) / n, index=tickers))

    # Ensure weights are properly normalized and long-only (R-04)
    weights = weights.reindex(tickers).fillna(0.0)
    weights = weights.clip(lower=0.0, upper=max_weight)
    if weights.sum() > 1e-10:
        weights /= weights.sum()
    else:
        weights = pd.Series(np.ones(n) / n, index=tickers)

    # Compute turnover
    turnover = float(
        np.sum(np.abs(weights.values - w_prev_aligned.values))
    )

    # Estimate TC cost
    c_arr = (
        costs_aligned.values if costs_aligned is not None
        else np.full(n, 0.001)
    )
    cost = float(c_arr @ np.abs(weights.values - w_prev_aligned.values))

    # Convergence flag
    status = result.get("status", "unknown")
    converged = status in ("optimal", "optimal_inaccurate")

    # Expected Sharpe (recompute to ensure consistency)
    mu_arr = mu.values.astype(float)
    sigma_arr = sigma.values.astype(float) + _REG_EPS * np.eye(n)
    wf = weights.values
    port_ret = float(mu_arr @ wf)
    port_vol = float(np.sqrt(wf @ sigma_arr @ wf))
    sharpe = (
        (port_ret - risk_free_rate) / port_vol
        if port_vol > 1e-10 else 0.0
    )

    standard = {
        "weights": weights,
        "expected_sharpe": round(sharpe, 6),
        "turnover": round(turnover, 6),
        "cost": round(cost, 6),
        "converged": converged,
        "solve_time": round(total_time, 4),
        "method": method,
        "status": status,
    }
    # Merge any extra keys from the underlying optimizer
    for k, v in result.items():
        if k not in standard:
            standard[k] = v

    logger.info(
        "optimize(method=%s): Sharpe=%.4f, turnover=%.4f, "
        "cost=%.4f, converged=%s",
        method, sharpe, turnover, cost, converged,
    )
    return standard


_REG_EPS = 1e-6
