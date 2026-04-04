"""
optimizer_with_tc.py — T-14: Optimizador con costos de transacción.

Dos modos (R-01):
  "penalty": lambda * ||w - w_prev||^2 penalizado en el objetivo.
  "budget":  sum(|w - w_prev| * c_i) <= budget_tc como restricción.

Usa objetivo cuadrático MV: max mu'w - (gamma/2)*w'Sigma*w - TC.
Solver: cvxpy + OSQP.

Restricciones cubiertas: R-01 (costos de transacción), R-04.
"""

import logging
import time
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger("optimizer_with_tc")

SEED = 42
TRADING_DAYS = 252
_DEFAULT_MAX_WEIGHT = 0.30
_DEFAULT_GAMMA = 1.0         # risk aversion
_DEFAULT_TC_LAMBDA = 0.001   # L2 penalty coefficient
_DEFAULT_BUDGET_TC = 0.01    # 1% total cost budget
_REG_EPS = 1e-6


def _make_pd(sigma: np.ndarray, eps: float = _REG_EPS) -> np.ndarray:
    return sigma + eps * np.eye(sigma.shape[0])


def optimize_with_tc(
    mu: pd.Series,
    sigma: pd.DataFrame,
    w_prev: Optional[pd.Series] = None,
    mode: str = "penalty",
    tc_lambda: float = _DEFAULT_TC_LAMBDA,
    budget_tc: float = _DEFAULT_BUDGET_TC,
    costs: Optional[pd.Series] = None,
    max_weight: float = _DEFAULT_MAX_WEIGHT,
    risk_aversion: float = _DEFAULT_GAMMA,
) -> Dict:
    """Optimize portfolio accounting for transaction costs (R-01).

    Objective (both modes):
      max  mu'w - (gamma/2) * w' Sigma w - TC_term
      s.t. sum(w) = 1, w >= 0, w_i <= max_weight

    Mode "penalty":
      TC_term = tc_lambda * ||w - w_prev||^2   (L2, R-01)

    Mode "budget":
      TC_term = 0  +  constraint: costs' |w - w_prev| <= budget_tc
      |w - w_prev| linearized via delta_p, delta_n >= 0.

    Args:
        mu: Expected returns (annualized), indexed by ticker.
        sigma: Annualized covariance matrix (n x n DataFrame).
        w_prev: Previous weights. None → zero (new portfolio).
        mode: 'penalty' or 'budget'.
        tc_lambda: Penalty coefficient for mode='penalty'.
        budget_tc: Max total TC for mode='budget'.
        costs: Per-asset round-trip cost. None → uniform 0.001.
        max_weight: Per-asset weight cap (R-04).
        risk_aversion: Risk aversion gamma in MV objective.

    Returns:
        Dict with keys:
            weights (pd.Series): Optimal weights.
            expected_return (float): Expected portfolio return.
            expected_vol (float): Expected portfolio volatility.
            expected_sharpe (float): Sharpe ratio (rf=0).
            total_cost (float): Estimated transaction cost.
            turnover (float): sum(|w - w_prev|).
            status (str): Solver status.
            solve_time (float): Solve time in seconds.

    Raises:
        ValueError: If mode is not 'penalty' or 'budget'.
    """
    if mode not in ("penalty", "budget"):
        raise ValueError(
            f"mode must be 'penalty' or 'budget', got '{mode}'"
        )

    tickers = mu.index.tolist()
    n = len(tickers)
    mu_arr = mu.values.astype(float)
    sigma_arr = _make_pd(sigma.values.astype(float))

    # Previous weights (default: zero = brand new portfolio)
    if w_prev is None:
        w_prev_arr = np.zeros(n)
    else:
        w_prev_arr = (
            w_prev.reindex(tickers).fillna(0.0).values.astype(float)
        )

    # Per-asset costs (default: uniform 10 bps each side)
    if costs is None:
        costs_arr = np.full(n, 0.001)
    else:
        costs_arr = (
            costs.reindex(tickers).fillna(0.001).values.astype(float)
        )

    w = cp.Variable(n, nonneg=True)
    mv_base = (
        mu_arr @ w
        - (risk_aversion / 2.0) * cp.quad_form(w, sigma_arr)
    )
    base_constraints = [
        cp.sum(w) == 1,
        w <= max_weight,
    ]

    if mode == "penalty":
        # L2 TC penalty (R-01)
        tc_pen = tc_lambda * cp.sum_squares(w - w_prev_arr)
        objective = cp.Maximize(mv_base - tc_pen)
        constraints = base_constraints
        prob = cp.Problem(objective, constraints)
        t0 = time.perf_counter()
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            prob.solve(solver=cp.SCS, verbose=False)
        solve_time = time.perf_counter() - t0

    else:  # budget
        # Linearize |w - w_prev| via delta_p, delta_n (R-01)
        delta_p = cp.Variable(n, nonneg=True)
        delta_n = cp.Variable(n, nonneg=True)
        objective = cp.Maximize(mv_base)
        constraints = base_constraints + [
            delta_p - delta_n == w - w_prev_arr,
            costs_arr @ (delta_p + delta_n) <= budget_tc,
        ]
        prob = cp.Problem(objective, constraints)
        t0 = time.perf_counter()
        try:
            prob.solve(solver=cp.OSQP, verbose=False)
        except cp.SolverError:
            prob.solve(solver=cp.SCS, verbose=False)
        solve_time = time.perf_counter() - t0

    status = prob.status or "unknown"
    if w.value is None or status not in (
        cp.OPTIMAL, cp.OPTIMAL_INACCURATE
    ):
        logger.warning(
            "optimize_with_tc: did not converge "
            "(mode=%s, status=%s).", mode, status,
        )
        w_arr = np.ones(n) / n
    else:
        w_arr = np.clip(w.value, 0.0, max_weight)
        total = w_arr.sum()
        w_arr = w_arr / total if total > 1e-10 else np.ones(n) / n

    w_series = pd.Series(np.round(w_arr, 4), index=tickers)
    w_series /= w_series.sum()
    wf = w_series.values

    port_ret = float(mu_arr @ wf)
    port_vol = float(np.sqrt(wf @ sigma_arr @ wf))
    sharpe = port_ret / port_vol if port_vol > 1e-10 else 0.0
    turnover = float(np.sum(np.abs(wf - w_prev_arr)))
    total_cost = float(costs_arr @ np.abs(wf - w_prev_arr))

    logger.info(
        "optimize_with_tc(mode=%s): Sharpe=%.4f, turnover=%.4f, "
        "TC=%.4f, status=%s",
        mode, sharpe, turnover, total_cost, status,
    )
    return {
        "weights": w_series,
        "expected_return": round(port_ret, 6),
        "expected_vol": round(port_vol, 6),
        "expected_sharpe": round(sharpe, 6),
        "total_cost": round(total_cost, 6),
        "turnover": round(turnover, 6),
        "status": status,
        "solve_time": round(solve_time, 4),
    }
