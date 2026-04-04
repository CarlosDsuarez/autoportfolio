"""
turnover_constraint.py — T-15: Restricción de turnover.

Dos modos (R-02):
  "hard": sum(|w - w_prev|) <= delta (restricción dura).
  "soft": regularización L1 sobre cambios de pesos.

|w - w_prev| se lineariza con variables auxiliares delta_p, delta_n.
Solver: cvxpy + OSQP.

Restricciones cubiertas: R-02 (control de turnover), R-04.
"""

import logging
import time
from typing import Dict, Optional

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger("turnover_constraint")

SEED = 42
TRADING_DAYS = 252
_DEFAULT_DELTA = 0.20
_DEFAULT_MAX_WEIGHT = 0.30
_DEFAULT_GAMMA = 1.0
_DEFAULT_L1_LAMBDA = 0.01
_REG_EPS = 1e-6


def _make_pd(sigma: np.ndarray, eps: float = _REG_EPS) -> np.ndarray:
    return sigma + eps * np.eye(sigma.shape[0])


def optimize_with_turnover(
    mu: pd.Series,
    sigma: pd.DataFrame,
    w_prev: Optional[pd.Series] = None,
    mode: str = "hard",
    delta: float = _DEFAULT_DELTA,
    l1_lambda: float = _DEFAULT_L1_LAMBDA,
    max_weight: float = _DEFAULT_MAX_WEIGHT,
    risk_aversion: float = _DEFAULT_GAMMA,
) -> Dict:
    """Optimize portfolio with turnover constraint (R-02).

    Objective:
      max  mu'w - (gamma/2) * w' Sigma w  [- L1 penalty if soft]
      s.t. sum(w) = 1, w >= 0, w_i <= max_weight
           [hard: sum(|w - w_prev|) <= delta]

    |w - w_prev| linearized as delta_p + delta_n where
    delta_p - delta_n = w - w_prev, delta_p,delta_n >= 0.

    Args:
        mu: Expected returns (annualized), indexed by ticker.
        sigma: Annualized covariance matrix (n x n DataFrame).
        w_prev: Previous weights. None → zero.
        mode: 'hard' or 'soft'.
        delta: Maximum turnover for mode='hard' (R-02, default 0.20).
        l1_lambda: L1 penalty coefficient for mode='soft'.
        max_weight: Per-asset weight cap (R-04).
        risk_aversion: Risk aversion gamma.

    Returns:
        Dict with keys:
            weights (pd.Series): Optimal weights.
            expected_sharpe (float): Sharpe ratio (rf=0).
            turnover (float): Actual sum(|w - w_prev|).
            status (str): Solver status.
            solve_time (float): Solve time in seconds.
            delta_active (bool): True if hard constraint was binding.

    Raises:
        ValueError: If mode not in ('hard', 'soft').
    """
    if mode not in ("hard", "soft"):
        raise ValueError(
            f"mode must be 'hard' or 'soft', got '{mode}'"
        )

    tickers = mu.index.tolist()
    n = len(tickers)
    mu_arr = mu.values.astype(float)
    sigma_arr = _make_pd(sigma.values.astype(float))

    if w_prev is None:
        w_prev_arr = np.zeros(n)
    else:
        w_prev_arr = (
            w_prev.reindex(tickers).fillna(0.0).values.astype(float)
        )

    w = cp.Variable(n, nonneg=True)
    # Auxiliary variables for |w - w_prev| = delta_p + delta_n
    delta_p = cp.Variable(n, nonneg=True)
    delta_n = cp.Variable(n, nonneg=True)

    mv_base = (
        mu_arr @ w
        - (risk_aversion / 2.0) * cp.quad_form(w, sigma_arr)
    )
    base_constraints = [
        cp.sum(w) == 1,
        w <= max_weight,
        # Linearization of |w - w_prev|
        delta_p - delta_n == w - w_prev_arr,
    ]

    if mode == "hard":
        # Hard turnover constraint (R-02)
        objective = cp.Maximize(mv_base)
        constraints = base_constraints + [
            cp.sum(delta_p + delta_n) <= delta,
        ]
    else:  # soft
        l1_pen = l1_lambda * cp.sum(delta_p + delta_n)
        objective = cp.Maximize(mv_base - l1_pen)
        constraints = base_constraints

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
            "optimize_with_turnover: did not converge "
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

    logger.info(
        "optimize_with_turnover(mode=%s): Sharpe=%.4f, "
        "turnover=%.4f, status=%s",
        mode, sharpe, turnover, status,
    )
    return {
        "weights": w_series,
        "expected_sharpe": round(sharpe, 6),
        "turnover": round(turnover, 6),
        "status": status,
        "solve_time": round(solve_time, 4),
        "delta_active": (mode == "hard" and turnover >= delta - 1e-4),
    }
