"""
robust_optimizer.py — T-13: Optimizador robusto SOCP.

Incertidumbre elipsoidal sobre mu. El retorno "worst-case" es:
  r_wc(w) = mu_hat' w - kappa * ||Omega^{1/2} w||_2

donde Omega = diag(sigma_ii / T) es la incertidumbre estimada.
kappa=0 reduce al caso clásico MV (T-12).

Formulación Charnes-Cooper extendida con penalidad de incertidumbre.
Solver: cvxpy + ECOS.

Restricciones cubiertas: R-04 (long-only, cap).
"""

import logging
import time
from typing import Dict

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger("robust_optimizer")

SEED = 42
TRADING_DAYS = 252
_DEFAULT_MAX_WEIGHT = 0.30
_DEFAULT_RISK_FREE = 0.0
_DEFAULT_KAPPA = 0.1
_REG_EPS = 1e-6


def _make_pd(sigma: np.ndarray, eps: float = _REG_EPS) -> np.ndarray:
    return sigma + eps * np.eye(sigma.shape[0])


def _cholesky_safe(sigma: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sigma)
        eigvals = np.maximum(eigvals, 1e-8)
        return eigvecs @ np.diag(np.sqrt(eigvals))


def robust_optimize(
    mu: pd.Series,
    sigma: pd.DataFrame,
    n_obs: int = TRADING_DAYS,
    kappa: float = _DEFAULT_KAPPA,
    max_weight: float = _DEFAULT_MAX_WEIGHT,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
) -> Dict:
    """Maximize robust worst-case Sharpe via SOCP.

    Charnes-Cooper transform with uncertainty penalty:
      Maximize: (mu - rf)' y - kappa * ||Omega^{1/2} y||_2
      Subject to: ||L' y||_2 <= 1
                  y >= 0
                  y_i <= max_w * sum(y)
                  sum(y) >= 0

    Omega = diag(sigma_ii / n_obs): uncertainty shrinks with T.
    kappa=0 is equivalent to mv_optimize (T-12).

    Args:
        mu: Expected returns (annualized), indexed by ticker.
        sigma: Annualized covariance matrix (n x n DataFrame).
        n_obs: Number of observations used to estimate mu.
        kappa: Uncertainty aversion. 0=classical, 0.1-0.5=robust.
        max_weight: Per-asset weight cap (R-04).
        risk_free_rate: Risk-free rate.

    Returns:
        Dict with keys:
            weights (pd.Series): Optimal weights.
            expected_sharpe (float): Nominal Sharpe at optimal weights.
            robust_sharpe (float): Worst-case Sharpe estimate.
            status (str): Solver status.
            solve_time (float): Solve time in seconds.
    """
    tickers = mu.index.tolist()
    n = len(tickers)
    mu_arr = mu.values.astype(float)
    sigma_arr = _make_pd(sigma.values.astype(float))
    L = _cholesky_safe(sigma_arr)

    # Uncertainty set: Omega_i = sigma_ii / n_obs
    omega_diag = np.diag(sigma_arr) / max(n_obs, 1)
    Omega_sqrt = np.diag(np.sqrt(np.maximum(omega_diag, 1e-12)))

    y = cp.Variable(n, nonneg=True)
    excess_mu = mu_arr - risk_free_rate

    # Robust objective: nominal return - kappa * uncertainty
    if kappa > 0.0:
        robust_obj = (
            excess_mu @ y - kappa * cp.norm(Omega_sqrt @ y, 2)
        )
    else:
        robust_obj = excess_mu @ y

    objective = cp.Maximize(robust_obj)
    constraints = [
        cp.norm(L.T @ y, 2) <= 1,
        y <= max_weight * cp.sum(y),
        cp.sum(y) >= 1e-6,
    ]

    prob = cp.Problem(objective, constraints)
    t0 = time.perf_counter()
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError:
        logger.warning("ECOS failed — retrying with SCS.")
        prob.solve(solver=cp.SCS, verbose=False)
    solve_time = time.perf_counter() - t0

    status = prob.status or "unknown"

    if y.value is None or status not in (
        cp.OPTIMAL, cp.OPTIMAL_INACCURATE
    ):
        logger.warning(
            "robust_optimize: did not converge (kappa=%.2f, status=%s). "
            "Returning equal weights.",
            kappa, status,
        )
        w = pd.Series(np.ones(n) / n, index=tickers)
        return {
            "weights": w,
            "expected_sharpe": 0.0,
            "robust_sharpe": 0.0,
            "status": status,
            "solve_time": round(solve_time, 4),
        }

    y_val = np.maximum(y.value, 0.0)
    denom = y_val.sum()
    w_arr = y_val / denom if denom > 1e-10 else np.ones(n) / n
    w_arr = np.clip(w_arr, 0.0, max_weight)
    w_arr /= w_arr.sum()
    w = pd.Series(np.round(w_arr, 4), index=tickers)
    w /= w.sum()
    wf = w.values

    port_ret = float(mu_arr @ wf)
    port_vol = float(np.sqrt(wf @ sigma_arr @ wf))
    sharpe = (
        (port_ret - risk_free_rate) / port_vol
        if port_vol > 1e-10 else 0.0
    )
    # Worst-case: subtract kappa * uncertainty
    uncertainty = float(
        np.sqrt(wf @ (Omega_sqrt @ Omega_sqrt) @ wf)
    )
    robust_sharpe = (
        (port_ret - risk_free_rate - kappa * uncertainty) / port_vol
        if port_vol > 1e-10 else 0.0
    )

    logger.info(
        "robust_optimize: kappa=%.2f, Sharpe=%.4f, "
        "RobustSharpe=%.4f, status=%s",
        kappa, sharpe, robust_sharpe, status,
    )
    return {
        "weights": w,
        "expected_sharpe": round(sharpe, 6),
        "robust_sharpe": round(robust_sharpe, 6),
        "status": status,
        "solve_time": round(solve_time, 4),
    }
