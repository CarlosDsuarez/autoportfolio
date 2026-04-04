"""
liquidity_constraint.py — T-16: Restricción de liquidez en el solver.

Implementa límites de peso proporcionales al volumen (R-03):
  w_i <= vol_share_i * scale   (capped at global_max)
  Activos ilíquidos (Amihud > umbral) → w_i = 0 antes del solver.

Integra con universe_filter.precompute_rolling_metrics para
obtener métricas de liquidez sin look-ahead (R-05).

Restricciones cubiertas: R-03 (liquidez mínima), R-04, R-05.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd

logger = logging.getLogger("liquidity_constraint")

SEED = 42
TRADING_DAYS = 252
_DEFAULT_MAX_WEIGHT = 0.30
_DEFAULT_GAMMA = 1.0
_DEFAULT_AMIHUD_PCTILE = 90.0   # exclude above p90 Amihud
_DEFAULT_VOL_SCALE = 3.0        # w_max_i = vol_share_i * scale
_DEFAULT_VOL_WINDOW = 63        # 3-month volume window
_REG_EPS = 1e-6


def _make_pd(sigma: np.ndarray, eps: float = _REG_EPS) -> np.ndarray:
    return sigma + eps * np.eye(sigma.shape[0])


def compute_volume_upper_bounds(
    tickers: List[str],
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_VOL_WINDOW,
    scale: float = _DEFAULT_VOL_SCALE,
    global_max: float = _DEFAULT_MAX_WEIGHT,
) -> pd.Series:
    """Compute per-asset weight upper bounds from rolling volume.

    w_max_i = min(global_max, mean_vol_i / sum(mean_vols) * scale)

    Uses only data strictly before t (R-05, no look-ahead).

    Args:
        tickers: Tickers to compute bounds for.
        volumes: Full volume DataFrame.
        t: Reference date (exclusive upper bound).
        window: Lookback for volume averaging (trading days).
        scale: Multiplier on volume share.
        global_max: Hard cap on any single weight.

    Returns:
        Series of upper bounds indexed by ticker.
    """
    # R-05: strictly exclude t
    v = volumes[volumes.index < t].iloc[-window:]
    available = [tk for tk in tickers if tk in v.columns]
    if not available:
        logger.warning(
            "No volume data for tickers at %s; using uniform bounds.", t
        )
        return pd.Series(global_max, index=tickers)

    mean_vol = v[available].fillna(0.0).mean()
    total_vol = mean_vol.sum()
    if total_vol < 1e-10:
        logger.warning(
            "Zero total volume at %s; using uniform bounds.", t
        )
        return pd.Series(global_max, index=tickers)

    vol_share = mean_vol / total_vol
    upper = (vol_share * scale).clip(
        lower=1.0 / max(len(tickers), 1) * 0.5,
        upper=global_max,
    )
    # Fill missing tickers with minimum bound
    full = pd.Series(
        upper.reindex(tickers).fillna(1.0 / max(len(tickers), 1) * 0.5)
    )
    return full


def apply_amihud_exclusion(
    tickers: List[str],
    illiquidity_scores: pd.Series,
    percentile_threshold: float = _DEFAULT_AMIHUD_PCTILE,
) -> Tuple[List[str], List[str]]:
    """Exclude most illiquid assets by Amihud score.

    Args:
        tickers: Full ticker list.
        illiquidity_scores: Amihud illiquidity (higher = less liquid).
        percentile_threshold: Exclude above this percentile (e.g. 90).

    Returns:
        Tuple of (active_tickers, excluded_tickers).
    """
    scores = (
        illiquidity_scores.reindex(tickers)
        .fillna(illiquidity_scores.max() if len(illiquidity_scores) else 0.0)
    )
    if len(scores) == 0:
        return tickers, []

    threshold = np.percentile(scores.values, percentile_threshold)
    excluded = scores[scores > threshold].index.tolist()
    active = [tk for tk in tickers if tk not in excluded]

    if len(active) < 3:
        logger.warning(
            "Only %d liquid tickers after Amihud exclusion; "
            "keeping all.", len(active),
        )
        return tickers, []
    return active, excluded


def optimize_with_liquidity(
    mu: pd.Series,
    sigma: pd.DataFrame,
    t: pd.Timestamp,
    volumes: pd.DataFrame,
    illiquidity_scores: Optional[pd.Series] = None,
    w_prev: Optional[pd.Series] = None,
    volume_window: int = _DEFAULT_VOL_WINDOW,
    volume_scale: float = _DEFAULT_VOL_SCALE,
    global_max_weight: float = _DEFAULT_MAX_WEIGHT,
    amihud_percentile: float = _DEFAULT_AMIHUD_PCTILE,
    risk_aversion: float = _DEFAULT_GAMMA,
) -> Dict:
    """Optimize portfolio with dynamic liquidity constraints (R-03).

    Steps:
      1. Exclude illiquid assets by Amihud threshold.
      2. Compute per-asset weight upper bounds from volume.
      3. Solve MV optimization on active subset.
      4. Map weights back to full universe (zero for excluded).

    Args:
        mu: Expected returns (annualized), indexed by ticker.
        sigma: Annualized covariance matrix (n x n DataFrame).
        t: Current rebalance date.
        volumes: Historical volume DataFrame.
        illiquidity_scores: Amihud scores. None → skip Amihud step.
        w_prev: Previous weights (not used in optimization here).
        volume_window: Window for volume averaging.
        volume_scale: Multiplier on volume share for upper bounds.
        global_max_weight: Hard cap on any single weight (R-04).
        amihud_percentile: Exclude above this Amihud percentile.
        risk_aversion: Risk aversion gamma in MV objective.

    Returns:
        Dict with keys:
            weights (pd.Series): Full-universe weights (zeros for excluded).
            active_tickers (list): Tickers included in optimization.
            excluded_tickers (list): Tickers excluded by liquidity.
            expected_sharpe (float): Sharpe ratio on active subset.
            status (str): Solver status.
            solve_time (float): Solve time in seconds.
    """
    all_tickers = mu.index.tolist()

    # Step 1: Amihud exclusion (R-03)
    if illiquidity_scores is not None and len(illiquidity_scores) > 0:
        active_tickers, excluded = apply_amihud_exclusion(
            all_tickers, illiquidity_scores, amihud_percentile
        )
    else:
        active_tickers = all_tickers
        excluded = []

    # Step 2: Volume-proportional upper bounds (R-03)
    upper_bounds = compute_volume_upper_bounds(
        active_tickers, volumes, t,
        window=volume_window,
        scale=volume_scale,
        global_max=global_max_weight,
    )

    n = len(active_tickers)
    mu_active = mu.reindex(active_tickers).values.astype(float)
    sigma_sub = sigma.reindex(
        index=active_tickers, columns=active_tickers
    )
    sigma_arr = _make_pd(sigma_sub.values.astype(float))
    ub_arr = upper_bounds.reindex(active_tickers).values.astype(float)

    # Step 3: MV optimization with volume bounds
    w = cp.Variable(n, nonneg=True)
    objective = cp.Maximize(
        mu_active @ w
        - (risk_aversion / 2.0) * cp.quad_form(w, sigma_arr)
    )
    constraints = [
        cp.sum(w) == 1,
        w <= ub_arr,
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
            "optimize_with_liquidity: did not converge (status=%s). "
            "Using equal weights on active universe.", status,
        )
        w_active = np.ones(n) / n
    else:
        w_active = np.clip(w.value, 0.0, global_max_weight)
        total = w_active.sum()
        w_active = (
            w_active / total if total > 1e-10 else np.ones(n) / n
        )

    # Step 4: Map back to full universe
    w_full = pd.Series(0.0, index=all_tickers)
    w_active_s = pd.Series(
        np.round(w_active, 4), index=active_tickers
    )
    w_active_s /= w_active_s.sum()
    w_full.update(w_active_s)

    wf = w_full.reindex(active_tickers).values
    port_ret = float(mu_active @ wf)
    port_vol = float(np.sqrt(wf @ sigma_arr @ wf))
    sharpe = port_ret / port_vol if port_vol > 1e-10 else 0.0

    logger.info(
        "optimize_with_liquidity: active=%d, excluded=%d, "
        "Sharpe=%.4f, status=%s",
        len(active_tickers), len(excluded), sharpe, status,
    )
    return {
        "weights": w_full,
        "active_tickers": active_tickers,
        "excluded_tickers": excluded,
        "expected_sharpe": round(sharpe, 6),
        "status": status,
        "solve_time": round(solve_time, 4),
    }
