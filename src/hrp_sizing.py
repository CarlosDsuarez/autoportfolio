"""
hrp_sizing.py — T-35: Hierarchical Risk Parity (López de Prado).

Método de sizing alternativo/overlay a portfolio_optimizer:
  1. Distancia de correlación → clustering jerárquico (scipy)
  2. Quasi-diagonalización (orden de hojas del dendrograma)
  3. Bisección recursiva asignando pesos por varianza de cluster

Output compatible con rebalancing_engine.rebalance() (Series de pesos
long-only que suman 1).

Restricciones cubiertas: R-04 (pesos ≥ 0, suman 1), R-10 (determinismo).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger("hrp_sizing")

SEED = 42
_REG_EPS = 1e-10


# ===================================================================
# Helpers
# ===================================================================
def _corr_distance(corr: pd.DataFrame) -> np.ndarray:
    """Convert correlation matrix to condensed distance vector.

    d_ij = sqrt(0.5 * (1 - corr_ij)), clipped to [0, 1].
    """
    c = corr.values.astype(float)
    np.fill_diagonal(c, 1.0)
    c = np.clip(c, -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - c))
    np.fill_diagonal(dist, 0.0)
    # Numerical symmetry
    dist = (dist + dist.T) / 2.0
    return squareform(dist, checks=False)


def _quasi_diag_order(link: np.ndarray) -> List[int]:
    """Return leaf order from hierarchical linkage (quasi-diagonalization)."""
    return list(leaves_list(link))


def _cluster_variance(
    cov: np.ndarray,
    items: List[int],
) -> float:
    """Variance of an inverse-variance weighted cluster portfolio."""
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.maximum(np.diag(sub), _REG_EPS)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def _recursive_bisection(
    cov: np.ndarray,
    sorted_items: List[int],
) -> np.ndarray:
    """Assign HRP weights via recursive cluster bisection.

    Args:
        cov: Full covariance matrix (ndarray).
        sorted_items: Quasi-diagonal leaf order (indices into cov).

    Returns:
        Weight vector aligned to original asset order.
    """
    n = cov.shape[0]
    w = np.ones(n, dtype=float)

    clusters = [sorted_items]
    while clusters:
        # Split each cluster with >1 member into two halves
        new_clusters: List[List[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left = cluster[:mid]
            right = cluster[mid:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right + _REG_EPS)
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= 1.0 - alpha
            if len(left) > 1:
                new_clusters.append(left)
            if len(right) > 1:
                new_clusters.append(right)
        clusters = new_clusters

    # Normalize
    total = w.sum()
    if total > _REG_EPS:
        w = w / total
    else:
        w = np.ones(n) / n
    return w


# ===================================================================
# Public API
# ===================================================================
def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Compute Hierarchical Risk Parity portfolio weights.

    Args:
        returns: DataFrame of asset returns (date × ticker). Columns
            with insufficient variance are dropped then reinserted as 0.

    Returns:
        Series of long-only weights summing to 1, indexed by ticker.
        Compatible with rebalancing_engine.rebalance() target weights.

    Raises:
        ValueError: If fewer than 2 valid assets remain.
    """
    if returns is None or returns.shape[1] < 1:
        raise ValueError("returns must have at least one column.")

    rets = returns.dropna(axis=1, how="all").copy()
    # Drop zero-variance columns
    std = rets.std(ddof=1)
    valid = std[std > _REG_EPS].index.tolist()
    if len(valid) < 2:
        # Degenerate: equal weight on whatever we have
        cols = returns.columns.tolist()
        if not cols:
            raise ValueError("No assets to allocate.")
        logger.warning("HRP degenerate (%d valid) → equal weight.", len(valid))
        return pd.Series(1.0 / len(cols), index=cols)

    rets = rets[valid]
    cov = np.asarray(rets.cov().values, dtype=float).copy()
    corr_df = rets.corr().fillna(0.0)
    corr_arr = np.asarray(corr_df.values, dtype=float).copy()
    np.fill_diagonal(corr_arr, 1.0)
    corr = pd.DataFrame(corr_arr, index=corr_df.index, columns=corr_df.columns)

    dist = _corr_distance(corr)
    link = linkage(dist, method="single")
    order = _quasi_diag_order(link)
    w_arr = _recursive_bisection(cov, order)

    weights = pd.Series(w_arr, index=valid, dtype=float)
    # Reindex to original columns (zeros for dropped)
    weights = weights.reindex(returns.columns).fillna(0.0)
    s = weights.sum()
    if s > _REG_EPS:
        weights = weights / s
    else:
        weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)

    # Enforce long-only numerically
    weights = weights.clip(lower=0.0)
    weights = weights / weights.sum()

    logger.info(
        "HRP weights: n=%d, max=%.3f, min_nonzero=%.4f",
        (weights > 1e-12).sum(),
        float(weights.max()),
        float(weights[weights > 1e-12].min()) if (weights > 1e-12).any() else 0.0,
    )
    return weights
