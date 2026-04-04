"""
min_variance_replacement.py — T-22: Regla de reemplazo por mínima varianza.

Estrategia de reemplazo basada en contribución marginal al riesgo (R-07):
  1. Dentro del portafolio, excluir el activo de mayor contribución
     marginal al riesgo (MCR = dσ_p/dw_i).
  2. Del universo líquido externo, agregar el activo de menor varianza
     residual (volatilidad no explicada por el portafolio).
  3. Tercera variante para comparación estadística en Experimento 3.

MCR_i = (Sigma @ w)_i / sigma_p
Varianza residual = sigma_ii - beta_i^2 * sigma_p^2

Restricciones cubiertas: R-03, R-05, R-07, R-09.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("min_variance_replacement")

SEED = 42
TRADING_DAYS = 252
_DEFAULT_REPLACE_PCT = 0.10
_DEFAULT_LOOKBACK = 63    # 3-month window for variance estimation


@dataclass
class MinVarReplacementResult:
    """Output of a min-variance replacement operation."""

    t: pd.Timestamp
    removed_tickers: List[str]
    added_tickers: List[str]
    new_universe: List[str]
    mcr_scores: pd.Series         # Marginal contribution to risk
    residual_var_scores: pd.Series  # Residual variance of candidates
    replacement_log: pd.DataFrame


def compute_marginal_risk_contribution(
    w: pd.Series,
    sigma: pd.DataFrame,
) -> pd.Series:
    """Compute per-asset marginal contribution to portfolio risk.

    MCR_i = (Sigma @ w)_i / sigma_p

    Interpretation: proportion of total portfolio volatility
    attributable to a marginal increase in w_i.

    Args:
        w: Portfolio weights (indexed by ticker).
        sigma: Annualized covariance matrix (n x n DataFrame).

    Returns:
        Series of MCR values per ticker.
    """
    tickers = w.index.tolist()
    w_arr = w.values.astype(float)
    s_arr = sigma.reindex(index=tickers, columns=tickers).values.astype(float)

    port_var = float(w_arr @ s_arr @ w_arr)
    port_vol = np.sqrt(max(port_var, 1e-10))

    marginal = s_arr @ w_arr  # (n,)
    mcr = marginal / port_vol
    return pd.Series(mcr, index=tickers)


def compute_residual_variance(
    candidates: List[str],
    current_portfolio: List[str],
    prices: pd.DataFrame,
    t: pd.Timestamp,
    lookback: int = _DEFAULT_LOOKBACK,
) -> pd.Series:
    """Estimate residual variance of candidate tickers vs. portfolio.

    Residual variance_i = var(r_i) - beta_i^2 * var(r_portfolio)
    where beta_i = cov(r_i, r_portfolio) / var(r_portfolio).

    Uses only data strictly before t (R-05, no look-ahead).

    Args:
        candidates: Tickers to evaluate.
        current_portfolio: Current portfolio tickers.
        prices: Full clean price DataFrame.
        t: Reference date.
        lookback: Window for variance estimation.

    Returns:
        Series of residual variances indexed by candidate ticker.
    """
    # R-05: strictly exclude t
    past = prices[prices.index < t].iloc[-lookback:]
    if len(past) < 20:
        logger.warning(
            "Insufficient history for residual variance at %s.", t.date()
        )
        return pd.Series(np.nan, index=candidates)

    log_ret = np.log(past / past.shift(1)).dropna()
    available_cands = [c for c in candidates if c in log_ret.columns]
    available_port = [p for p in current_portfolio if p in log_ret.columns]

    if not available_cands or not available_port:
        return pd.Series(np.nan, index=candidates)

    # Equal-weight portfolio returns
    r_port = log_ret[available_port].mean(axis=1)
    var_port = float(r_port.var())
    if var_port < 1e-12:
        return pd.Series(np.nan, index=candidates)

    residuals = {}
    for cand in available_cands:
        r_i = log_ret[cand]
        var_i = float(r_i.var())
        cov_i = float(r_i.cov(r_port))
        beta_i = cov_i / var_port
        residual_var = max(var_i - beta_i**2 * var_port, 0.0)
        residuals[cand] = residual_var

    result = pd.Series(residuals).reindex(candidates)
    return result


def min_variance_replace(
    t: pd.Timestamp,
    current_portfolio: List[str],
    universe: List[str],
    prices: pd.DataFrame,
    w_current: Optional[pd.Series] = None,
    sigma: Optional[pd.DataFrame] = None,
    active_tickers: Optional[List[str]] = None,
    replace_pct: float = _DEFAULT_REPLACE_PCT,
    lookback: int = _DEFAULT_LOOKBACK,
) -> MinVarReplacementResult:
    """Apply min-variance replacement rule (R-07).

    Removes highest MCR assets and adds lowest residual variance
    candidates from the liquid universe.

    Args:
        t: Current rebalance date.
        current_portfolio: Tickers currently in portfolio.
        universe: Full available ticker universe.
        prices: Full clean price DataFrame.
        w_current: Current portfolio weights. None → equal weight.
        sigma: Covariance matrix. None → computed from prices.
        active_tickers: Liquid tickers (R-03). None → all universe.
        replace_pct: Fraction to replace (default 10%).
        lookback: Window for variance/covariance estimation.

    Returns:
        MinVarReplacementResult with new universe and log.
    """
    if active_tickers is None:
        active_tickers = universe

    n_replace = max(1, int(np.floor(len(current_portfolio) * replace_pct)))

    # Default to equal weights
    if w_current is None:
        n = len(current_portfolio)
        w_current = pd.Series(np.ones(n) / n, index=current_portfolio)

    # Compute sigma if not provided (from lookback window)
    if sigma is None:
        # R-05: strictly exclude t
        past = prices[prices.index < t].iloc[-lookback:]
        log_ret = np.log(past / past.shift(1)).dropna()
        port_cols = [c for c in current_portfolio if c in log_ret.columns]
        if port_cols:
            sigma = log_ret[port_cols].cov() * TRADING_DAYS
        else:
            n = len(current_portfolio)
            sigma = pd.DataFrame(
                np.eye(n), index=current_portfolio, columns=current_portfolio
            )

    # Step 1: Identify highest MCR assets in portfolio
    mcr = compute_marginal_risk_contribution(
        w_current.reindex(current_portfolio).fillna(0.0),
        sigma.reindex(index=current_portfolio, columns=current_portfolio),
    )
    worst_mcr = mcr.nlargest(n_replace).index.tolist()

    # Step 2: Identify lowest residual variance candidates
    portfolio_set = set(current_portfolio)
    candidates = [
        tk for tk in active_tickers
        if tk not in portfolio_set
    ]

    if not candidates:
        logger.warning(
            "No candidates for min-variance replacement at %s.", t.date()
        )
        return MinVarReplacementResult(
            t=t,
            removed_tickers=[],
            added_tickers=[],
            new_universe=current_portfolio,
            mcr_scores=mcr,
            residual_var_scores=pd.Series(dtype=float),
            replacement_log=pd.DataFrame(),
        )

    res_var = compute_residual_variance(
        candidates, current_portfolio, prices, t, lookback
    )
    n_add = min(n_replace, len(candidates))
    best_candidates = res_var.dropna().nsmallest(n_add).index.tolist()

    # Build new universe
    new_universe = [
        tk for tk in current_portfolio if tk not in worst_mcr
    ] + best_candidates

    records = [
        {
            "date": t,
            "removed": out_tk,
            "removed_mcr": float(mcr.get(out_tk, np.nan)),
            "added": in_tk,
            "added_res_var": float(res_var.get(in_tk, np.nan)),
            "rule": "min_variance",
        }
        for out_tk, in_tk in zip(worst_mcr, best_candidates)
    ]
    log_df = pd.DataFrame(records)

    logger.info(
        "min_variance_replace(%s): removed=%s, added=%s",
        t.date(), worst_mcr, best_candidates,
    )
    return MinVarReplacementResult(
        t=t,
        removed_tickers=worst_mcr,
        added_tickers=best_candidates,
        new_universe=new_universe,
        mcr_scores=mcr,
        residual_var_scores=res_var,
        replacement_log=log_df,
    )
