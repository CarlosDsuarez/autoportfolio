"""
momentum_replacement.py — T-20: Regla de reemplazo por momentum.

Estrategia de reemplazo momentum 12-1 (R-07):
  1. Calcular retorno 12-1 para todos los activos del universo ampliado.
  2. Excluir el 10% de peor desempeño del portafolio actual.
  3. Agregar el 10% superior del universo (no en portafolio) que cumpla R-03.
  4. Registrar historial de reemplazos.

Retorno 12-1: retorno acumulado de t-252 a t-21 (excluye último mes
para evitar reversión de corto plazo).

Restricciones cubiertas: R-03 (liquidez), R-05 (sin look-ahead), R-07.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("momentum_replacement")

SEED = 42
TRADING_DAYS = 252
_DEFAULT_REPLACE_PCT = 0.10   # replace bottom 10%
_DEFAULT_SKIP_MONTH = 21      # skip last 21 days (short-term reversal)
_DEFAULT_LOOKBACK = 252       # 12-month lookback


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class ReplacementResult:
    """Output of a single replacement operation."""

    t: pd.Timestamp
    removed_tickers: List[str]
    added_tickers: List[str]
    new_universe: List[str]
    momentum_scores: pd.Series      # Score of all universe tickers at t
    replacement_log: pd.DataFrame   # Details of each swap


# ===================================================================
# Momentum computation
# ===================================================================
def compute_momentum_scores(
    prices: pd.DataFrame,
    t: pd.Timestamp,
    lookback: int = _DEFAULT_LOOKBACK,
    skip_month: int = _DEFAULT_SKIP_MONTH,
) -> pd.Series:
    """Compute momentum 12-1 scores for all tickers at date t.

    Momentum_i = cumulative return from (t - lookback) to (t - skip_month).
    CAUTION: Uses only data strictly before t (R-05, no look-ahead).

    Args:
        prices: Full clean price DataFrame.
        t: Reference date (exclusive upper bound).
        lookback: Lookback window in trading days (default 252 = 12m).
        skip_month: Days to skip at the recent end (default 21 = 1m).

    Returns:
        Series of momentum scores indexed by ticker.
        NaN if insufficient history.
    """
    # R-05: strictly use data before t
    past = prices[prices.index < t]
    if len(past) < lookback + skip_month:
        logger.warning(
            "Insufficient history for momentum at %s "
            "(%d rows, need %d).",
            t.date(), len(past), lookback + skip_month,
        )
        return pd.Series(np.nan, index=prices.columns)

    # Price at start of momentum window
    p_start = past.iloc[-(lookback + skip_month)]
    # Price at end (skip last `skip_month` days)
    p_end = past.iloc[-skip_month] if skip_month > 0 else past.iloc[-1]

    momentum = p_end / p_start - 1.0
    momentum.name = f"momentum_{lookback}_{skip_month}"
    return momentum


# ===================================================================
# Replacement rule
# ===================================================================
def momentum_replace(
    t: pd.Timestamp,
    current_portfolio: List[str],
    universe: List[str],
    prices: pd.DataFrame,
    active_tickers: Optional[List[str]] = None,
    replace_pct: float = _DEFAULT_REPLACE_PCT,
    lookback: int = _DEFAULT_LOOKBACK,
    skip_month: int = _DEFAULT_SKIP_MONTH,
    min_candidates: int = 3,
) -> ReplacementResult:
    """Apply momentum replacement rule (R-07).

    Steps:
      1. Compute momentum scores for all universe tickers.
      2. Within current portfolio, rank by momentum.
         Remove bottom `replace_pct` (worst momentum).
      3. From non-portfolio universe (filtered by active_tickers for
         liquidity), add top `replace_pct` by momentum.
      4. Enforce: new tickers must be in active_tickers (R-03).

    Args:
        t: Current rebalance date.
        current_portfolio: Tickers currently in portfolio.
        universe: Full available ticker universe.
        prices: Full clean price DataFrame.
        active_tickers: Liquid tickers at t (from universe_filter).
                        None → use all universe.
        replace_pct: Fraction of portfolio to replace (default 10%).
        lookback: Momentum lookback in trading days.
        skip_month: Recent days to skip in momentum computation.
        min_candidates: Minimum replacement candidates required.
                        If fewer available, reduce n_replace.

    Returns:
        ReplacementResult with new universe and replacement log.
    """
    if active_tickers is None:
        active_tickers = universe

    # Compute momentum scores for entire universe
    scores = compute_momentum_scores(prices, t, lookback, skip_month)

    # Number to replace
    n_in_portfolio = len(current_portfolio)
    n_replace = max(1, int(np.floor(n_in_portfolio * replace_pct)))

    # Rank portfolio members by momentum (ascending = worst first)
    portfolio_scores = scores.reindex(current_portfolio).dropna()
    if portfolio_scores.empty:
        logger.warning(
            "No momentum scores for portfolio at %s.", t.date()
        )
        return ReplacementResult(
            t=t,
            removed_tickers=[],
            added_tickers=[],
            new_universe=current_portfolio,
            momentum_scores=scores,
            replacement_log=pd.DataFrame(),
        )

    worst_performers = portfolio_scores.nsmallest(n_replace).index.tolist()

    # Candidate replacements: in universe + liquid + not in portfolio
    portfolio_set = set(current_portfolio)
    candidates = [
        tk for tk in active_tickers
        if tk not in portfolio_set and tk in scores.index
        and not np.isnan(scores[tk])
    ]

    # Sort candidates by momentum (descending)
    if len(candidates) < min_candidates:
        logger.warning(
            "Only %d replacement candidates at %s "
            "(need >= %d). Reducing n_replace.",
            len(candidates), t.date(), min_candidates,
        )
        n_replace = min(n_replace, len(candidates))

    candidate_scores = scores.reindex(candidates).dropna()
    best_candidates = candidate_scores.nlargest(n_replace).index.tolist()

    # Build new universe
    new_universe = [
        tk for tk in current_portfolio if tk not in worst_performers
    ] + best_candidates

    # Replacement log
    records = []
    for out_tk, in_tk in zip(worst_performers, best_candidates):
        records.append({
            "date": t,
            "removed": out_tk,
            "removed_momentum": portfolio_scores.get(out_tk, np.nan),
            "added": in_tk,
            "added_momentum": candidate_scores.get(in_tk, np.nan),
            "rule": "momentum",
        })
    log_df = pd.DataFrame(records)

    logger.info(
        "momentum_replace(%s): removed=%s, added=%s",
        t.date(), worst_performers, best_candidates,
    )
    return ReplacementResult(
        t=t,
        removed_tickers=worst_performers,
        added_tickers=best_candidates,
        new_universe=new_universe,
        momentum_scores=scores,
        replacement_log=log_df,
    )
