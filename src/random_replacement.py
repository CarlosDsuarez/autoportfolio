"""
random_replacement.py — T-21: Regla de reemplazo aleatorio.

Baseline experimental para comparación estadística (R-06, R-07):
  1. Excluir aleatoriamente el 10% del portafolio actual.
  2. Agregar aleatoriamente el 10% del universo disponible.
  3. Seed fija (SEED=42) para reproducibilidad exacta entre runs (R-10).

La aleatoriedad es determinista dado (seed, t, portfolio) → mismo
resultado en cualquier ejecución independiente.

Restricciones cubiertas: R-03 (candidatos deben ser líquidos),
R-06 (reproducible), R-07 (variante baseline).
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("random_replacement")

SEED = 42
_DEFAULT_REPLACE_PCT = 0.10


@dataclass
class RandomReplacementResult:
    """Output of a random replacement operation."""

    t: pd.Timestamp
    removed_tickers: List[str]
    added_tickers: List[str]
    new_universe: List[str]
    seed_used: int
    replacement_log: pd.DataFrame


def _date_seed(t: pd.Timestamp, base_seed: int = SEED) -> int:
    """Derive a deterministic seed from date + base_seed.

    Guarantees same replacement choices across independent runs
    (R-10 reproducibility requirement).

    Args:
        t: Reference date.
        base_seed: Global seed constant.

    Returns:
        Integer seed unique to this date.
    """
    date_str = t.strftime("%Y%m%d")
    h = hashlib.md5(f"{base_seed}_{date_str}".encode()).hexdigest()
    return int(h[:8], 16) % (2**31)


def random_replace(
    t: pd.Timestamp,
    current_portfolio: List[str],
    universe: List[str],
    active_tickers: Optional[List[str]] = None,
    replace_pct: float = _DEFAULT_REPLACE_PCT,
    base_seed: int = SEED,
) -> RandomReplacementResult:
    """Apply random replacement rule (R-07 baseline).

    Removes random `replace_pct` of portfolio and adds random tickers
    from the liquid universe not already in portfolio.

    Seed is deterministic per (base_seed, t) so results are identical
    across independent runs (R-10).

    Args:
        t: Current rebalance date.
        current_portfolio: Tickers currently in portfolio.
        universe: Full available ticker universe.
        active_tickers: Liquid tickers at t (R-03 filter).
                        None → use all universe.
        replace_pct: Fraction to replace (default 10%).
        base_seed: Global seed for reproducibility (R-10).

    Returns:
        RandomReplacementResult with new universe and log.
    """
    if active_tickers is None:
        active_tickers = universe

    seed = _date_seed(t, base_seed)
    rng = np.random.default_rng(seed)

    n_replace = max(1, int(np.floor(len(current_portfolio) * replace_pct)))

    # Random removal from portfolio
    portfolio_arr = np.array(current_portfolio)
    n_remove = min(n_replace, len(portfolio_arr))
    remove_idx = rng.choice(len(portfolio_arr), size=n_remove, replace=False)
    removed = portfolio_arr[remove_idx].tolist()

    # Candidates: liquid, not in portfolio
    portfolio_set = set(current_portfolio)
    candidates = [
        tk for tk in active_tickers if tk not in portfolio_set
    ]
    n_add = min(n_replace, len(candidates))
    if n_add == 0:
        logger.warning(
            "No replacement candidates at %s. No swap performed.", t.date()
        )
        return RandomReplacementResult(
            t=t,
            removed_tickers=[],
            added_tickers=[],
            new_universe=current_portfolio,
            seed_used=seed,
            replacement_log=pd.DataFrame(),
        )

    candidates_arr = np.array(candidates)
    add_idx = rng.choice(len(candidates_arr), size=n_add, replace=False)
    added = candidates_arr[add_idx].tolist()

    # Build new universe
    new_universe = [
        tk for tk in current_portfolio if tk not in removed
    ] + added

    # Replacement log
    records = [
        {
            "date": t,
            "removed": out_tk,
            "added": in_tk,
            "rule": "random",
            "seed": seed,
        }
        for out_tk, in_tk in zip(removed, added)
    ]
    log_df = pd.DataFrame(records)

    logger.info(
        "random_replace(%s, seed=%d): removed=%s, added=%s",
        t.date(), seed, removed, added,
    )
    return RandomReplacementResult(
        t=t,
        removed_tickers=removed,
        added_tickers=added,
        new_universe=new_universe,
        seed_used=seed,
        replacement_log=log_df,
    )
