"""
signal_stack.py — T-34: Motor multi-factor de señales (precios/volúmenes).

Combina factores calculados sobre el universo OHLCV de autoportfolio
(precios ajustados + volúmenes vía yfinance). Sin fundamentales ni opciones.

Factores implementados:
  - Momentum 12-1  (reusa compute_momentum_scores de momentum_replacement)
  - Mean-reversion (z-score precio vs MA)
  - Volatilidad    (rank inverso de vol realizada → tilt low-vol)
  - Liquidez       (rank inverso Amihud / ADV → tilt líquido)

Factores de ai-hedge-fund NO replicables por falta de datos:
  - Factores fundamentales (P/E, earnings surprise, book-to-market, …)
  - Superficie de volatilidad implícita / skew de opciones
  - Sentiment / conviction vía LLM
  - Short interest / borrow rates
  - Datos macro de alta frecuencia no presentes en data_client

Output: compute_signal_scores(prices, volumes, t) → Series normalizado.

Restricciones cubiertas: R-05 (sin look-ahead), R-03 (liquidez), R-10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.momentum_replacement import compute_momentum_scores

logger = logging.getLogger("signal_stack")

SEED = 42
TRADING_DAYS = 252

_DEFAULT_MR_WINDOW = 21
_DEFAULT_VOL_WINDOW = 63
_DEFAULT_LIQ_WINDOW = 63


# ===================================================================
# Config
# ===================================================================
@dataclass
class SignalStackConfig:
    """Configurable factor weights for the composite signal score.

    Weights are re-normalized to sum to 1 at compute time if they
    do not already. Defaults favour momentum + low-vol.
    """

    w_momentum: float = 0.35
    w_mean_reversion: float = 0.20
    w_volatility: float = 0.25
    w_liquidity: float = 0.20
    mr_window: int = _DEFAULT_MR_WINDOW
    vol_window: int = _DEFAULT_VOL_WINDOW
    liq_window: int = _DEFAULT_LIQ_WINDOW
    momentum_lookback: int = 252
    momentum_skip: int = 21


# ===================================================================
# Cross-sectional normalization
# ===================================================================
def _cs_zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score; returns zeros if std ≈ 0."""
    s = s.replace([np.inf, -np.inf], np.nan)
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).fillna(0.0)


def _cs_rank_to_unit(s: pd.Series) -> pd.Series:
    """Rank to approximately [-1, 1]."""
    r = s.rank(method="average", na_option="keep")
    n = r.notna().sum()
    if n < 2:
        return pd.Series(0.0, index=s.index)
    # percentile in (0,1] → map to [-1, 1]
    pct = r / n
    return ((pct - 0.5) * 2.0).fillna(0.0)


# ===================================================================
# Individual factors (causal: data strictly before t)
# ===================================================================
def _mean_reversion_scores(
    prices: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_MR_WINDOW,
) -> pd.Series:
    """Z-score of price vs rolling MA: negative z → oversold (buy signal).

    Score = -z so that high score prefers mean-reversion longs.
    """
    past = prices[prices.index < t]
    if len(past) < window + 1:
        return pd.Series(np.nan, index=prices.columns)
    ma = past.iloc[-window:].mean()
    std = past.iloc[-window:].std(ddof=1).replace(0, np.nan)
    last = past.iloc[-1]
    z = (last - ma) / std
    return (-z).astype(float)


def _volatility_scores(
    prices: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_VOL_WINDOW,
) -> pd.Series:
    """Inverse realized-vol rank: high score = low volatility (low-vol tilt)."""
    past = prices[prices.index < t]
    if len(past) < window + 1:
        return pd.Series(np.nan, index=prices.columns)
    rets = past.pct_change().iloc[-window:]
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    # Invert: low vol → high score
    return (-vol).astype(float)


def _liquidity_scores(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_LIQ_WINDOW,
) -> pd.Series:
    """Liquidity score from inverse Amihud illiquidity (higher = more liquid).

    Amihud_i = mean(|r_i| / dollar_volume_i) over window.
    Score = -Amihud so liquid names score high.
    """
    past_px = prices[prices.index < t]
    past_vol = volumes[volumes.index < t]
    if len(past_px) < window + 1 or past_vol.empty:
        return pd.Series(np.nan, index=prices.columns)

    px = past_px.iloc[-(window + 1):]
    vol = past_vol.reindex(px.index).iloc[-window:]
    rets = px.pct_change().iloc[-window:]
    # Align columns
    cols = [c for c in prices.columns if c in vol.columns and c in rets.columns]
    if not cols:
        return pd.Series(np.nan, index=prices.columns)

    abs_ret = rets[cols].abs()
    # Dollar volume ≈ price * volume (use mid-window prices)
    mid_px = px[cols].iloc[-window:]
    dvol = (mid_px * vol[cols]).replace(0, np.nan)
    amihud = (abs_ret / dvol).mean()
    # High liquidity = low Amihud → negate
    out = (-amihud).reindex(prices.columns)
    return out.astype(float)


# ===================================================================
# Composite
# ===================================================================
def compute_signal_scores(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    config: Optional[SignalStackConfig] = None,
) -> pd.Series:
    """Compute composite multi-factor signal scores at date t.

    Each factor is cross-sectionally z-scored then combined with
    configurable weights. Output is again z-scored across tickers.

    Args:
        prices: Full clean price DataFrame.
        volumes: Volume DataFrame aligned with prices.
        t: Reference date (exclusive upper bound for factor data, R-05).
        config: SignalStackConfig. None → defaults.

    Returns:
        Series of composite scores indexed by ticker (mean≈0, std≈1).
    """
    if config is None:
        config = SignalStackConfig()

    mom = compute_momentum_scores(
        prices, t,
        lookback=config.momentum_lookback,
        skip_month=config.momentum_skip,
    )
    mr = _mean_reversion_scores(prices, t, window=config.mr_window)
    vol = _volatility_scores(prices, t, window=config.vol_window)
    liq = _liquidity_scores(prices, volumes, t, window=config.liq_window)

    factors = {
        "momentum": (_cs_zscore(mom), config.w_momentum),
        "mean_reversion": (_cs_zscore(mr), config.w_mean_reversion),
        "volatility": (_cs_zscore(vol), config.w_volatility),
        "liquidity": (_cs_zscore(liq), config.w_liquidity),
    }

    w_sum = sum(w for _, w in factors.values())
    if w_sum < 1e-12:
        w_sum = 1.0

    composite = pd.Series(0.0, index=prices.columns, dtype=float)
    for name, (scores, w) in factors.items():
        aligned = scores.reindex(prices.columns).fillna(0.0)
        composite = composite + aligned * (w / w_sum)
        logger.debug("Factor %s weight=%.3f", name, w / w_sum)

    result = _cs_zscore(composite)
    logger.info(
        "signal_scores at %s: mean=%.3f std=%.3f",
        t.date() if hasattr(t, "date") else t,
        float(result.mean()), float(result.std(ddof=1) or 0.0),
    )
    return result
