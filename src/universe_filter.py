"""
universe_filter.py — T-11: Filtro dinámico de liquidez.

En cada fecha t, excluye activos que no cumplen el umbral mínimo
de liquidez (Amihud illiquidity ratio). El universo activo varía
por periodo.

Restricciones cubiertas: R-03 (liquidez mínima), R-05 (sin look-ahead).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("universe_filter")

SEED = 42


def compute_rolling_amihud(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling Amihud illiquidity ratio for each ticker at each date.

    Amihud(i, t) = rolling_mean(|r_s| / dollar_volume_s, s in [t-window, t])

    Args:
        prices: Clean price DataFrame.
        volumes: Volume DataFrame aligned with prices.
        window: Rolling lookback in trading days.

    Returns:
        DataFrame of rolling Amihud ratios (same shape as prices).
        Lower values = more liquid.
    """
    log_rets = np.log(prices / prices.shift(1))
    abs_rets = log_rets.abs()

    # Dollar volume
    dollar_vol = volumes * prices
    dollar_vol = dollar_vol.replace(0, np.nan)

    # Amihud ratio per day
    daily_amihud = abs_rets / dollar_vol

    # Rolling mean
    rolling_amihud = daily_amihud.rolling(window=window, min_periods=window // 2).mean()

    return rolling_amihud


def compute_rolling_volume(
    volumes: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Compute rolling average daily volume.

    Args:
        volumes: Volume DataFrame.
        window: Rolling lookback.

    Returns:
        DataFrame of rolling average volumes.
    """
    return volumes.rolling(window=window, min_periods=window // 2).mean()


def filter_universe_at_date(
    t: pd.Timestamp,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    amihud_percentile_threshold: float = 90.0,
    min_volume: float = 100_000,
    window: int = 30,
    rolling_amihud: Optional[pd.DataFrame] = None,
    rolling_avg_vol: Optional[pd.DataFrame] = None,
) -> tuple[list[str], pd.DataFrame]:
    """
    Filter the asset universe at date t based on liquidity criteria.

    Criteria:
      1. Rolling average volume >= min_volume
      2. Amihud illiquidity <= percentile threshold of the universe

    Args:
        t: Reference date (no data after t used).
        prices: Full clean prices.
        volumes: Full volume data.
        amihud_percentile_threshold: Exclude assets above this
            Amihud percentile (default 90 = exclude top 10% illiquid).
        min_volume: Minimum average daily volume threshold.
        window: Lookback window for rolling metrics.
        rolling_amihud: Pre-computed rolling Amihud (optional, for speed).
        rolling_avg_vol: Pre-computed rolling avg volume (optional).

    Returns:
        Tuple of (active_tickers, exclusion_log).
        exclusion_log is a DataFrame with columns [ticker, reason, value].

    R-05: Only uses data at or before t.
    R-03: Enforces minimum liquidity.
    """
    # R-05: only data up to t
    mask = prices.index <= t
    if mask.sum() < window:
        logger.warning(
            "filter_universe_at_date(%s): insufficient data "
            "(%d rows < window=%d). Returning all tickers.",
            t.date(), mask.sum(), window,
        )
        return list(prices.columns), pd.DataFrame()

    # Get Amihud and volume at t
    if rolling_amihud is not None and t in rolling_amihud.index:
        amihud_t = rolling_amihud.loc[t]
    else:
        p_window = prices.loc[mask].iloc[-window:]
        v_window = volumes.loc[mask].iloc[-window:]
        log_rets = np.log(p_window / p_window.shift(1)).dropna()
        abs_rets = log_rets.abs()
        dvol = v_window.iloc[-len(log_rets):] * p_window.iloc[-len(log_rets):]
        dvol = dvol.replace(0, np.nan)
        amihud_t = (abs_rets / dvol).mean()

    if rolling_avg_vol is not None and t in rolling_avg_vol.index:
        avg_vol_t = rolling_avg_vol.loc[t]
    else:
        v_window = volumes.loc[mask].iloc[-window:]
        avg_vol_t = v_window.mean()

    # --- Apply filters ---
    exclusions = []
    all_tickers = list(prices.columns)

    # Filter 1: minimum volume (NaN/inf avg volume must also be excluded —
    # `NaN < min_volume` is False, so a bare `<` let broken yfinance volume
    # series into the active set and downstream ADV→impact path).
    vol_missing = ~np.isfinite(avg_vol_t)
    vol_too_low = avg_vol_t < min_volume
    low_volume = avg_vol_t[vol_missing | vol_too_low]
    for ticker in low_volume.index:
        val = avg_vol_t[ticker]
        exclusions.append({
            "ticker": ticker,
            "reason": (
                "avg_volume non-finite"
                if not np.isfinite(val)
                else f"avg_volume < {min_volume:,.0f}"
            ),
            "value": "nan" if not np.isfinite(val) else f"{val:,.0f}",
        })

    # Filter 2: Amihud percentile
    amihud_valid = amihud_t.dropna()
    if len(amihud_valid) > 0:
        threshold = np.percentile(
            amihud_valid.values, amihud_percentile_threshold
        )
        too_illiquid = amihud_valid[amihud_valid > threshold]
        for ticker in too_illiquid.index:
            if ticker not in [e["ticker"] for e in exclusions]:
                exclusions.append({
                    "ticker": ticker,
                    "reason": f"amihud > p{amihud_percentile_threshold:.0f}",
                    "value": f"{amihud_valid[ticker]:.2e}",
                })

    # Build active set
    excluded_tickers = {e["ticker"] for e in exclusions}
    active = [t for t in all_tickers if t not in excluded_tickers]

    exclusion_log = pd.DataFrame(exclusions) if exclusions else pd.DataFrame(
        columns=["ticker", "reason", "value"]
    )

    logger.info(
        "filter_universe(%s): %d active / %d excluded / %d total",
        t.date(), len(active), len(excluded_tickers), len(all_tickers),
    )
    if exclusions:
        logger.debug(
            "Excluded: %s", [e["ticker"] for e in exclusions]
        )

    return active, exclusion_log


def precompute_rolling_metrics(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pre-compute rolling Amihud and volume for efficiency in backtesting.

    Args:
        prices: Full clean prices.
        volumes: Full volume data.
        window: Rolling lookback.

    Returns:
        Tuple of (rolling_amihud, rolling_avg_vol) DataFrames.
    """
    rolling_amihud = compute_rolling_amihud(prices, volumes, window)
    rolling_avg_vol = compute_rolling_volume(volumes, window)
    logger.info(
        "Pre-computed rolling metrics: %d dates × %d tickers",
        len(rolling_amihud), rolling_amihud.shape[1],
    )
    return rolling_amihud, rolling_avg_vol


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prices = pd.read_parquet("data/clean_prices.parquet")
    volumes = pd.read_parquet("data/volumes.parquet")

    # Pre-compute
    r_amihud, r_vol = precompute_rolling_metrics(prices, volumes)

    # Filter at a specific date
    test_date = prices.index[-1]
    active, excluded = filter_universe_at_date(
        t=test_date,
        prices=prices,
        volumes=volumes,
        rolling_amihud=r_amihud,
        rolling_avg_vol=r_vol,
        amihud_percentile_threshold=90.0,
        min_volume=100_000,
    )

    print(f"\nDate: {test_date.date()}")
    print(f"Active tickers ({len(active)}): {active}")
    if not excluded.empty:
        print(f"\nExcluded:")
        print(excluded.to_string(index=False))
