"""
rolling_engine.py — T-07: Motor de ventanas móviles.

Genera slices temporales para estimación de parámetros y backtesting.
Validación estricta: nunca incluye datos posteriores a t.

Restricciones cubiertas: R-05 (rolling window sin look-ahead).
"""

import logging
from dataclasses import dataclass
from typing import Iterator, Optional

import pandas as pd

logger = logging.getLogger("rolling_engine")

SEED = 42


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class WindowSlice:
    """Container for a single rolling window slice."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp  # Inclusive: last date in training set
    test_start: pd.Timestamp
    test_end: pd.Timestamp  # Inclusive: last date in test set
    train_data: pd.DataFrame  # Prices in [train_start, train_end]
    test_data: pd.DataFrame  # Prices in [test_start, test_end]
    window_id: int


# ===================================================================
# Core: get_window
# ===================================================================
def get_window(
    prices: pd.DataFrame,
    t: pd.Timestamp,
    window: int = 252,
) -> pd.DataFrame:
    """
    Return price data strictly before or at date t, last `window` rows.

    CAUTION: This is the single point of look-ahead control.
    No data after t is ever returned.

    Args:
        prices: Full clean price DataFrame (date-indexed).
        t: Reference date (inclusive upper bound).
        window: Number of trading days to include.

    Returns:
        DataFrame slice of shape (<=window, n_tickers).

    Raises:
        ValueError: If fewer than window//2 rows available.
    """
    # R-05: strict temporal filter — no data after t
    mask = prices.index <= t
    available = prices.loc[mask]

    if len(available) < window // 2:
        raise ValueError(
            f"Insufficient data at {t}: {len(available)} rows "
            f"(need >= {window // 2})"
        )

    # Take last `window` rows
    sliced = available.iloc[-window:]

    logger.debug(
        "get_window(t=%s, window=%d): %d rows, [%s → %s]",
        t.date(), window, len(sliced),
        sliced.index[0].date(), sliced.index[-1].date(),
    )
    return sliced


# ===================================================================
# Rolling window generator
# ===================================================================
def rolling_windows(
    prices: pd.DataFrame,
    window: int = 252,
    step: int = 21,
    test_periods: int = 126,
    min_train_rows: Optional[int] = None,
) -> Iterator[WindowSlice]:
    """
    Generate rolling train/test window slices.

    Args:
        prices: Clean price DataFrame (date-indexed, sorted).
        window: Training window size in trading days.
        step: Step size between consecutive windows (21 ≈ monthly).
        test_periods: Out-of-sample test window in trading days.
        min_train_rows: Minimum rows required for training
            (defaults to window).

    Yields:
        WindowSlice objects with train/test splits.

    Restricciones: R-05 (no look-ahead), R-06 (OOS backtesting).
    """
    if min_train_rows is None:
        min_train_rows = window

    dates = prices.index.sort_values()
    n = len(dates)
    window_id = 0

    # First valid train_end: need `window` rows before it
    start_idx = window - 1

    for i in range(start_idx, n - 1, step):
        train_end_date = dates[i]

        # Training slice: last `window` rows up to and including train_end
        train = prices.loc[prices.index <= train_end_date].iloc[-window:]

        if len(train) < min_train_rows:
            continue

        # Test slice: next `test_periods` rows after train_end
        future_mask = prices.index > train_end_date
        test_pool = prices.loc[future_mask]

        if test_pool.empty:
            break

        test = test_pool.iloc[:test_periods]

        # R-05 VALIDATION: assert no temporal overlap
        assert train.index.max() < test.index.min(), (
            f"Look-ahead detected! Train ends {train.index.max()}, "
            f"test starts {test.index.min()}"
        )

        window_id += 1
        yield WindowSlice(
            train_start=train.index[0],
            train_end=train.index[-1],
            test_start=test.index[0],
            test_end=test.index[-1],
            train_data=train,
            test_data=test,
            window_id=window_id,
        )

        logger.info(
            "Window %03d: train [%s → %s] (%d rows) | "
            "test [%s → %s] (%d rows)",
            window_id,
            train.index[0].date(), train.index[-1].date(), len(train),
            test.index[0].date(), test.index[-1].date(), len(test),
        )


# ===================================================================
# Helper: get rebalance dates
# ===================================================================
def get_rebalance_dates(
    prices: pd.DataFrame,
    freq: str = "M",
    warmup: int = 252,
) -> list[pd.Timestamp]:
    """
    Return end-of-period rebalance dates after warmup period.

    Args:
        prices: Price DataFrame.
        freq: Rebalance frequency ('M'=monthly, 'Q'=quarterly).
        warmup: Minimum rows before first rebalance.

    Returns:
        List of rebalance dates.
    """
    dates = prices.index.sort_values()

    # Group by period, take last date of each group
    if freq == "M":
        groups = dates.to_period("M")
    elif freq == "Q":
        groups = dates.to_period("Q")
    else:
        raise ValueError(f"Unsupported freq: {freq}. Use 'M' or 'Q'.")

    # Last trading day of each period
    rebal_dates = (
        pd.Series(dates, index=dates)
        .groupby(groups)
        .last()
        .values
    )

    # Filter: only dates with enough warmup
    rebal_dates = [
        pd.Timestamp(d) for d in rebal_dates
        if (dates <= d).sum() >= warmup
    ]

    logger.info(
        "Rebalance dates (%s): %d dates from %s to %s",
        freq, len(rebal_dates),
        rebal_dates[0].date() if rebal_dates else "N/A",
        rebal_dates[-1].date() if rebal_dates else "N/A",
    )
    return rebal_dates


if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.INFO)

    # Quick demo with synthetic data
    prices = pd.read_parquet("data/clean_prices.parquet")
    print(f"Loaded: {prices.shape}")

    # Generate windows
    windows = list(rolling_windows(
        prices, window=252, step=21, test_periods=126
    ))
    print(f"\nTotal windows: {len(windows)}")
    if windows:
        w = windows[0]
        print(f"First window: train [{w.train_start.date()} → "
              f"{w.train_end.date()}], test [{w.test_start.date()} → "
              f"{w.test_end.date()}]")
        w = windows[-1]
        print(f"Last window:  train [{w.train_start.date()} → "
              f"{w.train_end.date()}], test [{w.test_start.date()} → "
              f"{w.test_end.date()}]")

    # Rebalance dates
    rebal = get_rebalance_dates(prices, freq="M", warmup=252)
    print(f"\nMonthly rebalance dates: {len(rebal)}")
