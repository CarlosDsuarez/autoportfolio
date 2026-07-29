"""
regime_detector.py — T-33: Clasificador rolling de régimen de mercado (4 estados).

Clasifica el mercado en cada fecha t usando únicamente precios/retornos
disponibles hasta t (R-05, sin look-ahead). Ventana configurable
(default 63 ≈ 1 trimestre).

Estados:
  - low_vol_trending
  - low_vol_ranging
  - high_vol_trending   (crisis / momentum extremo)
  - high_vol_ranging    (choppy / crisis)

Features:
  - Volatilidad realizada anualizada → percentil histórico hasta t
  - Trend score: pendiente normalizada de media móvil
  - Breadth opcional: % del universo sobre su MA 200d

VIX (^VIX) NO está en el path default. Si se desea en el futuro,
pasarlo como serie opcional — no requerido por este módulo.

Restricciones cubiertas: R-05 (sin look-ahead), R-10 (determinismo).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("regime_detector")

SEED = 42
TRADING_DAYS = 252

# Named regime states
LOW_VOL_TRENDING = "low_vol_trending"
LOW_VOL_RANGING = "low_vol_ranging"
HIGH_VOL_TRENDING = "high_vol_trending"
HIGH_VOL_RANGING = "high_vol_ranging"

REGIME_STATES = (
    LOW_VOL_TRENDING,
    LOW_VOL_RANGING,
    HIGH_VOL_TRENDING,
    HIGH_VOL_RANGING,
)

_DEFAULT_WINDOW = 63
_DEFAULT_MA_TREND = 50
_DEFAULT_MA_BREADTH = 200
_DEFAULT_VOL_PCTILE_HIGH = 70.0   # above → high_vol
_DEFAULT_TREND_THRESHOLD = 0.5    # |trend_score| above → trending


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class RegimeReport:
    """Single-date regime classification."""

    t: pd.Timestamp
    state: str
    volatility_percentile: float
    trend_score: float
    breadth: Optional[float] = None


# ===================================================================
# Feature helpers (strictly causal)
# ===================================================================
def _realized_vol(
    returns: pd.Series,
    window: int = _DEFAULT_WINDOW,
) -> float:
    """Annualized realized volatility over the last `window` returns."""
    if len(returns) < max(5, window // 4):
        return float("nan")
    r = returns.iloc[-window:] if len(returns) >= window else returns
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def _vol_percentile(
    vol_history: pd.Series,
    current_vol: float,
) -> float:
    """Percentile of current_vol within historical vol series (inclusive)."""
    hist = vol_history.dropna()
    if len(hist) < 5 or not np.isfinite(current_vol):
        return 50.0
    return float((hist <= current_vol).mean() * 100.0)


def _trend_score(
    prices: pd.Series,
    ma_window: int = _DEFAULT_MA_TREND,
) -> float:
    """Normalized slope of MA as trend proxy in roughly [-2, 2].

    Positive = uptrend, negative = downtrend, near 0 = ranging.
    """
    if len(prices) < ma_window + 5:
        return 0.0
    ma = prices.rolling(ma_window).mean()
    # Slope over last ~21 days of the MA
    look = min(21, len(ma.dropna()))
    if look < 5:
        return 0.0
    y = ma.dropna().iloc[-look:].values
    x = np.arange(len(y), dtype=float)
    # Linear regression slope / price level
    slope = np.polyfit(x, y, 1)[0]
    level = float(y[-1]) if y[-1] != 0 else 1.0
    # Scale: daily fractional slope * 100 → roughly percent-per-day * 100
    raw = (slope / level) * 100.0
    # Soft-clip to [-2, 2]
    return float(np.clip(raw, -2.0, 2.0))


def _breadth(
    prices: pd.DataFrame,
    ma_window: int = _DEFAULT_MA_BREADTH,
) -> Optional[float]:
    """Fraction of assets above their MA. None if insufficient columns/history."""
    if prices.shape[1] < 2 or len(prices) < ma_window:
        return None
    ma = prices.rolling(ma_window).mean().iloc[-1]
    last = prices.iloc[-1]
    valid = ma.notna() & last.notna()
    if valid.sum() < 2:
        return None
    return float((last[valid] > ma[valid]).mean())


def _equal_weight_returns(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight portfolio simple returns from price panel."""
    rets = prices.pct_change().dropna(how="all")
    return rets.mean(axis=1)


# ===================================================================
# Classification
# ===================================================================
def classify_regime(
    prices: pd.DataFrame,
    t: pd.Timestamp,
    window: int = _DEFAULT_WINDOW,
    vol_pctile_high: float = _DEFAULT_VOL_PCTILE_HIGH,
    trend_threshold: float = _DEFAULT_TREND_THRESHOLD,
    ma_trend: int = _DEFAULT_MA_TREND,
    ma_breadth: int = _DEFAULT_MA_BREADTH,
) -> RegimeReport:
    """Classify market regime at date t using only data up to t (R-05).

    Args:
        prices: Full price DataFrame (date × ticker). May contain future
            rows; they are ignored (filtered to index <= t).
        t: Classification date (inclusive upper bound).
        window: Rolling window for realized vol / features.
        vol_pctile_high: Percentile threshold above which vol is 'high'.
        trend_threshold: |trend_score| above this → trending.
        ma_trend: MA window for trend slope.
        ma_breadth: MA window for breadth.

    Returns:
        RegimeReport with state and feature values.
    """
    # R-05: strictly use data up to and including t
    past = prices[prices.index <= t]
    if past.empty:
        return RegimeReport(
            t=pd.Timestamp(t),
            state=LOW_VOL_RANGING,
            volatility_percentile=50.0,
            trend_score=0.0,
        )

    ew_rets = _equal_weight_returns(past)
    # Build rolling vol history up to t for percentile
    if len(ew_rets) >= 10:
        vol_series = ew_rets.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
        current_vol = float(vol_series.iloc[-1]) if np.isfinite(vol_series.iloc[-1]) else _realized_vol(ew_rets, window)
        vol_pct = _vol_percentile(vol_series.iloc[:-1], current_vol) if len(vol_series) > 1 else 50.0
    else:
        current_vol = _realized_vol(ew_rets, window)
        vol_pct = 50.0

    # Equal-weight price index for trend
    ew_price = past.mean(axis=1)
    trend = _trend_score(ew_price, ma_window=ma_trend)
    breadth = _breadth(past, ma_window=ma_breadth)

    high_vol = vol_pct >= vol_pctile_high
    trending = abs(trend) >= trend_threshold

    if high_vol and trending:
        state = HIGH_VOL_TRENDING
    elif high_vol and not trending:
        state = HIGH_VOL_RANGING
    elif not high_vol and trending:
        state = LOW_VOL_TRENDING
    else:
        state = LOW_VOL_RANGING

    report = RegimeReport(
        t=pd.Timestamp(t),
        state=state,
        volatility_percentile=float(vol_pct),
        trend_score=float(trend),
        breadth=breadth,
    )
    logger.debug(
        "Regime at %s: %s (vol_pct=%.1f, trend=%.3f)",
        t.date() if hasattr(t, "date") else t,
        state, vol_pct, trend,
    )
    return report


def rolling_regime_series(
    prices: pd.DataFrame,
    window: int = _DEFAULT_WINDOW,
    min_history: int = 252,
    step: int = 1,
) -> pd.Series:
    """Compute regime state for each eligible date in prices.

    Args:
        prices: Full price panel.
        window: Feature window.
        min_history: Skip dates with fewer than this many rows of history.
        step: Evaluate every `step` trading days (1 = daily).

    Returns:
        Series of regime state strings indexed by date.
    """
    dates = prices.index.sort_values()
    records = {}
    for i, t in enumerate(dates):
        if (i + 1) < min_history:
            continue
        if step > 1 and (i - (min_history - 1)) % step != 0:
            continue
        report = classify_regime(prices, t, window=window)
        records[t] = report.state
    series = pd.Series(records, dtype=object)
    series.index.name = "date"
    logger.info(
        "rolling_regime_series: %d classifications from %s to %s",
        len(series),
        series.index[0].date() if len(series) else "N/A",
        series.index[-1].date() if len(series) else "N/A",
    )
    return series
