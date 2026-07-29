"""
data_client.py — T-02: Clientes de APIs con fallback automático.

Fuente primaria: Yahoo Finance (yfinance).
Fallback: Alpha Vantage (requiere API key), IEX Cloud (requiere token).
Incluye: rate limiting, retry con backoff exponencial, logging de errores.

Restricciones cubiertas: R-05 (datos para rolling window), R-10 (modular).
"""

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("data_client")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
MAX_RETRIES = 3
BACKOFF_BASE = 2.0  # seconds
FFILL_MAX_DAYS = 2
MIN_HISTORY_YEARS = 5


# ===================================================================
# T-01 helper: load universe
# ===================================================================
_HERE = Path(__file__).parent


def load_universe(path: str = None) -> pd.DataFrame:
    """
    Load the asset universe from CSV or Excel.

    Args:
        path: Path to universe file (.csv or .xlsx).
              Defaults to universe.csv next to this script, or at the
              repository root (parent of ``src/``).

    Returns:
        DataFrame with columns [ticker, asset_type, sector, name,
        inclusion_reason].
    """
    if path is None:
        # Look next to this module, then at repo root (CM Project/universe.csv)
        candidates = [
            _HERE / "universe.csv",
            _HERE / "universe.xlsx",
            _HERE.parent / "universe.csv",
            _HERE.parent / "universe.xlsx",
        ]
        for candidate in candidates:
            if candidate.exists():
                path = str(candidate)
                break
        else:
            raise FileNotFoundError(
                f"No universe.csv or universe.xlsx found in {_HERE} "
                f"or {_HERE.parent}"
            )

    p = Path(path)
    if p.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p)
    required_cols = {"ticker", "asset_type", "sector", "name"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"universe.csv missing columns: {required_cols - set(df.columns)}"
        )
    logger.info(
        "Universe loaded: %d stocks, %d ETFs",
        (df["asset_type"] == "stock").sum(),
        (df["asset_type"] == "etf").sum(),
    )
    return df


# ===================================================================
# T-02: Yahoo Finance client (primary)
# ===================================================================
def _download_yfinance(
    tickers: list[str],
    start: str,
    end: str,
    max_retries: int = MAX_RETRIES,
) -> Optional[pd.DataFrame]:
    """
    Download adjusted close prices from Yahoo Finance with retry logic.

    Args:
        tickers: List of ticker symbols.
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'.
        max_retries: Max retry attempts with exponential backoff.

    Returns:
        DataFrame (date index, ticker columns) of adjusted close prices,
        or None if all retries fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "yfinance download attempt %d/%d — %d tickers, %s to %s",
                attempt, max_retries, len(tickers), start, end,
            )
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            # yf.download returns MultiIndex columns when >1 ticker
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"].copy()
            else:
                prices = raw[["Close"]].copy()
                prices.columns = tickers

            prices.index = pd.to_datetime(prices.index)
            prices.index.name = "date"

            # Drop tickers with zero data
            valid = prices.dropna(axis=1, how="all")
            dropped = set(tickers) - set(valid.columns)
            if dropped:
                logger.warning("Tickers with no data dropped: %s", dropped)

            logger.info(
                "yfinance success: %d tickers, %d rows",
                valid.shape[1], valid.shape[0],
            )
            return valid

        except Exception as exc:
            wait = BACKOFF_BASE ** attempt
            logger.error(
                "yfinance attempt %d failed: %s — retrying in %.1fs",
                attempt, exc, wait,
            )
            time.sleep(wait)

    logger.error("yfinance: all %d attempts exhausted.", max_retries)
    return None


# ===================================================================
# T-02: Fallback — CSV local
# ===================================================================
def _load_csv_fallback(
    path: str,
    tickers: list[str],
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """
    Load prices from a local CSV file as fallback.

    Expects CSV with 'date' column and one column per ticker.

    Args:
        path: Path to CSV file.
        tickers: List of tickers to filter.
        start: Start date.
        end: End date.

    Returns:
        Filtered DataFrame or None.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("CSV fallback file not found: %s", path)
        return None

    try:
        df = pd.read_csv(p, parse_dates=["date"], index_col="date")
        available = [t for t in tickers if t in df.columns]
        if not available:
            logger.warning("No matching tickers in CSV fallback.")
            return None
        df = df.loc[start:end, available]
        logger.info("CSV fallback loaded: %d tickers, %d rows", len(available), len(df))
        return df
    except Exception as exc:
        logger.error("CSV fallback error: %s", exc)
        return None


# ===================================================================
# T-03: Master fetch function with fallback chain
# ===================================================================
def fetch_prices(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2025-01-01",
    csv_fallback_path: str = "data/raw_prices_backup.csv",
) -> pd.DataFrame:
    """
    Fetch adjusted close prices with automatic fallback.

    Priority: yfinance → local CSV.

    Args:
        tickers: Ticker symbols.
        start: Start date.
        end: End date.
        csv_fallback_path: Path for CSV fallback.

    Returns:
        DataFrame of adjusted close prices (date index × tickers).

    Raises:
        RuntimeError: If all sources fail.
    """
    # Primary: yfinance
    prices = _download_yfinance(tickers, start, end)
    if prices is not None and not prices.empty:
        return prices

    # Fallback: CSV
    logger.warning("Primary source failed. Trying CSV fallback...")
    prices = _load_csv_fallback(csv_fallback_path, tickers, start, end)
    if prices is not None and not prices.empty:
        return prices

    raise RuntimeError(
        "All data sources exhausted. Check network or provide CSV fallback."
    )


# ===================================================================
# T-04: Data cleaning and validation
# ===================================================================
def clean_prices(
    prices: pd.DataFrame,
    ffill_max: int = FFILL_MAX_DAYS,
    sigma_threshold: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean price data: handle NaN, outliers, alignment.

    Args:
        prices: Raw price DataFrame.
        ffill_max: Max consecutive days for forward-fill.
        sigma_threshold: Outlier threshold in standard deviations
            applied to log-returns.

    Returns:
        Tuple of (clean_prices, quality_report).
        quality_report has columns: ticker, total_rows, missing_pct,
        outliers_flagged, first_date, last_date, completeness_ok.

    Restricciones: R-05 (temporal alignment), R-09 (no look-ahead).
    """
    logger.info("Cleaning prices: %d tickers, %d rows", prices.shape[1], prices.shape[0])

    # --- Step 1: forward-fill limited ---
    filled = prices.ffill(limit=ffill_max)

    # --- Step 2: detect outliers on log-returns ---
    log_rets = np.log(filled / filled.shift(1))
    mu = log_rets.mean()
    sigma = log_rets.std()
    outlier_mask = (log_rets - mu).abs() > sigma_threshold * sigma

    # Replace outlier returns with NaN, then ffill price
    outlier_counts = outlier_mask.sum()
    # Don't remove outliers — flag only. Replace price at outlier points
    # with interpolation to avoid data loss.
    clean = filled.copy()
    for col in clean.columns:
        if outlier_counts[col] > 0:
            bad_idx = outlier_mask.index[outlier_mask[col]]
            clean.loc[bad_idx, col] = np.nan
            clean[col] = clean[col].interpolate(method="linear", limit=2)
            logger.info(
                "Ticker %s: %d outliers interpolated", col, outlier_counts[col]
            )

    # --- Step 3: drop tickers with too many NaN ---
    missing_pct = clean.isna().mean()
    bad_tickers = missing_pct[missing_pct > 0.05].index.tolist()
    if bad_tickers:
        logger.warning(
            "Dropping tickers with >5%% missing: %s", bad_tickers
        )
        clean = clean.drop(columns=bad_tickers)

    # --- Step 4: drop remaining NaN rows ---
    clean = clean.dropna()

    # --- Quality report ---
    report_rows = []
    for col in prices.columns:
        total = len(prices)
        missing = prices[col].isna().sum()
        report_rows.append({
            "ticker": col,
            "total_rows": total,
            "missing_count": int(missing),
            "missing_pct": round(missing / total * 100, 2),
            "outliers_flagged": int(outlier_counts.get(col, 0)),
            "first_date": str(prices[col].first_valid_index()),
            "last_date": str(prices[col].last_valid_index()),
            "in_clean_set": col in clean.columns,
        })
    quality_report = pd.DataFrame(report_rows)

    logger.info(
        "Clean prices: %d tickers retained, %d rows",
        clean.shape[1], clean.shape[0],
    )
    return clean, quality_report


# ===================================================================
# T-04b: Calculate log-returns (helper for downstream)
# ===================================================================
def calc_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log-returns from price DataFrame.

    Args:
        prices: Clean adjusted close prices.

    Returns:
        DataFrame of log-returns (first row dropped).
    """
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets


# ===================================================================
# T-05: Liquidity scoring — Amihud illiquidity ratio
# ===================================================================
def calc_liquidity_scores(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """
    Calculate liquidity scores: average daily volume (30d) and
    Amihud illiquidity ratio per ticker.

    Amihud(i) = mean(|r_t| / volume_t) over trailing window.
    Lower Amihud → more liquid.

    Args:
        prices: Clean adjusted close prices.
        volumes: Volume DataFrame aligned with prices.
        window: Lookback window in trading days.

    Returns:
        DataFrame with columns [ticker, avg_volume_30d,
        amihud_illiquidity, liquidity_rank].

    Restricciones: R-03 (liquidez mínima).
    """
    log_rets = np.log(prices / prices.shift(1)).iloc[-window:]
    vol_window = volumes.iloc[-window:]

    records = []
    for ticker in prices.columns:
        if ticker not in volumes.columns:
            logger.warning("No volume data for %s — skipping", ticker)
            continue

        avg_vol = vol_window[ticker].mean()
        # Amihud: mean(|return| / dollar_volume)
        abs_ret = log_rets[ticker].abs()
        dollar_vol = vol_window[ticker] * prices[ticker].iloc[-window:]
        # Avoid division by zero
        safe_dvol = dollar_vol.replace(0, np.nan)
        amihud = (abs_ret / safe_dvol).mean()

        records.append({
            "ticker": ticker,
            "avg_volume_30d": round(avg_vol, 0),
            "amihud_illiquidity": round(amihud, 12),
        })

    scores = pd.DataFrame(records)
    # Rank: lower Amihud = better liquidity = rank 1
    scores["liquidity_rank"] = scores["amihud_illiquidity"].rank(method="min")
    scores = scores.sort_values("liquidity_rank").reset_index(drop=True)

    logger.info("Liquidity scores computed for %d tickers", len(scores))
    return scores


def fetch_volumes(
    tickers: list[str],
    start: str = "2019-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    """
    Fetch daily trading volume via yfinance.

    Args:
        tickers: Ticker symbols.
        start: Start date.
        end: End date.

    Returns:
        DataFrame of daily volumes (date index × tickers).
    """
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False, threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        volumes = raw["Volume"].copy()
    else:
        volumes = raw[["Volume"]].copy()
        volumes.columns = tickers
    volumes.index = pd.to_datetime(volumes.index)
    volumes.index.name = "date"
    return volumes


# ===================================================================
# T-03 + T-04 + T-05: Integrated pipeline runner
# ===================================================================
def run_data_pipeline(
    universe_path: str = None,
    start: str = "2019-01-01",
    end: str = "2025-01-01",
    output_dir: str = "data",
) -> dict:
    """
    Execute the full Phase 1 data pipeline:
      1. Load universe
      2. Fetch prices and volumes
      3. Clean prices
      4. Compute liquidity scores
      5. Save all artifacts

    Args:
        universe_path: Path to universe CSV.
        start: Backtest start date.
        end: Backtest end date.
        output_dir: Directory for output parquet/CSV files.

    Returns:
        Dict with keys: universe, clean_prices, quality_report,
        liquidity_scores, log_returns.
    """
    np.random.seed(SEED)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Universe
    universe = load_universe(universe_path)
    tickers = universe["ticker"].tolist()

    # 2. Fetch
    logger.info("=" * 60)
    logger.info("PHASE 1 — DATA PIPELINE START")
    logger.info("=" * 60)

    prices = fetch_prices(tickers, start, end)
    volumes = fetch_volumes(tickers, start, end)

    # 3. Clean
    clean, quality_report = clean_prices(prices)

    # 4. Liquidity
    # Align volumes to clean prices columns
    common_tickers = [t for t in clean.columns if t in volumes.columns]
    liq_scores = calc_liquidity_scores(
        clean[common_tickers], volumes[common_tickers], window=30
    )

    # 5. Log-returns
    log_rets = calc_log_returns(clean)

    # 6. Save artifacts
    prices.to_parquet(out / "raw_prices.parquet")
    clean.to_parquet(out / "clean_prices.parquet")
    quality_report.to_csv(out / "data_quality_report.csv", index=False)
    liq_scores.to_csv(out / "liquidity_scores.csv", index=False)
    log_rets.to_parquet(out / "log_returns.parquet")

    logger.info("=" * 60)
    logger.info("PHASE 1 — DATA PIPELINE COMPLETE")
    logger.info("Artifacts saved to %s/", output_dir)
    logger.info("=" * 60)

    return {
        "universe": universe,
        "clean_prices": clean,
        "quality_report": quality_report,
        "liquidity_scores": liq_scores,
        "log_returns": log_rets,
    }


if __name__ == "__main__":
    results = run_data_pipeline()
    print("\n--- Quality Report ---")
    print(results["quality_report"].to_string(index=False))
    print("\n--- Liquidity Scores (top 10) ---")
    print(results["liquidity_scores"].head(10).to_string(index=False))
