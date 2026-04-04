"""
expected_returns.py — T-08: Estimación de retornos esperados.

Tres métodos:
  (a) Media histórica de log-retornos.
  (b) CAPM shrunk hacia benchmark (shrinkage hacia retorno de equilibrio).
  (c) Black-Litterman básico (equilibrio implícito sin views).

Todos retornan vector mu anualizado, compatible con optimizadores.

Restricciones cubiertas: R-05 (usa solo datos de la ventana), R-09 (sin
look-ahead, sin overfitting — métodos estándar documentados).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("expected_returns")

SEED = 42
TRADING_DAYS = 252


# ===================================================================
# Method (a): Historical mean of log-returns
# ===================================================================
def mean_historical_return(
    prices: pd.DataFrame,
    frequency: int = TRADING_DAYS,
) -> pd.Series:
    """
    Annualized mean of daily log-returns.

    mu_i = mean(ln(P_t / P_{t-1})) * frequency

    Args:
        prices: Clean price DataFrame for the estimation window.
        frequency: Annualization factor (252 for daily).

    Returns:
        pd.Series of annualized expected returns per ticker.
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    mu = log_rets.mean() * frequency

    logger.debug(
        "mean_historical_return: %d tickers, mu range [%.4f, %.4f]",
        len(mu), mu.min(), mu.max(),
    )
    return mu


# ===================================================================
# Method (b): CAPM shrunk towards benchmark equilibrium
# ===================================================================
def capm_shrunk_return(
    prices: pd.DataFrame,
    benchmark_col: str = "SPY",
    risk_free_rate: float = 0.03,
    shrinkage_alpha: float = 0.5,
    frequency: int = TRADING_DAYS,
) -> pd.Series:
    """
    CAPM expected return shrunk towards historical mean.

    Steps:
      1. Compute beta_i = cov(r_i, r_m) / var(r_m)
      2. mu_capm_i = rf + beta_i * (E[r_m] - rf)
      3. mu_shrunk_i = alpha * mu_capm_i + (1 - alpha) * mu_hist_i

    The shrinkage blends CAPM equilibrium (stable, low variance) with
    historical means (high variance, potentially more accurate short-term).

    Args:
        prices: Clean prices. Must include benchmark_col.
        benchmark_col: Ticker used as market proxy (default SPY).
        risk_free_rate: Annualized risk-free rate.
        shrinkage_alpha: Weight on CAPM estimate (0=pure historical,
            1=pure CAPM). Default 0.5.
        frequency: Annualization factor.

    Returns:
        pd.Series of shrunk annualized expected returns.

    Raises:
        ValueError: If benchmark not in prices.
    """
    if benchmark_col not in prices.columns:
        raise ValueError(
            f"Benchmark '{benchmark_col}' not found in prices. "
            f"Available: {list(prices.columns[:5])}..."
        )

    log_rets = np.log(prices / prices.shift(1)).dropna()

    # Market return
    r_m = log_rets[benchmark_col]
    var_m = r_m.var() * frequency
    mu_m = r_m.mean() * frequency

    # Betas
    cov_with_market = log_rets.apply(lambda col: col.cov(r_m)) * frequency
    betas = cov_with_market / var_m

    # CAPM expected returns
    market_premium = mu_m - risk_free_rate
    mu_capm = risk_free_rate + betas * market_premium

    # Historical mean
    mu_hist = mean_historical_return(prices, frequency)

    # Shrinkage blend
    mu_shrunk = shrinkage_alpha * mu_capm + (1 - shrinkage_alpha) * mu_hist

    logger.debug(
        "capm_shrunk_return: alpha=%.2f, market_premium=%.4f, "
        "beta range [%.2f, %.2f]",
        shrinkage_alpha, market_premium, betas.min(), betas.max(),
    )
    return mu_shrunk


# ===================================================================
# Method (c): Black-Litterman implied equilibrium (no views)
# ===================================================================
def black_litterman_equilibrium(
    prices: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    market_caps: Optional[pd.Series] = None,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.03,
    frequency: int = TRADING_DAYS,
) -> pd.Series:
    """
    Black-Litterman implied equilibrium returns (no investor views).

    Pi = delta * Sigma * w_mkt

    where:
      - delta = risk aversion coefficient
      - Sigma = covariance matrix (annualized)
      - w_mkt = market-cap weights (or equal-weight if not provided)

    Without views, this returns the "equilibrium prior" — the expected
    returns implied by current market weights and the covariance.

    Args:
        prices: Clean prices (used for ticker alignment).
        cov_matrix: Annualized covariance matrix (from T-09).
        market_caps: Market cap per ticker for weighting. If None,
            uses equal-weight proxy.
        risk_aversion: Risk aversion parameter (default 2.5,
            typical range 1-5).
        risk_free_rate: Annualized risk-free rate.
        frequency: Annualization factor.

    Returns:
        pd.Series of implied equilibrium returns.
    """
    tickers = prices.columns.tolist()
    n = len(tickers)

    # Market weights
    if market_caps is not None:
        # Normalize to sum to 1
        caps = market_caps.reindex(tickers).fillna(0)
        w_mkt = caps / caps.sum()
    else:
        # Equal-weight proxy
        w_mkt = pd.Series(1.0 / n, index=tickers)
        logger.debug(
            "No market caps provided — using equal-weight proxy."
        )

    # Align covariance matrix
    sigma = cov_matrix.reindex(index=tickers, columns=tickers)

    # Implied equilibrium returns: Pi = delta * Sigma * w_mkt
    pi = risk_aversion * sigma.values @ w_mkt.values
    mu_eq = pd.Series(pi, index=tickers) + risk_free_rate

    logger.debug(
        "black_litterman_equilibrium: delta=%.1f, "
        "mu_eq range [%.4f, %.4f]",
        risk_aversion, mu_eq.min(), mu_eq.max(),
    )
    return mu_eq


# ===================================================================
# Unified interface
# ===================================================================
def estimate_expected_returns(
    prices: pd.DataFrame,
    method: str = "historical",
    cov_matrix: Optional[pd.DataFrame] = None,
    benchmark_col: str = "SPY",
    risk_free_rate: float = 0.03,
    shrinkage_alpha: float = 0.5,
    risk_aversion: float = 2.5,
    market_caps: Optional[pd.Series] = None,
    frequency: int = TRADING_DAYS,
) -> pd.Series:
    """
    Unified expected returns estimator.

    Args:
        prices: Clean prices for estimation window.
        method: One of 'historical', 'capm_shrunk', 'black_litterman'.
        cov_matrix: Required for 'black_litterman' method.
        benchmark_col: Benchmark ticker for CAPM.
        risk_free_rate: Annualized risk-free rate.
        shrinkage_alpha: Shrinkage weight for CAPM method.
        risk_aversion: Risk aversion for Black-Litterman.
        market_caps: Market caps for Black-Litterman weighting.
        frequency: Annualization factor.

    Returns:
        pd.Series of annualized expected returns.

    Raises:
        ValueError: If method is unknown or required params missing.
    """
    method = method.lower().strip()

    if method == "historical":
        mu = mean_historical_return(prices, frequency)

    elif method == "capm_shrunk":
        mu = capm_shrunk_return(
            prices, benchmark_col, risk_free_rate,
            shrinkage_alpha, frequency,
        )

    elif method == "black_litterman":
        if cov_matrix is None:
            raise ValueError(
                "black_litterman method requires cov_matrix argument."
            )
        mu = black_litterman_equilibrium(
            prices, cov_matrix, market_caps,
            risk_aversion, risk_free_rate, frequency,
        )

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: historical, capm_shrunk, black_litterman"
        )

    logger.info(
        "Expected returns (%s): %d tickers, "
        "mean=%.4f, std=%.4f",
        method, len(mu), mu.mean(), mu.std(),
    )
    return mu


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prices = pd.read_parquet("data/clean_prices.parquet")
    # Use last 252 days as estimation window
    window_prices = prices.iloc[-252:]

    print("=" * 60)
    print("EXPECTED RETURNS — 3 METHODS COMPARISON")
    print("=" * 60)

    methods = ["historical", "capm_shrunk", "black_litterman"]
    results = {}

    for m in methods:
        if m == "black_litterman":
            # Need covariance matrix — use sample for demo
            log_rets = np.log(window_prices / window_prices.shift(1)).dropna()
            cov = log_rets.cov() * TRADING_DAYS
            mu = estimate_expected_returns(
                window_prices, method=m, cov_matrix=cov
            )
        else:
            mu = estimate_expected_returns(window_prices, method=m)
        results[m] = mu

    comparison = pd.DataFrame(results).round(4)
    print(comparison.to_string())
    print(f"\nCorrelation between methods:")
    print(comparison.corr().round(3).to_string())
