"""
covariance_estimators.py — T-09: Estimación de matrices de covarianza.

Tres métodos:
  (a) Muestral (sample covariance).
  (b) Ledoit-Wolf shrinkage (hacia diagonal constante).
  (c) PCA con k=5 factores (factor covariance model).

Todas retornan Sigma anualizada, positiva definida, alineada con tickers.
Alerta si número de condición > 1e10.

Restricciones cubiertas: R-09 (estabilidad de estimadores, sin overfitting).
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numpy.linalg import eigh, cond

logger = logging.getLogger("covariance_estimators")

SEED = 42
TRADING_DAYS = 252
CONDITION_NUMBER_THRESHOLD = 1e10


# ===================================================================
# Validation helpers
# ===================================================================
def _validate_psd(
    sigma: pd.DataFrame,
    method: str,
    fix: bool = True,
) -> pd.DataFrame:
    """
    Validate positive semi-definiteness. Fix if needed via eigenvalue
    clipping.

    Args:
        sigma: Covariance matrix DataFrame.
        method: Estimator name for logging.
        fix: If True, clip negative eigenvalues to small positive.

    Returns:
        Validated (and possibly fixed) covariance matrix.
    """
    eigenvalues = eigh(sigma.values)[0]
    min_eig = eigenvalues.min()

    if min_eig < -1e-10:
        logger.warning(
            "%s: NOT positive semi-definite (min eigenvalue=%.2e).",
            method, min_eig,
        )
        if fix:
            # Clip negative eigenvalues
            vals, vecs = eigh(sigma.values)
            vals = np.maximum(vals, 1e-10)
            fixed = vecs @ np.diag(vals) @ vecs.T
            # Symmetrize
            fixed = (fixed + fixed.T) / 2
            sigma = pd.DataFrame(
                fixed, index=sigma.index, columns=sigma.columns
            )
            logger.info("%s: Fixed — eigenvalues clipped to 1e-10.", method)

    # Condition number check
    cn = cond(sigma.values)
    if cn > CONDITION_NUMBER_THRESHOLD:
        logger.warning(
            "%s: HIGH condition number = %.2e (threshold: %.2e). "
            "Matrix may be ill-conditioned.",
            method, cn, CONDITION_NUMBER_THRESHOLD,
        )

    return sigma


def _annualize_cov(
    cov_daily: pd.DataFrame,
    frequency: int = TRADING_DAYS,
) -> pd.DataFrame:
    """Scale daily covariance to annualized."""
    return cov_daily * frequency


# ===================================================================
# Method (a): Sample covariance
# ===================================================================
def sample_covariance(
    prices: pd.DataFrame,
    frequency: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Standard sample covariance matrix of log-returns, annualized.

    Sigma = (1/(T-1)) * sum((r_t - mu)(r_t - mu)^T) * frequency

    Simple but noisy when n_assets >> n_observations.

    Args:
        prices: Clean price DataFrame for estimation window.
        frequency: Annualization factor.

    Returns:
        Annualized sample covariance matrix (n_assets × n_assets).
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    cov_daily = log_rets.cov()
    sigma = _annualize_cov(cov_daily, frequency)
    sigma = _validate_psd(sigma, "sample_covariance")

    logger.debug(
        "sample_covariance: %dx%d, condition=%.2e",
        sigma.shape[0], sigma.shape[1], cond(sigma.values),
    )
    return sigma


# ===================================================================
# Method (b): Ledoit-Wolf shrinkage
# ===================================================================
def ledoit_wolf_covariance(
    prices: pd.DataFrame,
    frequency: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance estimator.

    Sigma_LW = alpha * F + (1 - alpha) * S

    where:
      - S = sample covariance
      - F = structured target (constant correlation model)
      - alpha = optimal shrinkage intensity (data-driven)

    Reference: Ledoit & Wolf (2004), JMVA.

    Args:
        prices: Clean price DataFrame.
        frequency: Annualization factor.

    Returns:
        Annualized Ledoit-Wolf shrinkage covariance matrix.
    """
    from sklearn.covariance import LedoitWolf

    log_rets = np.log(prices / prices.shift(1)).dropna()
    tickers = prices.columns.tolist()

    # Fit Ledoit-Wolf on daily returns
    lw = LedoitWolf()
    lw.fit(log_rets.values)

    cov_daily = pd.DataFrame(
        lw.covariance_, index=tickers, columns=tickers
    )
    sigma = _annualize_cov(cov_daily, frequency)
    sigma = _validate_psd(sigma, "ledoit_wolf")

    logger.debug(
        "ledoit_wolf: shrinkage_intensity=%.4f, condition=%.2e",
        lw.shrinkage_, cond(sigma.values),
    )
    return sigma


# ===================================================================
# Method (c): PCA factor covariance model
# ===================================================================
def pca_factor_covariance(
    prices: pd.DataFrame,
    n_factors: int = 5,
    frequency: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    PCA-based factor covariance model.

    Decompose returns into k principal component factors:
      R = B * F + epsilon

    Covariance approximation:
      Sigma = B * Sigma_F * B^T + D

    where:
      - B = factor loadings (n_assets × k)
      - Sigma_F = factor covariance (k × k, diagonal)
      - D = diagonal residual variance matrix

    This produces a lower-rank + diagonal covariance that is more
    stable than the full sample matrix.

    Reference: Deng et al. (2023), arXiv:2303.12751.

    Args:
        prices: Clean price DataFrame.
        n_factors: Number of principal components (default 5).
        frequency: Annualization factor.

    Returns:
        Annualized PCA factor covariance matrix.
    """
    log_rets = np.log(prices / prices.shift(1)).dropna()
    tickers = prices.columns.tolist()
    X = log_rets.values  # (T, n)
    T, n = X.shape

    if n_factors >= n:
        logger.warning(
            "n_factors (%d) >= n_assets (%d). Falling back to sample.",
            n_factors, n,
        )
        return sample_covariance(prices, frequency)

    # Center returns
    X_centered = X - X.mean(axis=0)

    # SVD decomposition
    U, S_vals, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Factor loadings: top k components
    # B = V_k * diag(s_k) / sqrt(T-1)
    B = Vt[:n_factors, :].T  # (n, k)
    factor_variance = (S_vals[:n_factors] ** 2) / (T - 1)  # eigenvalues

    # Factor covariance (diagonal)
    Sigma_F = np.diag(factor_variance)  # (k, k)

    # Factor-model covariance
    cov_factor = B @ Sigma_F @ B.T  # (n, n)

    # Residual variance (diagonal)
    cov_sample = (X_centered.T @ X_centered) / (T - 1)
    residual = np.diag(cov_sample) - np.diag(cov_factor)
    residual = np.maximum(residual, 1e-10)  # ensure positive
    D = np.diag(residual)

    # Final covariance: factor + residual
    cov_daily_arr = cov_factor + D

    cov_daily = pd.DataFrame(cov_daily_arr, index=tickers, columns=tickers)
    sigma = _annualize_cov(cov_daily, frequency)
    sigma = _validate_psd(sigma, f"pca_factor(k={n_factors})")

    # Log variance explained
    total_var = (S_vals ** 2).sum()
    explained = (S_vals[:n_factors] ** 2).sum() / total_var
    logger.debug(
        "pca_factor(k=%d): variance_explained=%.2f%%, condition=%.2e",
        n_factors, explained * 100, cond(sigma.values),
    )
    return sigma


# ===================================================================
# Unified interface
# ===================================================================
def estimate_covariance(
    prices: pd.DataFrame,
    method: str = "ledoit_wolf",
    n_factors: int = 5,
    frequency: int = TRADING_DAYS,
) -> pd.DataFrame:
    """
    Unified covariance estimator.

    Args:
        prices: Clean prices for estimation window.
        method: One of 'sample', 'ledoit_wolf', 'pca'.
        n_factors: Number of PCA factors (only for 'pca' method).
        frequency: Annualization factor.

    Returns:
        Annualized covariance matrix (pd.DataFrame).

    Raises:
        ValueError: If method is unknown.
    """
    method = method.lower().strip()

    if method == "sample":
        sigma = sample_covariance(prices, frequency)
    elif method == "ledoit_wolf":
        sigma = ledoit_wolf_covariance(prices, frequency)
    elif method == "pca":
        sigma = pca_factor_covariance(prices, n_factors, frequency)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: sample, ledoit_wolf, pca"
        )

    logger.info(
        "Covariance estimated (%s): %dx%d, condition=%.2e",
        method, sigma.shape[0], sigma.shape[1], cond(sigma.values),
    )
    return sigma


# ===================================================================
# Stability diagnostics (for T-10 notebook)
# ===================================================================
def eigenvalue_diagnostics(
    sigma: pd.DataFrame,
    method: str = "",
) -> dict:
    """
    Compute eigenvalue-based diagnostics for a covariance matrix.

    Args:
        sigma: Covariance matrix.
        method: Label for the estimator.

    Returns:
        Dict with eigenvalues, condition_number, top_k_explained, etc.
    """
    eigenvalues = np.sort(eigh(sigma.values)[0])[::-1]
    total = eigenvalues.sum()

    return {
        "method": method,
        "n_assets": sigma.shape[0],
        "condition_number": eigenvalues[0] / max(eigenvalues[-1], 1e-15),
        "min_eigenvalue": eigenvalues[-1],
        "max_eigenvalue": eigenvalues[0],
        "top1_explained": eigenvalues[0] / total,
        "top5_explained": eigenvalues[:5].sum() / total,
        "top10_explained": eigenvalues[:10].sum() / total,
        "eigenvalues": eigenvalues,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    prices = pd.read_parquet("data/clean_prices.parquet")
    window_prices = prices.iloc[-252:]

    print("=" * 60)
    print("COVARIANCE ESTIMATORS — COMPARISON")
    print("=" * 60)

    for m in ["sample", "ledoit_wolf", "pca"]:
        sigma = estimate_covariance(window_prices, method=m)
        diag = eigenvalue_diagnostics(sigma, m)
        print(f"\n{m.upper()}")
        print(f"  Condition number: {diag['condition_number']:.2e}")
        print(f"  Min eigenvalue:   {diag['min_eigenvalue']:.6f}")
        print(f"  Top-1 explained:  {diag['top1_explained']:.2%}")
        print(f"  Top-5 explained:  {diag['top5_explained']:.2%}")
        print(f"  Top-10 explained: {diag['top10_explained']:.2%}")
