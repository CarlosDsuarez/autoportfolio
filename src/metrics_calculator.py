"""
metrics_calculator.py — T-26: Métricas de rendimiento del portafolio.

Calcula (R-08):
  - Sharpe ratio anualizado
  - Sortino ratio anualizado
  - Maximum Drawdown (MaxDD)
  - Calmar ratio
  - VaR 95% (histórico y paramétrico)
  - CVaR 95% (Expected Shortfall)
  - Retorno neto anualizado
  - Turnover promedio anual
  - Beta vs S&P500
  - Information Ratio vs benchmark

Bonus:
  - bootstrap_ic: Intervalo de confianza 95% para Sharpe via bootstrap
    con 1000 iteraciones (R-06).

Restricciones cubiertas: R-06, R-08.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("metrics_calculator")

TRADING_DAYS = 252
_DEFAULT_RISK_FREE = 0.0
_DEFAULT_CONFIDENCE = 0.95
_DEFAULT_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class PortfolioMetrics:
    """Full performance metrics for a backtest result."""

    # Return metrics
    total_return: float
    annualized_return: float

    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float

    # Risk metrics
    annualized_volatility: float
    max_drawdown: float
    var_95_historical: float
    var_95_parametric: float
    cvar_95: float

    # Market metrics
    beta: float
    alpha: float

    # Operational metrics
    avg_annual_turnover: float
    n_rebalances: int

    # Bootstrap confidence interval for Sharpe (R-06)
    sharpe_ci_low: float
    sharpe_ci_high: float

    # Metadata
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    n_days: int = 0


# ===================================================================
# Individual metric functions
# ===================================================================
def annualized_return(returns: pd.Series) -> float:
    """Compute annualized geometric mean return.

    Args:
        returns: Daily return series.

    Returns:
        Annualized return as a decimal.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    total = float((1.0 + returns).prod())
    years = n / TRADING_DAYS
    if total <= 0 or years <= 0:
        return 0.0
    return float(total ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    """Compute annualized return volatility.

    Args:
        returns: Daily return series.

    Returns:
        Annualized volatility (std dev).
    """
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
) -> float:
    """Compute annualized Sharpe ratio.

    Sharpe = (E[r] - rf) / std(r), annualized.

    Args:
        returns: Daily return series.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Annualized Sharpe ratio.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    std = float(excess.std())
    if std < 1e-10:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
) -> float:
    """Compute annualized Sortino ratio.

    Sortino = (E[r] - rf) / downside_std(r), annualized.
    Downside std uses only negative excess returns.

    Args:
        returns: Daily return series.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Annualized Sortino ratio.
    """
    if len(returns) < 2:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = returns - daily_rf
    downside = excess[excess < 0.0]
    if len(downside) < 2:
        return float(excess.mean() * TRADING_DAYS / 1e-10)
    downside_std = float(downside.std() * np.sqrt(TRADING_DAYS))
    if downside_std < 1e-10:
        return 0.0
    ann_excess = float(excess.mean() * TRADING_DAYS)
    return ann_excess / downside_std


def max_drawdown(nav: pd.Series) -> float:
    """Compute maximum drawdown from peak NAV.

    MaxDD = min( (NAV_t - peak_t) / peak_t )

    Args:
        nav: NAV series (levels, not returns).

    Returns:
        Maximum drawdown as a negative fraction (e.g., -0.25).
    """
    if len(nav) < 2:
        return 0.0
    peak = nav.cummax()
    dd = (nav - peak) / peak
    return float(dd.min())


def calmar_ratio(
    returns: pd.Series,
    nav: pd.Series,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
) -> float:
    """Compute Calmar ratio.

    Calmar = annualized_return / abs(max_drawdown)

    Args:
        returns: Daily return series.
        nav: NAV series.
        risk_free_rate: Annual risk-free rate (unused, kept for interface).

    Returns:
        Calmar ratio (positive).
    """
    ann_ret = annualized_return(returns)
    mdd = abs(max_drawdown(nav))
    if mdd < 1e-10:
        return 0.0
    return ann_ret / mdd


def var_historical(
    returns: pd.Series,
    confidence: float = _DEFAULT_CONFIDENCE,
) -> float:
    """Compute historical Value at Risk.

    VaR = quantile(returns, 1 - confidence).
    Reported as a positive number (loss magnitude).

    Args:
        returns: Daily return series.
        confidence: Confidence level (e.g., 0.95).

    Returns:
        VaR as a positive loss value.
    """
    if len(returns) < 10:
        return 0.0
    return float(-np.quantile(returns, 1.0 - confidence))


def var_parametric(
    returns: pd.Series,
    confidence: float = _DEFAULT_CONFIDENCE,
) -> float:
    """Compute parametric (Gaussian) Value at Risk.

    VaR = -(mu + z * sigma), where z = Phi^{-1}(1 - confidence).

    Args:
        returns: Daily return series.
        confidence: Confidence level.

    Returns:
        Parametric VaR as a positive loss value.
    """
    if len(returns) < 10:
        return 0.0
    mu = float(returns.mean())
    sigma = float(returns.std())
    z = float(stats.norm.ppf(1.0 - confidence))
    return float(-(mu + z * sigma))


def cvar(
    returns: pd.Series,
    confidence: float = _DEFAULT_CONFIDENCE,
) -> float:
    """Compute Conditional Value at Risk (Expected Shortfall).

    CVaR = -E[r | r <= VaR quantile]

    Args:
        returns: Daily return series.
        confidence: Confidence level.

    Returns:
        CVaR as a positive loss value.
    """
    if len(returns) < 10:
        return 0.0
    threshold = np.quantile(returns, 1.0 - confidence)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return var_historical(returns, confidence)
    return float(-tail.mean())


def beta_alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
) -> Tuple[float, float]:
    """Compute portfolio beta and alpha vs benchmark.

    Uses OLS: r_p - rf = alpha + beta * (r_b - rf) + epsilon.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Tuple (beta, annualized_alpha).
    """
    daily_rf = risk_free_rate / TRADING_DAYS
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 20:
        return 1.0, 0.0

    rp = portfolio_returns.reindex(common) - daily_rf
    rb = benchmark_returns.reindex(common) - daily_rf

    cov = float(rp.cov(rb))
    var_b = float(rb.var())
    if var_b < 1e-12:
        return 1.0, 0.0

    b = cov / var_b
    a = float(rp.mean() - b * rb.mean()) * TRADING_DAYS
    return float(b), float(a)


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Compute Information Ratio vs benchmark.

    IR = annualized(active_return) / tracking_error

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.

    Returns:
        Information ratio.
    """
    common = portfolio_returns.index.intersection(benchmark_returns.index)
    if len(common) < 20:
        return 0.0

    active = portfolio_returns.reindex(common) - benchmark_returns.reindex(common)
    te = float(active.std())
    if te < 1e-10:
        return 0.0
    return float(active.mean() / te * np.sqrt(TRADING_DAYS))


def avg_annual_turnover(turnover_series: pd.Series) -> float:
    """Compute average annual one-way turnover.

    Sums daily turnover values (non-zero on rebalance dates) and
    scales to annual by multiplying by TRADING_DAYS / n_days.

    Args:
        turnover_series: Daily turnover series (0 on non-rebalance days).

    Returns:
        Average annualized turnover.
    """
    if len(turnover_series) < 2:
        return 0.0
    total = float(turnover_series.sum())
    years = len(turnover_series) / TRADING_DAYS
    return total / years if years > 0 else 0.0


# ===================================================================
# Bootstrap confidence interval for Sharpe (R-06)
# ===================================================================
def bootstrap_sharpe_ci(
    returns: pd.Series,
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
    confidence: float = _DEFAULT_CONFIDENCE,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
    seed: int = _BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Compute 95% bootstrap confidence interval for Sharpe ratio (R-06).

    Resamples with replacement n_bootstrap times and computes Sharpe
    for each resample. Returns the [2.5%, 97.5%] percentile interval.

    Args:
        returns: Daily return series.
        n_bootstrap: Number of bootstrap iterations (default 1000).
        confidence: Interval confidence (0.95 → [2.5%, 97.5%]).
        risk_free_rate: Annual risk-free rate.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple (ci_low, ci_high).
    """
    if len(returns) < 20:
        s = sharpe_ratio(returns, risk_free_rate)
        return s, s

    rng = np.random.default_rng(seed)
    ret_arr = returns.values
    n = len(ret_arr)
    daily_rf = risk_free_rate / TRADING_DAYS

    sharpe_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = ret_arr[idx] - daily_rf
        std = sample.std()
        sharpe_samples[i] = (
            sample.mean() / std * np.sqrt(TRADING_DAYS) if std > 1e-10 else 0.0
        )

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(sharpe_samples, alpha))
    ci_high = float(np.quantile(sharpe_samples, 1.0 - alpha))
    return ci_low, ci_high


# ===================================================================
# Unified metrics computation
# ===================================================================
def compute_metrics(
    returns: pd.Series,
    nav: pd.Series,
    turnover: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = _DEFAULT_RISK_FREE,
    n_bootstrap: int = _DEFAULT_N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> PortfolioMetrics:
    """Compute full suite of portfolio performance metrics (R-08).

    Args:
        returns: Daily return series indexed by date.
        nav: Daily NAV series indexed by date.
        turnover: Daily one-way turnover (0 on non-rebalance days).
                  None → avg_annual_turnover = NaN.
        benchmark_returns: Benchmark daily returns for beta/IR.
                           None → beta=1.0, alpha=0.0, IR=0.0.
        risk_free_rate: Annual risk-free rate.
        n_bootstrap: Bootstrap iterations for Sharpe CI (R-06).
        seed: RNG seed.

    Returns:
        PortfolioMetrics dataclass with all computed metrics.
    """
    clean_ret = returns.dropna()
    clean_nav = nav.dropna()

    ann_ret = annualized_return(clean_ret)
    ann_vol = annualized_volatility(clean_ret)
    sr = sharpe_ratio(clean_ret, risk_free_rate)
    so = sortino_ratio(clean_ret, risk_free_rate)
    mdd = max_drawdown(clean_nav)
    cal = calmar_ratio(clean_ret, clean_nav, risk_free_rate)
    var_h = var_historical(clean_ret)
    var_p = var_parametric(clean_ret)
    cvar_val = cvar(clean_ret)
    total_ret = float((1.0 + clean_ret).prod() - 1.0)

    # Benchmark metrics
    if benchmark_returns is not None and len(benchmark_returns) > 20:
        b, a = beta_alpha(clean_ret, benchmark_returns, risk_free_rate)
        ir = information_ratio(clean_ret, benchmark_returns)
    else:
        b, a, ir = 1.0, 0.0, 0.0

    # Turnover
    annual_to = (
        avg_annual_turnover(turnover.fillna(0.0))
        if turnover is not None
        else 0.0
    )
    n_rebal = (
        int((turnover > 0).sum())
        if turnover is not None
        else 0
    )

    # Bootstrap CI for Sharpe (R-06)
    ci_low, ci_high = bootstrap_sharpe_ci(
        clean_ret, n_bootstrap=n_bootstrap,
        risk_free_rate=risk_free_rate, seed=seed,
    )

    start = clean_ret.index[0] if len(clean_ret) > 0 else None
    end = clean_ret.index[-1] if len(clean_ret) > 0 else None

    logger.info(
        "compute_metrics: total_ret=%.2f%%, Sharpe=%.3f, "
        "MaxDD=%.2f%%, VaR95=%.4f, CVaR95=%.4f, "
        "Sharpe_CI=[%.3f, %.3f]",
        total_ret * 100, sr, mdd * 100,
        var_h, cvar_val, ci_low, ci_high,
    )

    return PortfolioMetrics(
        total_return=round(total_ret, 6),
        annualized_return=round(ann_ret, 6),
        sharpe_ratio=round(sr, 4),
        sortino_ratio=round(so, 4),
        calmar_ratio=round(cal, 4),
        information_ratio=round(ir, 4),
        annualized_volatility=round(ann_vol, 6),
        max_drawdown=round(mdd, 6),
        var_95_historical=round(var_h, 6),
        var_95_parametric=round(var_p, 6),
        cvar_95=round(cvar_val, 6),
        beta=round(b, 4),
        alpha=round(a, 6),
        avg_annual_turnover=round(annual_to, 4),
        n_rebalances=n_rebal,
        sharpe_ci_low=round(ci_low, 4),
        sharpe_ci_high=round(ci_high, 4),
        start_date=start,
        end_date=end,
        n_days=len(clean_ret),
    )


def metrics_to_dict(m: PortfolioMetrics) -> Dict[str, float]:
    """Convert PortfolioMetrics to a flat dictionary.

    Args:
        m: PortfolioMetrics instance.

    Returns:
        Dict of metric_name → value.
    """
    return {
        "total_return": m.total_return,
        "annualized_return": m.annualized_return,
        "annualized_volatility": m.annualized_volatility,
        "sharpe_ratio": m.sharpe_ratio,
        "sortino_ratio": m.sortino_ratio,
        "calmar_ratio": m.calmar_ratio,
        "information_ratio": m.information_ratio,
        "max_drawdown": m.max_drawdown,
        "var_95_historical": m.var_95_historical,
        "var_95_parametric": m.var_95_parametric,
        "cvar_95": m.cvar_95,
        "beta": m.beta,
        "alpha": m.alpha,
        "avg_annual_turnover": m.avg_annual_turnover,
        "n_rebalances": m.n_rebalances,
        "sharpe_ci_low": m.sharpe_ci_low,
        "sharpe_ci_high": m.sharpe_ci_high,
        "n_days": m.n_days,
    }


def compare_metrics(
    results: Dict[str, PortfolioMetrics],
) -> pd.DataFrame:
    """Build a comparison table from multiple PortfolioMetrics.

    Args:
        results: Dict {strategy_label → PortfolioMetrics}.

    Returns:
        DataFrame with strategies as columns, metrics as rows.
    """
    rows = {label: metrics_to_dict(m) for label, m in results.items()}
    return pd.DataFrame(rows)
