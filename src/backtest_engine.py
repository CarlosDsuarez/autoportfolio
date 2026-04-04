"""
backtest_engine.py — T-25: Motor de backtesting rolling OOS.

Loop temporal:
  for t in rebalance_dates:
      1. Obtener ventana de entrenamiento (get_window, R-05)
      2. Filtrar universo activo (filter_universe_at_date, R-03)
      3. Estimar mu y Sigma
      4. Optimizar portafolio (portfolio_optimizer.optimize)
      5. Reequilibrar (rebalancing_engine.rebalance, R-07)
      6. Simular NAV diario en periodo OOS hasta siguiente rebalanceo

Registra por fecha: NAV, pesos, trades, costos, turnover.
Validacion de no look-ahead en cada iteracion (R-05).
seed(42) global para reproducibilidad (R-10).

Restricciones cubiertas: R-01, R-02, R-03, R-05, R-06, R-07, R-09, R-10.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from src.rolling_engine import get_window, get_rebalance_dates
from src.expected_returns import estimate_expected_returns
from src.covariance_estimators import estimate_covariance
from src.universe_filter import filter_universe_at_date
from src.portfolio_optimizer import optimize
from src.rebalancing_engine import rebalance

logger = logging.getLogger("backtest_engine")

SEED = 42
TRADING_DAYS = 252
_INITIAL_NAV = 1_000_000.0


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class BacktestConfig:
    """Configuration for a single backtest run."""

    opt_method: str = "mv_classic"     # portfolio_optimizer method
    rebalance_freq: str = "ME"         # 'ME'=monthly, 'QE'=quarterly (pandas 2.2+)
    window: int = 252                  # training window (days)
    warmup: int = 252                  # warmup before first rebalance
    mu_method: str = "historical"      # expected returns estimator
    cov_method: str = "ledoit_wolf"    # covariance estimator
    replacement_rule: str = "none"     # 'none','momentum','random','min_variance'
    max_weight: float = 0.30
    turnover_limit: float = 0.20
    tc_lambda: float = 0.001
    kappa: float = 0.10                # robustness (for robust methods)
    initial_nav: float = _INITIAL_NAV
    commission_bps: float = 5.0
    spread_bps: float = 2.0
    impact_coef: float = 0.1
    seed: int = SEED


@dataclass
class DailyRecord:
    """Single-day record in the backtest."""

    date: pd.Timestamp
    nav: float
    daily_return: float
    weights: pd.Series
    is_rebalance: bool
    turnover: float
    cost: float


@dataclass
class BacktestResult:
    """Full backtest output."""

    config: BacktestConfig
    nav: pd.Series                  # Daily NAV indexed by date
    returns: pd.Series              # Daily gross returns
    weights: pd.DataFrame           # Daily weights (date × ticker)
    rebalance_log: pd.DataFrame     # One row per rebalance event
    daily_records: List[DailyRecord] = field(default_factory=list)


# ===================================================================
# Core backtest loop
# ===================================================================
def run_backtest(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    config: Optional[BacktestConfig] = None,
    universe: Optional[List[str]] = None,
) -> BacktestResult:
    """Execute a full rolling OOS backtest.

    For each rebalance date:
      1. get_window → training prices (R-05 strict).
      2. filter_universe_at_date → active tickers (R-03).
      3. estimate mu + Sigma on active subset.
      4. optimize() → target weights (Fase 3).
      5. rebalance() → executed weights with costs (Fase 4).
      6. Simulate daily NAV in OOS period using actual prices.

    Args:
        prices: Clean price DataFrame (date-indexed, full history).
        volumes: Volume DataFrame aligned with prices.
        config: BacktestConfig object. None → defaults.
        universe: Ticker universe. None → all price columns.

    Returns:
        BacktestResult with NAV, weights, and rebalance log.
    """
    if config is None:
        config = BacktestConfig()

    np.random.seed(config.seed)

    if universe is None:
        universe = prices.columns.tolist()

    # Rebalance dates
    rebal_dates = get_rebalance_dates(
        prices, freq=config.rebalance_freq, warmup=config.warmup
    )
    if not rebal_dates:
        raise ValueError(
            "No rebalance dates generated. "
            "Check warmup / price history length."
        )

    logger.info(
        "Starting backtest: method=%s, freq=%s, window=%d, "
        "%d rebalance dates [%s → %s]",
        config.opt_method, config.rebalance_freq, config.window,
        len(rebal_dates),
        rebal_dates[0].date(), rebal_dates[-1].date(),
    )

    # Pre-compute rolling Amihud for speed
    from src.universe_filter import precompute_rolling_metrics
    rolling_amihud, rolling_avg_vol = precompute_rolling_metrics(
        prices, volumes, window=30
    )

    nav = config.initial_nav
    w_current: Optional[pd.Series] = None
    all_tickers = prices.columns.tolist()

    nav_records: Dict[pd.Timestamp, float] = {}
    weight_records: Dict[pd.Timestamp, pd.Series] = {}
    rebalance_log_rows: List[dict] = []
    daily_records: List[DailyRecord] = []

    rebal_set = set(rebal_dates)
    all_dates = prices.index

    # Iterate over all dates after warmup
    first_rebal = rebal_dates[0]
    oos_dates = all_dates[all_dates >= first_rebal]

    for i, t in enumerate(oos_dates):
        is_rebal = t in rebal_set
        turnover_today = 0.0
        cost_today = 0.0

        # --- Rebalance step ---
        if is_rebal:
            try:
                # R-05: training window strictly before t
                window_prices = get_window(prices, t, window=config.window)

                # R-03: filter universe at t
                active, _ = filter_universe_at_date(
                    t, prices, volumes,
                    rolling_amihud=rolling_amihud,
                    rolling_avg_vol=rolling_avg_vol,
                )
                active = [tk for tk in active if tk in universe]
                if len(active) < 3:
                    active = universe

                wp = window_prices[
                    [tk for tk in active if tk in window_prices.columns]
                ].dropna(axis=1, how="any")
                if wp.shape[1] < 2:
                    wp = window_prices.dropna(axis=1, how="any")

                active_clean = wp.columns.tolist()

                # Estimate parameters
                mu = estimate_expected_returns(
                    wp, method=config.mu_method
                )
                sigma = estimate_covariance(
                    wp, method=config.cov_method
                )

                # Align previous weights to current active tickers
                if w_current is not None:
                    w_prev = w_current.reindex(active_clean).fillna(0.0)
                    s = w_prev.sum()
                    w_prev = w_prev / s if s > 1e-10 else pd.Series(
                        np.ones(len(active_clean)) / len(active_clean),
                        index=active_clean,
                    )
                else:
                    w_prev = None

                # Optimize (Fase 3)
                opt_result = optimize(
                    mu=mu,
                    sigma=sigma,
                    method=config.opt_method,
                    w_prev=w_prev,
                    max_weight=config.max_weight,
                    turnover_limit=config.turnover_limit,
                    tc_lambda=config.tc_lambda,
                    kappa=config.kappa,
                )
                w_target = opt_result["weights"]

                # Rebalance (Fase 4)
                if w_prev is None:
                    w_prev = pd.Series(
                        np.ones(len(active_clean)) / len(active_clean),
                        index=active_clean,
                    )
                rebal_result = rebalance(
                    t=t,
                    w_current=w_prev,
                    w_target=w_target,
                    prices=prices,
                    volumes=volumes,
                    rule=config.replacement_rule,
                    universe=universe,
                    active_tickers=active,
                    portfolio_value=nav,
                    turnover_limit=config.turnover_limit,
                    is_rebalance_date=True,
                    commission_bps=config.commission_bps,
                    spread_bps=config.spread_bps,
                    impact_coef=config.impact_coef,
                )

                w_current = rebal_result.w_new
                turnover_today = rebal_result.actual_turnover
                # Cost as fraction of NAV (R-01)
                cost_today = (
                    rebal_result.cost / nav
                    if nav > 1e-6 else 0.0
                )

                rebalance_log_rows.append({
                    "date": t,
                    "opt_method": config.opt_method,
                    "n_active": len(active_clean),
                    "expected_sharpe": opt_result.get("expected_sharpe", 0.0),
                    "turnover": turnover_today,
                    "cost_fraction": cost_today,
                    "rule": config.replacement_rule,
                    "converged": opt_result.get("converged", False),
                })

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Rebalance at %s failed: %s. Keeping previous weights.",
                    t.date(), exc,
                )

        # --- NAV update ---
        if w_current is not None and i > 0:
            t_prev = oos_dates[i - 1]
            p_prev = prices.loc[t_prev] if t_prev in prices.index else None
            p_curr = prices.loc[t] if t in prices.index else None
            if p_prev is not None and p_curr is not None:
                common = w_current.index.intersection(p_prev.index).intersection(
                    p_curr.index
                )
                if len(common) > 0:
                    wf = w_current.reindex(common).fillna(0.0)
                    wf /= wf.sum() if wf.sum() > 1e-10 else 1.0
                    ret = (p_curr[common] / p_prev[common] - 1.0)
                    portfolio_ret = float((wf * ret).sum())
                    # Apply TC cost
                    nav *= (1.0 + portfolio_ret) * (1.0 - cost_today)
                    daily_ret = portfolio_ret - cost_today
                else:
                    daily_ret = 0.0
            else:
                daily_ret = 0.0
        else:
            daily_ret = 0.0

        nav_records[t] = nav
        weight_records[t] = (
            w_current.copy()
            if w_current is not None
            else pd.Series(dtype=float)
        )
        daily_records.append(DailyRecord(
            date=t,
            nav=nav,
            daily_return=daily_ret,
            weights=weight_records[t],
            is_rebalance=is_rebal,
            turnover=turnover_today,
            cost=cost_today,
        ))

    nav_series = pd.Series(nav_records, name="nav")
    returns_series = nav_series.pct_change().fillna(0.0)
    returns_series.name = "returns"

    weights_df = pd.DataFrame(weight_records).T.fillna(0.0)
    weights_df.index.name = "date"

    rebal_df = pd.DataFrame(rebalance_log_rows)
    if not rebal_df.empty:
        rebal_df = rebal_df.set_index("date")

    logger.info(
        "Backtest complete: %d days, final NAV=%.2f "
        "(%.1f%% total return), %d rebalances",
        len(nav_series), nav,
        (nav / config.initial_nav - 1) * 100,
        len(rebalance_log_rows),
    )
    return BacktestResult(
        config=config,
        nav=nav_series,
        returns=returns_series,
        weights=weights_df,
        rebalance_log=rebal_df,
        daily_records=daily_records,
    )


# ===================================================================
# Convenience: run multiple configs for comparison
# ===================================================================
def run_comparison(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    configs: Dict[str, BacktestConfig],
) -> Dict[str, BacktestResult]:
    """Run multiple backtest configurations for comparison.

    Args:
        prices: Clean price DataFrame.
        volumes: Volume DataFrame.
        configs: Dict {label → BacktestConfig}.

    Returns:
        Dict {label → BacktestResult}.
    """
    results = {}
    for label, cfg in configs.items():
        logger.info("Running backtest: %s", label)
        try:
            results[label] = run_backtest(prices, volumes, cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("Backtest '%s' failed: %s", label, exc)
    return results
