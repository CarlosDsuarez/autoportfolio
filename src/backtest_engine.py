"""
backtest_engine.py — T-25: Motor de backtesting rolling OOS.

Loop temporal:
  for t in rebalance_dates:
      1. Obtener ventana de entrenamiento (get_window, R-05)
      2. Filtrar universo activo (filter_universe_at_date, R-03)
      3. Estimar mu y Sigma  OR  HRP sizing (Fase 6)
      4. Optimizar portafolio (portfolio_optimizer.optimize / hrp_weights)
      5. Reequilibrar (rebalancing_engine.rebalance, R-07)
         o PositionLedger (Fase 6, simulación sin broker)
      6. Simular NAV diario en periodo OOS hasta siguiente rebalanceo

Fase 6 (extensión):
  - freq='15D' (días hábiles) vía get_rebalance_dates
  - opt_method='hrp'
  - use_signal_stack → regime + signals + conviction
  - use_position_ledger → trades/shares/FIFO audit trail

Registra por fecha: NAV, pesos, trades, costos, turnover.
Validacion de no look-ahead en cada iteracion (R-05).
seed(42) global para reproducibilidad (R-10).

Restricciones cubiertas: R-01, R-02, R-03, R-05, R-06, R-07, R-09, R-10.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.rolling_engine import get_window, get_rebalance_dates
from src.expected_returns import estimate_expected_returns
from src.covariance_estimators import estimate_covariance
from src.universe_filter import filter_universe_at_date
from src.portfolio_optimizer import optimize
from src.rebalancing_engine import rebalance

if TYPE_CHECKING:
    from src.position_ledger import LedgerSnapshot
    from src.signal_stack import SignalStackConfig

logger = logging.getLogger("backtest_engine")

SEED = 42
TRADING_DAYS = 252
_INITIAL_NAV = 1_000_000.0


# ===================================================================
# Data structures
# ===================================================================
@dataclass
class BacktestConfig:
    """Configuration for a single backtest run.

    Fase 5 defaults preserved. Fase 6 fields are opt-in so existing
    tests and callers keep identical behaviour.
    """

    opt_method: str = "mv_classic"     # portfolio_optimizer method OR 'hrp'
    rebalance_freq: str = "ME"         # 'ME'/'QE'/'M'/'Q'/'15D'/...
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
    # --- Fase 6 ---
    use_signal_stack: bool = False
    use_position_ledger: bool = False
    signal_config: Optional["SignalStackConfig"] = None
    regime_window: int = 63
    conviction_top_k: Optional[int] = None  # None = tilt all active names


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
    # Fase 6 extensions (additive; empty/None when unused)
    ledger_snapshots: List["LedgerSnapshot"] = field(default_factory=list)
    regime_series: Optional[pd.Series] = None


# ===================================================================
# Helpers (Fase 6)
# ===================================================================
def _estimate_cost_fraction(
    w_prev: pd.Series,
    w_new: pd.Series,
    commission_bps: float,
    spread_bps: float,
) -> float:
    """Rough L1-turnover based cost fraction when not using rebalance()."""
    aligned = w_prev.reindex(w_new.index).fillna(0.0)
    turnover = float((w_new - aligned).abs().sum())
    bps = commission_bps + spread_bps
    return turnover * (bps / 10_000.0)


_CAPM_BENCHMARK = "SPY"


def _estimate_mu_for_active(
    config: BacktestConfig,
    wp: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.Series:
    """Estimate expected returns on the active window, with CAPM guardrails.

    ``capm_shrunk`` needs the benchmark column for betas. The active
    investable set often excludes SPY (liquidity filter / stock-only
    universe). Estimating on ``wp`` alone then raises, and ``run_backtest``
    swallows the error → silent flat NAV / zero rebalances.

    When the benchmark is missing from ``wp`` but present in ``prices`` on
    the same dates, append it for estimation only, then drop it from mu.
    If the benchmark is unavailable entirely, fall back to historical mu
    rather than failing the rebalance.
    """
    method = (config.mu_method or "historical").lower().strip()
    mu_prices = wp

    if method == "capm_shrunk" and _CAPM_BENCHMARK not in wp.columns:
        if _CAPM_BENCHMARK in prices.columns:
            spy = prices[_CAPM_BENCHMARK].reindex(wp.index)
            if spy.notna().all() and len(spy) == len(wp):
                mu_prices = wp.copy()
                mu_prices[_CAPM_BENCHMARK] = spy.astype(float)
            else:
                logger.warning(
                    "capm_shrunk: %s incomplete on active window — "
                    "falling back to historical mu.",
                    _CAPM_BENCHMARK,
                )
                method = "historical"
        else:
            logger.warning(
                "capm_shrunk: %s not in price panel — "
                "falling back to historical mu.",
                _CAPM_BENCHMARK,
            )
            method = "historical"

    mu = estimate_expected_returns(mu_prices, method=method)
    # Drop benchmark (and any extras) so mu aligns with Sigma / active set
    mu = mu.reindex(wp.columns)
    if mu.isna().any():
        mu = mu.fillna(float(mu.mean()) if mu.notna().any() else 0.0)
    return mu


def _build_target_weights(
    config: BacktestConfig,
    wp: pd.DataFrame,
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    t: pd.Timestamp,
    active_clean: List[str],
    w_prev: Optional[pd.Series],
) -> tuple[pd.Series, dict, Optional[str]]:
    """Build target weights via optimizer, HRP, and optional signal stack.

    Returns:
        (w_target, opt_meta, regime_state_or_None)
    """
    regime_state: Optional[str] = None
    conviction = None

    if config.use_signal_stack:
        from src.regime_detector import classify_regime
        from src.signal_stack import compute_signal_scores
        from src.conviction_scoring import conviction_score, conviction_to_tilts

        report = classify_regime(
            prices[active_clean] if set(active_clean) <= set(prices.columns)
            else prices,
            t,
            window=config.regime_window,
        )
        regime_state = report.state
        signal_scores = compute_signal_scores(
            prices, volumes, t, config=config.signal_config,
        )
        signal_scores = signal_scores.reindex(active_clean).fillna(0.0)
        conviction = conviction_score(signal_scores, regime_state)
        if config.conviction_top_k is not None and config.conviction_top_k > 0:
            top = conviction.nlargest(min(config.conviction_top_k, len(conviction)))
            conviction = conviction.where(conviction.index.isin(top.index), other=-1e9)

    # --- Sizing ---
    if config.opt_method == "hrp":
        from src.hrp_sizing import hrp_weights
        rets = wp.pct_change().dropna(how="all")
        w_target = hrp_weights(rets)
        w_target = w_target.reindex(active_clean).fillna(0.0)
        s = w_target.sum()
        w_target = w_target / s if s > 1e-10 else pd.Series(
            np.ones(len(active_clean)) / len(active_clean), index=active_clean,
        )
        # Soft max-weight cap
        w_target = w_target.clip(upper=config.max_weight)
        w_target = w_target / w_target.sum()
        opt_meta = {
            "expected_sharpe": 0.0,
            "converged": True,
            "method": "hrp",
        }
    else:
        sigma = estimate_covariance(wp, method=config.cov_method)
        mu = _estimate_mu_for_active(
            config=config,
            wp=wp,
            prices=prices,
        )
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
        opt_meta = {
            "expected_sharpe": opt_result.get("expected_sharpe", 0.0),
            "converged": opt_result.get("converged", False),
            "method": config.opt_method,
        }

    # Apply conviction tilts if signal stack active
    if conviction is not None:
        from src.conviction_scoring import conviction_to_tilts
        w_target = conviction_to_tilts(conviction, base_weights=w_target)
        # Re-apply max weight
        w_target = w_target.clip(upper=config.max_weight)
        w_target = w_target / w_target.sum() if w_target.sum() > 1e-10 else w_target

    return w_target, opt_meta, regime_state


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
      3. estimate mu + Sigma on active subset (or HRP).
      4. optimize() / hrp_weights() → target weights.
      5. rebalance() or PositionLedger → executed weights with costs.
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

    # Rebalance dates (supports M/Q/ME/QE and '<N>D')
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
        "signals=%s, ledger=%s, %d rebalance dates [%s → %s]",
        config.opt_method, config.rebalance_freq, config.window,
        config.use_signal_stack, config.use_position_ledger,
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
    ledger_snapshots: List = []
    regime_records: Dict[pd.Timestamp, str] = {}

    ledger = None
    if config.use_position_ledger:
        from src.position_ledger import PositionLedger
        ledger = PositionLedger(initial_cash=config.initial_nav)

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

                w_target, opt_meta, regime_state = _build_target_weights(
                    config=config,
                    wp=wp,
                    prices=prices,
                    volumes=volumes,
                    t=t,
                    active_clean=active_clean,
                    w_prev=w_prev,
                )
                if regime_state is not None:
                    regime_records[t] = regime_state

                # --- Execution path ---
                if config.use_position_ledger and ledger is not None:
                    prices_t = prices.loc[t]
                    nav_pre = ledger.nav(prices_t)
                    if w_prev is None:
                        w_prev_exec = pd.Series(0.0, index=active_clean)
                    else:
                        w_prev_exec = w_prev

                    # Approximate turnover cost before applying trades
                    cost_frac = _estimate_cost_fraction(
                        w_prev_exec, w_target,
                        config.commission_bps, config.spread_bps,
                    )
                    cost_abs = cost_frac * max(nav_pre, 1e-6)
                    turnover_today = float(
                        (w_target.reindex(w_prev_exec.index).fillna(0.0)
                         - w_prev_exec).abs().sum()
                    )
                    # Also count new names
                    extra = w_target.index.difference(w_prev_exec.index)
                    turnover_today += float(w_target.reindex(extra).fillna(0.0).abs().sum())

                    trades = ledger.trades_from_target_weights(
                        w_target, prices_t, nav=nav_pre,
                    )
                    ledger.apply_trades(
                        trades, prices_t, costs=cost_abs, t=t,
                    )
                    snap = ledger.snapshot(t, prices_t)
                    ledger_snapshots.append(snap)
                    nav = snap.nav
                    w_current = ledger.weights_from_positions(prices_t)
                    # Normalize invested weights to sum 1 for compatibility
                    if w_current.sum() > 1e-10:
                        w_current = w_current / w_current.sum()
                    cost_today = cost_frac
                else:
                    # Classic weight-space path (Fase 5)
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
                    cost_today = (
                        rebal_result.cost / nav
                        if nav > 1e-6 else 0.0
                    )

                rebalance_log_rows.append({
                    "date": t,
                    "opt_method": config.opt_method,
                    "n_active": len(active_clean),
                    "expected_sharpe": opt_meta.get("expected_sharpe", 0.0),
                    "turnover": turnover_today,
                    "cost_fraction": cost_today,
                    "rule": config.replacement_rule,
                    "converged": opt_meta.get("converged", False),
                    "regime": regime_state,
                })

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Rebalance at %s failed: %s. Keeping previous weights.",
                    t.date(), exc,
                )

        # --- NAV update ---
        if config.use_position_ledger and ledger is not None:
            prices_t = prices.loc[t] if t in prices.index else None
            if prices_t is not None:
                nav_new = ledger.nav(prices_t)
                if i > 0 and nav > 1e-10:
                    # Cost already deducted on rebalance day inside ledger
                    daily_ret = (nav_new / nav) - 1.0
                    if is_rebal and cost_today > 0 and abs(daily_ret + cost_today) > 1e-15:
                        # cost already in nav_new; report gross-ish daily_ret
                        pass
                else:
                    daily_ret = 0.0
                nav = nav_new
                w_mtm = ledger.weights_from_positions(prices_t)
                if w_mtm.sum() > 1e-10:
                    w_current = w_mtm / w_mtm.sum()
            else:
                daily_ret = 0.0
        elif w_current is not None and i > 0:
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

    regime_series = (
        pd.Series(regime_records, dtype=object) if regime_records else None
    )

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
        ledger_snapshots=ledger_snapshots,
        regime_series=regime_series,
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
