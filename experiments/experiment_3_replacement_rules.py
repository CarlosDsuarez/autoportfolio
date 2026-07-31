"""
experiment_3_replacement_rules.py — T-29: Reglas de reemplazo de activos.

Compara cuatro reglas de reemplazo cada trimestre:
  1. Sin reemplazo (none)
  2. Momentum (top performers del universo)
  3. Aleatorio (baseline estadístico)
  4. Mínima varianza (menor contribución marginal al riesgo)

Test estadístico: bootstrap IC 95% para Sharpe (R-06) para
determinar si las diferencias son estadísticamente significativas.

Genera:
  - Tabla de métricas
  - NAV curves por regla
  - Box plot de Sharpe bootstrapped

Uso:
  python -m experiments.experiment_3_replacement_rules
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd
import numpy as np

from src.data_client import load_universe, fetch_prices, fetch_volumes, clean_prices
from src.backtest_engine import BacktestConfig, run_comparison
from src.metrics_calculator import (
    compute_metrics, compare_metrics, bootstrap_sharpe_ci,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("exp3")

# Universo canónico: 100 principales del S&P 500 (universe.csv)
TICKERS = load_universe()["ticker"].tolist()
START = "2022-01-01"
END = "2023-12-31"
WARMUP = 252
WINDOW = 252
REBAL_FREQ = "15D"

# Portfolio inicial = top 50 por orden del CSV; el resto es pool de reemplazo
PORTFOLIO_TICKERS = TICKERS[:50]

CONFIGS = {
    "no_replace": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        replacement_rule="none",
    ),
    "momentum": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        replacement_rule="momentum",
    ),
    "random": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        replacement_rule="random",
    ),
    "min_variance": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        replacement_rule="min_variance",
    ),
}

_N_BOOTSTRAP = 1000
_BOOTSTRAP_SEED = 42


def _bootstrap_distribution(returns: pd.Series, n: int = _N_BOOTSTRAP) -> np.ndarray:
    """Sample Sharpe distribution for box-plot visualization."""
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    arr = returns.values
    k = len(arr)
    samples = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, k, size=k)
        s = arr[idx]
        std = s.std()
        samples[i] = s.mean() / std * np.sqrt(252) if std > 1e-10 else 0.0
    return samples


def run() -> pd.DataFrame:
    """Run Experiment 3 and return a metrics comparison DataFrame."""
    logger.info("=== Experiment 3: Replacement Rules Comparison ===")

    logger.info("Downloading price data...")
    prices_raw = fetch_prices(TICKERS, start=START, end=END)
    volumes_raw = fetch_volumes(TICKERS, start=START, end=END)
    prices, _ = clean_prices(prices_raw)
    volumes = volumes_raw.reindex(prices.index).ffill().fillna(1e6)

    results = run_comparison(prices, volumes, CONFIGS,
                             universe=TICKERS)

    metrics_dict = {}
    bootstrap_dists = {}
    for label, res in results.items():
        turnover_s = pd.Series({r.date: r.turnover for r in res.daily_records})
        m = compute_metrics(
            returns=res.returns,
            nav=res.nav,
            turnover=turnover_s,
        )
        metrics_dict[label] = m
        bootstrap_dists[label] = _bootstrap_distribution(res.returns.dropna())

        logger.info(
            "%s → Sharpe=%.3f [%.3f, %.3f], Ann.Ret=%.1f%%, MaxDD=%.1f%%",
            label, m.sharpe_ratio, m.sharpe_ci_low, m.sharpe_ci_high,
            m.annualized_return * 100, m.max_drawdown * 100,
        )

    # Statistical significance check: do CIs overlap with 'no_replace'?
    base_ci = (metrics_dict["no_replace"].sharpe_ci_low,
               metrics_dict["no_replace"].sharpe_ci_high)
    logger.info(
        "Base (no_replace) Sharpe CI: [%.3f, %.3f]",
        base_ci[0], base_ci[1],
    )
    for label in ["momentum", "random", "min_variance"]:
        lo = metrics_dict[label].sharpe_ci_low
        hi = metrics_dict[label].sharpe_ci_high
        overlap = not (hi < base_ci[0] or lo > base_ci[1])
        logger.info(
            "  %s CI [%.3f, %.3f] — overlaps with base: %s",
            label, lo, hi, overlap,
        )

    table = compare_metrics(metrics_dict)
    print("\n=== Experiment 3 Results ===")
    print(table.T.to_string())

    os.makedirs("results", exist_ok=True)
    table.T.to_csv("results/exp3_replacement_rules.csv")

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NAV curves
        ax = axes[0]
        for label, res in results.items():
            nav_norm = res.nav / res.nav.iloc[0]
            ax.plot(nav_norm.index, nav_norm.values, label=label)
        ax.set_title("Experiment 3: NAV by Replacement Rule")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV (base=1)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Sharpe bootstrap box plots
        ax2 = axes[1]
        data_for_box = [bootstrap_dists[lbl] for lbl in CONFIGS.keys()]
        bp = ax2.boxplot(data_for_box, labels=list(CONFIGS.keys()),
                         patch_artist=True, notch=True)
        colors_box = ["steelblue", "darkorange", "green", "purple"]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title("Bootstrapped Sharpe Distribution (n=1000)")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/exp3_replacement_rules.png", dpi=120)
        logger.info("Plot saved: results/exp3_replacement_rules.png")
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")

    return table


if __name__ == "__main__":
    run()


def run_comparison(prices, volumes, configs, universe=None):
    """Thin wrapper that passes universe through."""
    from src.backtest_engine import run_backtest
    results = {}
    for label, cfg in configs.items():
        logger.info("Running backtest: %s", label)
        try:
            results[label] = run_backtest(prices, volumes, cfg,
                                          universe=universe)
        except Exception as exc:
            logger.error("Backtest '%s' failed: %s", label, exc)
    return results
