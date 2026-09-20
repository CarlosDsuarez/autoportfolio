"""
experiment_1_mv_vs_robust.py — T-27: MV clásico vs. optimizador robusto.

Compara tres configuraciones:
  1. MV clásico (mv_classic)
  2. Robusto kappa=0.10 (robust_k010)
  3. Robusto kappa=0.25 (robust_k025)

Métricas reportadas (R-08): Sharpe, Sortino, MaxDD, Calmar, VaR95,
CVaR95, retorno anualizado, Sharpe CI 95% (R-06).

Genera tabla de resultados y dos figuras:
  - NAV curves comparadas
  - Sharpe CIs con barras de error

Uso:
  python -m experiments.experiment_1_mv_vs_robust
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd
import numpy as np

from src.data_client import load_universe, fetch_prices, fetch_volumes, clean_prices
from src.backtest_engine import BacktestConfig, run_comparison
from src.metrics_calculator import compute_metrics, compare_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("exp1")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Universo canónico: 100 principales del S&P 500 (universe.csv)
TICKERS = load_universe()["ticker"].tolist()
START = "2022-01-01"
END = "2023-12-31"
WARMUP = 252
WINDOW = 252
REBAL_FREQ = "15D"

CONFIGS = {
    "mv_classic": BacktestConfig(
        opt_method="mv_classic",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
    ),
    "robust_k010": BacktestConfig(
        opt_method="robust",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        kappa=0.10,
    ),
    "robust_k025": BacktestConfig(
        opt_method="robust",
        rebalance_freq=REBAL_FREQ,
        window=WINDOW,
        warmup=WARMUP,
        kappa=0.25,
    ),
}


def run() -> pd.DataFrame:
    """Run Experiment 1 and return a metrics comparison DataFrame."""
    logger.info("=== Experiment 1: MV Classic vs Robust ===")

    # --- Data ---
    logger.info("Downloading price data...")
    prices_raw = fetch_prices(TICKERS, start=START, end=END)
    volumes_raw = fetch_volumes(TICKERS, start=START, end=END)
    prices, _ = clean_prices(prices_raw)
    # Align volumes
    volumes = volumes_raw.reindex(prices.index).ffill().fillna(1e6)

    logger.info("Prices shape: %s, Volumes shape: %s", prices.shape, volumes.shape)

    # --- Backtests ---
    results = run_comparison(prices, volumes, CONFIGS)

    # --- Metrics ---
    metrics_dict = {}
    for label, res in results.items():
        turnover_s = pd.Series(
            {r.date: r.turnover for r in res.daily_records}
        )
        m = compute_metrics(
            returns=res.returns,
            nav=res.nav,
            turnover=turnover_s,
        )
        metrics_dict[label] = m
        logger.info(
            "%s → Sharpe=%.3f [%.3f, %.3f], MaxDD=%.1f%%, Ann.Ret=%.1f%%",
            label, m.sharpe_ratio, m.sharpe_ci_low, m.sharpe_ci_high,
            m.max_drawdown * 100, m.annualized_return * 100,
        )

    table = compare_metrics(metrics_dict)
    print("\n=== Experiment 1 Results ===")
    print(table.T.to_string())

    # --- Save results ---
    os.makedirs("results", exist_ok=True)
    table.T.to_csv("results/exp1_mv_vs_robust.csv")

    # --- Plots (optional if matplotlib available) ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # NAV curves
        ax = axes[0]
        for label, res in results.items():
            nav_norm = res.nav / res.nav.iloc[0]
            ax.plot(nav_norm.index, nav_norm.values, label=label)
        ax.set_title("Experiment 1: NAV Comparison (Normalized)")
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV (base=1)")
        ax.legend()
        ax.grid(alpha=0.3)

        # Sharpe with CI
        ax2 = axes[1]
        labels = list(metrics_dict.keys())
        sharpes = [metrics_dict[l].sharpe_ratio for l in labels]
        ci_lows = [metrics_dict[l].sharpe_ci_low for l in labels]
        ci_highs = [metrics_dict[l].sharpe_ci_high for l in labels]
        yerr_low = [s - lo for s, lo in zip(sharpes, ci_lows)]
        yerr_high = [hi - s for s, hi in zip(sharpes, ci_highs)]
        ax2.bar(labels, sharpes, yerr=[yerr_low, yerr_high], capsize=6,
                color=["steelblue", "darkorange", "green"], alpha=0.8)
        ax2.set_title("Sharpe Ratio with 95% Bootstrap CI")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig("results/exp1_mv_vs_robust.png", dpi=120)
        logger.info("Plot saved: results/exp1_mv_vs_robust.png")
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")

    return table


if __name__ == "__main__":
    run()
