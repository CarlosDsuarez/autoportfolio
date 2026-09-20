"""
experiment_4_regime_analysis.py — T-30: Análisis por régimen de mercado.

Analiza el rendimiento del portafolio MV clásico en cinco regímenes:
  1. Bull 2017-2019
  2. Crash 2020-Q1 (COVID)
  3. Recovery 2020-Q2 – 2021
  4. Bear 2022 (Fed tightening)
  5. Rebound 2023

Para cada régimen reporta: Sharpe, MaxDD, Calmar, Ann.Return,
Volatility, VaR95, CVaR95.

Genera:
  - Tabla por régimen
  - Subplots de NAV por régimen con métricas anotadas

Uso:
  python -m experiments.experiment_4_regime_analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
import pandas as pd
import numpy as np

from src.data_client import load_universe, fetch_prices, fetch_volumes, clean_prices
from src.backtest_engine import BacktestConfig, run_backtest
from src.metrics_calculator import compute_metrics, compare_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("exp4")

# Universo canónico: 100 principales del S&P 500 (universe.csv)
TICKERS = load_universe()["ticker"].tolist()

# Full history needed to include warmup before first regime
START = "2022-01-01"
END = "2023-12-31"

REGIMES = {
    "bear_2022":          ("2022-01-01", "2022-12-31"),
    "rally_2023":         ("2023-01-01", "2023-12-31"),
    "continuation_2024":  ("2024-01-01", "2024-12-31"),
}

BACKTEST_CONFIG = BacktestConfig(
    opt_method="mv_classic",
    rebalance_freq="15D",
    window=252,
    warmup=252,
)


def run() -> pd.DataFrame:
    """Run Experiment 4 and return per-regime metrics DataFrame."""
    logger.info("=== Experiment 4: Regime Analysis ===")

    logger.info("Downloading price data...")
    prices_raw = fetch_prices(TICKERS, start=START, end=END)
    volumes_raw = fetch_volumes(TICKERS, start=START, end=END)
    prices, _ = clean_prices(prices_raw)
    volumes = volumes_raw.reindex(prices.index).ffill().fillna(1e6)

    # Run full backtest over entire period
    logger.info("Running full-period backtest...")
    full_result = run_backtest(prices, volumes, BACKTEST_CONFIG)

    # Slice results by regime
    metrics_by_regime = {}
    for regime_name, (r_start, r_end) in REGIMES.items():
        mask = (full_result.returns.index >= r_start) & \
               (full_result.returns.index <= r_end)
        regime_returns = full_result.returns[mask]
        regime_nav = full_result.nav[mask]

        if len(regime_returns) < 10:
            logger.warning("Not enough data for regime %s. Skipping.", regime_name)
            continue

        turnover_s = pd.Series({
            r.date: r.turnover
            for r in full_result.daily_records
            if r_start <= str(r.date.date()) <= r_end
        })

        m = compute_metrics(
            returns=regime_returns,
            nav=regime_nav,
            turnover=turnover_s if len(turnover_s) > 0 else None,
        )
        metrics_by_regime[regime_name] = m

        logger.info(
            "Regime %s [%s→%s]: Sharpe=%.3f, MaxDD=%.1f%%, Ann.Ret=%.1f%%",
            regime_name, r_start, r_end,
            m.sharpe_ratio, m.max_drawdown * 100, m.annualized_return * 100,
        )

    table = compare_metrics(metrics_by_regime)
    print("\n=== Experiment 4 Results ===")
    print(table.T.to_string())

    os.makedirs("results", exist_ok=True)
    table.T.to_csv("results/exp4_regime_analysis.csv")

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        # Full NAV with regime bands
        ax0 = axes[0]
        nav_norm = full_result.nav / full_result.nav.iloc[0]
        ax0.plot(nav_norm.index, nav_norm.values, color="steelblue", linewidth=1.5)

        regime_colors = {
            "bull_2017_2019":      "lightgreen",
            "crash_2020_Q1":       "lightcoral",
            "recovery_2020_2021":  "lightyellow",
            "bear_2022":           "lightsalmon",
            "rebound_2023":        "lightblue",
        }
        for rname, (rs, re) in REGIMES.items():
            ax0.axvspan(
                pd.Timestamp(rs), pd.Timestamp(re),
                alpha=0.3, color=regime_colors.get(rname, "grey"),
                label=rname,
            )
        ax0.set_title("Full Period NAV with Regime Bands")
        ax0.set_ylabel("NAV (base=1)")
        ax0.legend(fontsize=7, loc="upper left")
        ax0.grid(alpha=0.3)

        # Per-regime NAV zoom
        for idx, (regime_name, (r_start, r_end)) in enumerate(REGIMES.items()):
            ax = axes[idx + 1]
            mask = (full_result.nav.index >= r_start) & \
                   (full_result.nav.index <= r_end)
            regime_nav = full_result.nav[mask]
            if len(regime_nav) > 0:
                nav_norm_r = regime_nav / regime_nav.iloc[0]
                ax.plot(nav_norm_r.index, nav_norm_r.values,
                        color=list(regime_colors.values())[idx + 1]
                        if idx + 1 < len(regime_colors) else "steelblue",
                        linewidth=1.5)

                if regime_name in metrics_by_regime:
                    m = metrics_by_regime[regime_name]
                    ax.set_title(
                        f"{regime_name}\n"
                        f"Sharpe={m.sharpe_ratio:.2f} | "
                        f"MaxDD={m.max_drawdown*100:.1f}% | "
                        f"Ann={m.annualized_return*100:.1f}%",
                        fontsize=9,
                    )
                else:
                    ax.set_title(regime_name)
            ax.grid(alpha=0.3)
            ax.set_ylabel("NAV (base=1)")

        # Remove unused subplot
        if len(REGIMES) + 1 < len(axes):
            for extra_ax in axes[len(REGIMES) + 1:]:
                extra_ax.set_visible(False)

        plt.tight_layout()
        plt.savefig("results/exp4_regime_analysis.png", dpi=120)
        logger.info("Plot saved: results/exp4_regime_analysis.png")
        plt.close()
    except ImportError:
        logger.warning("matplotlib not available — skipping plots.")

    return table


if __name__ == "__main__":
    run()
