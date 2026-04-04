"""
dashboard_results.py — T-31: Dashboard interactivo con Plotly.

Genera un archivo HTML autónomo con cuatro secciones:
  1. NAV curves comparadas (todos los métodos/configs).
  2. Frontera eficiente ex-post (retorno vs volatilidad anualizada).
  3. Heatmap dinámico de pesos (fecha × ticker).
  4. Scatter turnover vs Sharpe por configuración.

Uso programático:
    from src.dashboard_results import build_dashboard
    build_dashboard(results_dict, output_path="dashboard.html")

Uso desde línea de comandos:
    python -m src.dashboard_results  (ejecuta demo con datos sintéticos)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtest_engine import BacktestResult
from src.metrics_calculator import compute_metrics, PortfolioMetrics

logger = logging.getLogger("dashboard_results")


# ===================================================================
# Helper: compute metrics for a results dict
# ===================================================================
def _compute_all_metrics(
    results: Dict[str, BacktestResult],
    risk_free_rate: float = 0.0,
) -> Dict[str, PortfolioMetrics]:
    metrics = {}
    for label, res in results.items():
        turnover_s = pd.Series({r.date: r.turnover for r in res.daily_records})
        metrics[label] = compute_metrics(
            returns=res.returns,
            nav=res.nav,
            turnover=turnover_s,
            risk_free_rate=risk_free_rate,
        )
    return metrics


# ===================================================================
# Individual chart builders
# ===================================================================
def _nav_chart(results: Dict[str, BacktestResult]):
    """Line chart: normalized NAV over time per strategy."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    fig = go.Figure()
    for label, res in results.items():
        nav_norm = res.nav / res.nav.iloc[0]
        fig.add_trace(go.Scatter(
            x=nav_norm.index,
            y=nav_norm.values,
            mode="lines",
            name=label,
            hovertemplate="%{x|%Y-%m-%d}: %{y:.4f}<extra>%{fullData.name}</extra>",
        ))

    fig.update_layout(
        title="NAV Curves (Normalized to 1.0)",
        xaxis_title="Date",
        yaxis_title="NAV (base = 1)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        template="plotly_white",
        height=450,
    )
    return fig


def _efficient_frontier_chart(
    results: Dict[str, BacktestResult],
    metrics: Dict[str, PortfolioMetrics],
):
    """Scatter: annualized return vs volatility (ex-post efficient frontier)."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    fig = go.Figure()
    for label, m in metrics.items():
        fig.add_trace(go.Scatter(
            x=[m.annualized_volatility],
            y=[m.annualized_return],
            mode="markers+text",
            marker=dict(size=14, symbol="circle"),
            text=[label],
            textposition="top center",
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Vol: %{x:.2%}<br>"
                "Return: %{y:.2%}<br>"
                f"Sharpe: {m.sharpe_ratio:.3f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Ex-Post Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=450,
        showlegend=True,
    )
    return fig


def _weights_heatmap(
    results: Dict[str, BacktestResult],
    strategy: Optional[str] = None,
    top_n: int = 15,
    resample: str = "ME",
):
    """Heatmap: portfolio weights over time for one strategy."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    label = strategy or list(results.keys())[0]
    weights_df = results[label].weights

    if weights_df.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Weights Heatmap — {label} (no data)")
        return fig

    # Normalise legacy aliases (pandas 2.2+ requires ME/QE instead of M/Q)
    _FREQ_MAP = {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE"}
    resample = _FREQ_MAP.get(resample.upper(), resample)

    # Resample to reduce density
    weights_df.index = pd.to_datetime(weights_df.index)
    weights_resampled = weights_df.resample(resample).last().fillna(0.0)

    # Keep top N tickers by mean weight
    mean_weights = weights_resampled.mean()
    top_tickers = mean_weights.nlargest(top_n).index.tolist()
    w_plot = weights_resampled[top_tickers]

    fig = go.Figure(go.Heatmap(
        z=w_plot.values.T,
        x=w_plot.index.strftime("%Y-%m"),
        y=w_plot.columns.tolist(),
        colorscale="Blues",
        colorbar=dict(title="Weight"),
        hovertemplate="Date: %{x}<br>Ticker: %{y}<br>Weight: %{z:.2%}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Portfolio Weights Over Time — {label}",
        xaxis_title="Date",
        yaxis_title="Ticker",
        template="plotly_white",
        height=500,
        xaxis=dict(tickangle=-45),
    )
    return fig


def _turnover_sharpe_scatter(
    metrics: Dict[str, PortfolioMetrics],
):
    """Scatter: avg annual turnover vs Sharpe ratio with CI error bars."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("plotly is required: pip install plotly")

    labels = list(metrics.keys())
    sharpes = [metrics[l].sharpe_ratio for l in labels]
    turnovers = [metrics[l].avg_annual_turnover for l in labels]
    ci_lows = [metrics[l].sharpe_ci_low for l in labels]
    ci_highs = [metrics[l].sharpe_ci_high for l in labels]

    fig = go.Figure()
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[turnovers[i]],
            y=[sharpes[i]],
            mode="markers+text",
            marker=dict(size=14),
            error_y=dict(
                type="data",
                symmetric=False,
                array=[ci_highs[i] - sharpes[i]],
                arrayminus=[sharpes[i] - ci_lows[i]],
            ),
            text=[label],
            textposition="top center",
            name=label,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "Turnover: %{x:.1%}<br>"
                "Sharpe: %{y:.3f}<br>"
                f"CI: [{ci_lows[i]:.3f}, {ci_highs[i]:.3f}]"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Turnover vs Sharpe (with 95% Bootstrap CI)",
        xaxis_title="Avg Annual Turnover",
        yaxis_title="Sharpe Ratio",
        xaxis_tickformat=".0%",
        template="plotly_white",
        height=450,
        showlegend=True,
    )
    return fig


# ===================================================================
# Main dashboard builder
# ===================================================================
def build_dashboard(
    results: Dict[str, BacktestResult],
    output_path: str = "dashboard.html",
    risk_free_rate: float = 0.0,
    heatmap_strategy: Optional[str] = None,
    title: str = "Autoportfolio Backtesting Dashboard",
) -> str:
    """Build a standalone interactive HTML dashboard.

    Sections:
      1. NAV curves (all strategies).
      2. Ex-post efficient frontier.
      3. Weights heatmap (first or specified strategy).
      4. Turnover vs Sharpe scatter with CI error bars.

    Args:
        results: Dict {label → BacktestResult}.
        output_path: Path to output HTML file.
        risk_free_rate: Annual risk-free rate for metrics.
        heatmap_strategy: Strategy to show in heatmap. None → first.
        title: Dashboard title.

    Returns:
        Path to the written HTML file.
    """
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "plotly is required for the dashboard. "
            "Install with: pip install plotly"
        )

    if not results:
        raise ValueError("results dict is empty — nothing to plot.")

    logger.info("Computing metrics for %d strategies...", len(results))
    all_metrics = _compute_all_metrics(results, risk_free_rate)

    # Build individual figures
    fig_nav = _nav_chart(results)
    fig_ef = _efficient_frontier_chart(results, all_metrics)
    fig_hm = _weights_heatmap(results, strategy=heatmap_strategy)
    fig_scatter = _turnover_sharpe_scatter(all_metrics)

    # Metrics summary table
    metrics_rows = []
    for label, m in all_metrics.items():
        metrics_rows.append({
            "Strategy": label,
            "Ann. Return": f"{m.annualized_return:.1%}",
            "Ann. Vol": f"{m.annualized_volatility:.1%}",
            "Sharpe": f"{m.sharpe_ratio:.3f}",
            "Sharpe CI": f"[{m.sharpe_ci_low:.3f}, {m.sharpe_ci_high:.3f}]",
            "Sortino": f"{m.sortino_ratio:.3f}",
            "Max DD": f"{m.max_drawdown:.1%}",
            "Calmar": f"{m.calmar_ratio:.3f}",
            "VaR 95%": f"{m.var_95_historical:.4f}",
            "CVaR 95%": f"{m.cvar_95:.4f}",
            "Beta": f"{m.beta:.3f}",
            "Avg TO": f"{m.avg_annual_turnover:.1%}",
        })
    metrics_df = pd.DataFrame(metrics_rows)

    table_trace = go.Table(
        header=dict(
            values=list(metrics_df.columns),
            fill_color="steelblue",
            font=dict(color="white", size=11),
            align="left",
        ),
        cells=dict(
            values=[metrics_df[c] for c in metrics_df.columns],
            fill_color=[["white", "#f0f4f8"] * (len(metrics_df) // 2 + 1)][0][:len(metrics_df)],
            align="left",
            font=dict(size=10),
        ),
    )
    fig_table = go.Figure(table_trace)
    fig_table.update_layout(
        title="Performance Metrics Summary",
        height=max(200, 60 + len(metrics_df) * 30),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    # Assemble HTML
    html_parts = [
        f"<!DOCTYPE html><html><head>",
        f"<meta charset='utf-8'>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; background: #fafafa; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #34495e; margin-top: 30px; border-bottom: 1px solid #bdc3c7; }",
        ".chart-container { background: white; border-radius: 8px; "
        "box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; padding: 10px; }",
        "</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>Generated by <b>dashboard_results.py</b> | "
        f"Strategies: {', '.join(results.keys())}</p>",
        "<h2>1. NAV Curves</h2>",
        "<div class='chart-container'>",
        fig_nav.to_html(full_html=False, include_plotlyjs="cdn"),
        "</div>",
        "<h2>2. Performance Metrics Summary</h2>",
        "<div class='chart-container'>",
        fig_table.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "<h2>3. Ex-Post Efficient Frontier</h2>",
        "<div class='chart-container'>",
        fig_ef.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "<h2>4. Portfolio Weights Heatmap</h2>",
        "<div class='chart-container'>",
        fig_hm.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "<h2>5. Turnover vs Sharpe</h2>",
        "<div class='chart-container'>",
        fig_scatter.to_html(full_html=False, include_plotlyjs=False),
        "</div>",
        "</body></html>",
    ]

    html_content = "\n".join(html_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Dashboard written to: %s", output_path)
    return output_path


# ===================================================================
# Demo: synthetic data (CLI entry point)
# ===================================================================
def _make_synthetic_results() -> Dict[str, "BacktestResult"]:
    """Generate minimal synthetic BacktestResult objects for demo."""
    from src.backtest_engine import BacktestResult, BacktestConfig, DailyRecord

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2019-01-01", "2023-12-31")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]

    def _make_result(label: str, mu: float, vol: float) -> BacktestResult:
        daily_rets = rng.normal(mu / 252, vol / np.sqrt(252), len(dates))
        nav_vals = 1_000_000 * np.cumprod(1 + daily_rets)
        nav = pd.Series(nav_vals, index=dates, name="nav")
        returns = pd.Series(daily_rets, index=dates, name="returns")

        n = len(dates)
        w_mat = rng.dirichlet(np.ones(len(tickers)), size=n)
        weights = pd.DataFrame(w_mat, index=dates, columns=tickers)

        config = BacktestConfig(opt_method=label)
        daily_records = [
            DailyRecord(
                date=d, nav=float(nav.iloc[i]),
                daily_return=float(daily_rets[i]),
                weights=weights.iloc[i],
                is_rebalance=(i % 63 == 0),
                turnover=0.15 if (i % 63 == 0) else 0.0,
                cost=0.0,
            )
            for i, d in enumerate(dates)
        ]
        return BacktestResult(
            config=config,
            nav=nav,
            returns=returns,
            weights=weights,
            rebalance_log=pd.DataFrame(),
            daily_records=daily_records,
        )

    return {
        "mv_classic":  _make_result("mv_classic",  mu=0.10, vol=0.15),
        "robust_k010": _make_result("robust_k010", mu=0.09, vol=0.13),
        "robust_k025": _make_result("robust_k025", mu=0.08, vol=0.12),
    }


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    synthetic = _make_synthetic_results()
    path = build_dashboard(synthetic, output_path="results/dashboard.html")
    print(f"Dashboard saved to: {path}")
