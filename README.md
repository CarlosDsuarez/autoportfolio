# Autoportfolio — Sistema Automatizado de Optimización de Portafolios v1.0

Sistema de optimización y re-optimización de portafolios de acciones y ETFs
construido en Python. Implementa optimización media-varianza clásica y robusta,
restricciones de liquidez/turnover/costos de transacción, backtesting rolling
OOS, reglas de reemplazo de activos y métricas de rendimiento con IC bootstrap.

---

## Tabla de Contenidos

1. [Arquitectura](#arquitectura)
2. [Instalación](#instalación)
3. [Inicio Rápido](#inicio-rápido)
4. [Módulos](#módulos)
5. [Configuración](#configuración)
6. [Experimentos](#experimentos)
7. [Dashboard](#dashboard)
8. [Tests](#tests)
9. [Restricciones Implementadas (R-01 a R-10)](#restricciones)

---

## Arquitectura

```
CM Project/
├── universe.csv                  # Universo canónico: 100 principales S&P 500
├── criterios.md                  # Criterios de selección del universo
├── src/                          # Código fuente principal
│   ├── data_client.py            # Fase 1: descarga y limpieza de datos
│   ├── rolling_engine.py         # Fase 2: ventanas rolling OOS
│   ├── expected_returns.py       # Fase 2: estimación de retornos esperados
│   ├── covariance_estimators.py  # Fase 2: estimación de covarianza
│   ├── universe_filter.py        # Fase 2: filtro de universo por liquidez
│   ├── mv_optimizer.py           # Fase 3: optimizador MV clásico (SOCP)
│   ├── robust_optimizer.py       # Fase 3: optimizador robusto (Charnes-Cooper)
│   ├── optimizer_with_tc.py      # Fase 3: optimizador con costos de TC
│   ├── turnover_constraint.py    # Fase 3: restricción de turnover
│   ├── liquidity_constraint.py   # Fase 3: restricción de liquidez Amihud
│   ├── portfolio_optimizer.py    # Fase 3: interfaz unificada
│   ├── drift_monitor.py          # Fase 4: monitor de drift de pesos
│   ├── momentum_replacement.py   # Fase 4: reemplazo por momentum 12-1
│   ├── random_replacement.py     # Fase 4: reemplazo aleatorio reproducible
│   ├── min_variance_replacement.py # Fase 4: reemplazo por mínima varianza
│   ├── order_execution.py        # Fase 4: ejecución TWAP
│   ├── rebalancing_engine.py     # Fase 4: motor de rebalanceo unificado
│   ├── backtest_engine.py        # Fase 5: backtest rolling OOS (+ Fase 6)
│   ├── metrics_calculator.py     # Fase 5: métricas (Sharpe, MaxDD, VaR, ...)
│   ├── dashboard_results.py      # Fase 5: dashboard interactivo HTML/Plotly
│   ├── position_ledger.py        # Fase 6: libro de posiciones FIFO (sin broker)
│   ├── regime_detector.py        # Fase 6: régimen de mercado (4 estados)
│   ├── signal_stack.py           # Fase 6: motor multi-factor (mom/MR/vol/liq)
│   ├── hrp_sizing.py             # Fase 6: Hierarchical Risk Parity
│   ├── conviction_scoring.py     # Fase 6: conviction rule-based (sin LLM)
│   └── tail_risk_overlay.py      # Fase 6: interfaz Tail Risk (bloqueada por IV)
├── tests/                        # Suite de pruebas (238+ tests)
│   ├── test_data_pipeline.py
│   ├── test_phase2_estimation.py
│   ├── test_optimizer.py
│   ├── test_rebalancing.py
│   ├── test_backtest.py
│   ├── test_position_ledger.py
│   ├── test_regime_detector.py
│   ├── test_signal_stack.py
│   ├── test_hrp_sizing.py
│   ├── test_conviction_scoring.py
│   ├── test_tail_risk_overlay.py
│   └── test_universe.py
├── experiments/                  # Scripts de experimentos
│   ├── experiment_1_mv_vs_robust.py
│   ├── experiment_2_transaction_costs.py
│   ├── experiment_3_replacement_rules.py
│   └── experiment_4_regime_analysis.py
├── results/                      # Outputs (CSVs, PNGs, HTML dashboard)
├── portfolio_optimizer_notebook.ipynb  # Notebook plug-and-play (Colab)
├── validation_checklist.md       # Checklist R-01 a R-10
├── production_roadmap.md         # Plan de despliegue en producción
├── risk_register.md              # Registro de riesgos técnicos y éticos
└── requirements.txt              # Dependencias con versiones fijas
```

### Flujo de datos

```
yfinance → data_client → universe_filter
                              ↓
             rolling_engine (ventanas OOS, freq M/Q/15D)
                              ↓
         expected_returns + covariance_estimators
                              ↓
   ┌──────────── portfolio_optimizer ────────────┐
   │  mv_classic / robust / mv_tc / turnover / liq │
   └──────────────────┬─────── hrp_sizing ────────┘
                      ↓
         [Fase 6 opcional] regime_detector
                      ↓
                  signal_stack → conviction_scoring
                      ↓
              rebalancing_engine  OR  position_ledger (FIFO)
                      ↓
                    backtest_engine (NAV diario + snapshots)
                      ↓
                  metrics_calculator + dashboard
```

### Fase 6 — Motor de Señales + Simulación Live (sin broker)

Simulación interna estilo live trading **sin conexión a broker**:
- `PositionLedger`: pesos → shares, costeo FIFO, snapshots auditables.
- `get_rebalance_dates(freq="15D")`: cada N **días hábiles** (no calendario).
- Régimen 4 estados + multi-factor (momentum, mean-reversion, vol, liquidez).
- Conviction **rule-based** en backtest (sin LLM → evita look-ahead).
- HRP como `opt_method="hrp"` comparable con MVO/robusto.
- `TailRiskOverlay`: solo interfaz; requiere IV surface (CBOE/ORATS/Polygon) no disponible en `data_client`.

```python
cfg = BacktestConfig(
    opt_method="hrp",
    rebalance_freq="15D",
    use_signal_stack=True,
    use_position_ledger=True,
    regime_window=63,
)
result = run_backtest(prices, volumes, cfg)
# result.ledger_snapshots  → List[LedgerSnapshot]
# result.regime_series     → pd.Series de estados por fecha de rebalanceo
```

---

## Instalación

### Requisitos del sistema
- Python 3.10+ (en Ubuntu/Debian: `python3` y `python3-venv`)
- **No** uses `pip install` sobre el Python del sistema (PEP 668)

### Clonar e instalar (venv)

```bash
git clone <repo-url>
cd autoportfolio   # o el nombre de tu carpeta local

# Crear entorno virtual (evita "externally-managed-environment")
sudo apt install -y python3-venv   # solo si falla: ensurepip is not available
python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt

# Descargar y limpiar los 100 tickers → data/
python -m src.data_client
```

En cada terminal nueva:

```bash
cd autoportfolio
source .venv/bin/activate
```

---

## Inicio Rápido

### Backtest mínimo

```python
from src.data_client import load_universe, fetch_prices, fetch_volumes, clean_prices
from src.backtest_engine import BacktestConfig, run_backtest
from src.metrics_calculator import compute_metrics

# 1. Universo canónico: 100 principales del S&P 500 (universe.csv)
tickers = load_universe()["ticker"].tolist()
prices_raw = fetch_prices(tickers, start="2019-01-01", end="2023-12-31")
volumes_raw = fetch_volumes(tickers, start="2019-01-01", end="2023-12-31")
prices, _ = clean_prices(prices_raw)
volumes = volumes_raw.reindex(prices.index).ffill().fillna(1e6)

# 2. Configurar backtest
config = BacktestConfig(
    opt_method="mv_classic",   # "mv_classic" | "robust" | "mv_tc" | ...
    rebalance_freq="Q",        # "M" | "Q"
    window=252,                # días de entrenamiento
    warmup=252,                # días de warmup inicial
    max_weight=0.30,           # máximo peso por activo
    turnover_limit=0.20,       # límite de turnover por rebalanceo
    seed=42,                   # reproducibilidad
)

# 3. Ejecutar backtest
result = run_backtest(prices, volumes, config)

# 4. Calcular métricas
import pandas as pd
turnover_s = pd.Series({r.date: r.turnover for r in result.daily_records})
metrics = compute_metrics(result.returns, result.nav, turnover=turnover_s)

print(f"Sharpe:  {metrics.sharpe_ratio:.3f}  [{metrics.sharpe_ci_low:.3f}, {metrics.sharpe_ci_high:.3f}]")
print(f"MaxDD:   {metrics.max_drawdown:.1%}")
print(f"Ann.Ret: {metrics.annualized_return:.1%}")
```

### Optimización puntual

```python
import numpy as np
from src.portfolio_optimizer import optimize

mu = np.array([0.10, 0.08, 0.12, 0.07, 0.09])
sigma = np.eye(5) * 0.04 + 0.01

result = optimize(mu, sigma, method="mv_classic", max_weight=0.30)
print(result["weights"])      # pd.Series con pesos
print(result["expected_sharpe"])
```

### Comparar múltiples configuraciones

```python
from src.backtest_engine import run_comparison, BacktestConfig

configs = {
    "mv_classic":  BacktestConfig(opt_method="mv_classic"),
    "robust_k010": BacktestConfig(opt_method="robust", kappa=0.10),
    "mv_tc":       BacktestConfig(opt_method="mv_tc", tc_lambda=0.001),
}
results = run_comparison(prices, volumes, configs)
```

---

## Módulos

### `src/portfolio_optimizer.py` — Interfaz Unificada

```python
optimize(mu, sigma, method="mv_classic", w_prev=None,
         max_weight=0.30, turnover_limit=0.20, tc_lambda=0.001,
         kappa=0.10, n_obs=252, risk_free_rate=0.0, ...)
# → {"weights", "expected_sharpe", "turnover", "cost",
#    "converged", "solve_time", "method", "status"}
```

**Métodos disponibles:**

| `method` | Descripción |
|----------|-------------|
| `mv_classic` | MV clásico vía Charnes-Cooper SOCP |
| `robust` | MV robusto con incertidumbre elipsoidal |
| `mv_tc` | MV con penalización/presupuesto de TC |
| `robust_tc` | Robusto + TC |
| `mv_turnover` | MV con restricción de turnover hard/soft |
| `liquidity` | MV con exclusión Amihud + bounds por ADV |

### `src/backtest_engine.py` — Backtest Engine

```python
@dataclass
class BacktestConfig:
    opt_method: str = "mv_classic"
    rebalance_freq: str = "M"         # "M" | "Q"
    window: int = 252                 # días ventana entrenamiento
    warmup: int = 252                 # días warmup inicial
    mu_method: str = "historical"     # "historical" | "capm_shrunk" | "black_litterman"
    cov_method: str = "ledoit_wolf"   # "sample" | "ledoit_wolf" | "pca"
    replacement_rule: str = "none"    # "none" | "momentum" | "random" | "min_variance"
    max_weight: float = 0.30
    turnover_limit: float = 0.20
    tc_lambda: float = 0.001
    kappa: float = 0.10
    initial_nav: float = 1_000_000.0
    commission_bps: float = 5.0
    spread_bps: float = 2.0
    impact_coef: float = 0.1
    seed: int = 42

result = run_backtest(prices, volumes, config)
# result.nav        → pd.Series (NAV diario)
# result.returns    → pd.Series (retornos diarios)
# result.weights    → pd.DataFrame (pesos por fecha)
# result.rebalance_log → pd.DataFrame (log de cada rebalanceo)
```

### `src/metrics_calculator.py` — Métricas

```python
metrics = compute_metrics(
    returns,              # pd.Series de retornos diarios
    nav,                  # pd.Series de NAV
    turnover=None,        # pd.Series de turnover (opcional)
    benchmark_returns=None, # pd.Series benchmark (opcional)
    risk_free_rate=0.0,
    n_bootstrap=1000,     # iteraciones bootstrap CI (R-06)
    seed=42,
)
# metrics.sharpe_ratio, .sortino_ratio, .max_drawdown, .calmar_ratio
# metrics.var_95_historical, .var_95_parametric, .cvar_95
# metrics.beta, .alpha, .information_ratio
# metrics.sharpe_ci_low, .sharpe_ci_high  ← IC 95% bootstrap
```

---

## Configuración

### Parámetros clave de `BacktestConfig`

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `opt_method` | `"mv_classic"` | Método: MVO/robusto/... o `"hrp"` (Fase 6) |
| `rebalance_freq` | `"M"` | `"M"`/`"Q"`/`"ME"`/`"QE"` o `"15D"` (días hábiles) |
| `window` | `252` | Días en ventana de entrenamiento |
| `warmup` | `252` | Días de warmup antes del primer rebalanceo |
| `mu_method` | `"historical"` | Estimador de retornos esperados |
| `cov_method` | `"ledoit_wolf"` | Estimador de covarianza |
| `replacement_rule` | `"none"` | Regla de reemplazo de activos |
| `max_weight` | `0.30` | Peso máximo por activo (30%) |
| `turnover_limit` | `0.20` | Límite de turnover por rebalanceo (20%) |
| `use_signal_stack` | `False` | Fase 6: régimen + señales + conviction |
| `use_position_ledger` | `False` | Fase 6: simulación en shares (FIFO) |
| `regime_window` | `63` | Fase 6: ventana del clasificador de régimen |
| `kappa` | `0.10` | Nivel de robustez (incertidumbre elipsoidal) |
| `initial_nav` | `1_000_000` | NAV inicial en USD |
| `commission_bps` | `5.0` | Comisión en bps |
| `spread_bps` | `2.0` | Bid-ask spread en bps |
| `impact_coef` | `0.1` | Coeficiente de market impact |
| `seed` | `42` | Semilla RNG para reproducibilidad |

---

## Experimentos

Ejecutar desde la raíz del proyecto:

```bash
# Experimento 1: MV clásico vs robusto
python -m experiments.experiment_1_mv_vs_robust

# Experimento 2: Impacto de costos de transacción
python -m experiments.experiment_2_transaction_costs

# Experimento 3: Reglas de reemplazo de activos
python -m experiments.experiment_3_replacement_rules

# Experimento 4: Análisis por régimen de mercado
python -m experiments.experiment_4_regime_analysis
```

Los resultados se guardan en `results/` como CSVs y PNGs.

---

## Dashboard

```python
from src.backtest_engine import run_comparison, BacktestConfig
from src.dashboard_results import build_dashboard

configs = {
    "mv_classic": BacktestConfig(opt_method="mv_classic"),
    "robust":     BacktestConfig(opt_method="robust", kappa=0.10),
}
results = run_comparison(prices, volumes, configs)
build_dashboard(results, output_path="results/dashboard.html")
```

O con datos sintéticos:
```bash
python -m src.dashboard_results
# Genera: results/dashboard.html
```

El dashboard incluye:
1. **NAV curves** — comparación normalizada de estrategias
2. **Tabla de métricas** — Sharpe, Sortino, MaxDD, Beta, IR, etc.
3. **Frontera eficiente ex-post** — retorno vs volatilidad anualizada
4. **Heatmap de pesos** — evolución de allocations en el tiempo
5. **Turnover vs Sharpe** — scatter con barras de error del IC 95%

---

## Tests

```bash
# Suite completa
python -m pytest tests/ -v

# Por módulo
python -m pytest tests/test_optimizer.py -v
python -m pytest tests/test_rebalancing.py -v
python -m pytest tests/test_backtest.py -v

# Con reporte de cobertura
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=html
```

**Estado actual**: 238+ passing (excl. `test_data_pipeline` network-dependent) ✅

---

## Restricciones

| ID | Descripción | Módulo(s) |
|----|-------------|-----------|
| R-01 | Costos de transacción explícitos (comisión + spread + market impact TWAP) | `optimizer_with_tc`, `order_execution` |
| R-02 | Restricción de turnover (hard cap + blending) | `turnover_constraint`, `rebalancing_engine` |
| R-03 | Restricción de liquidez Amihud + bounds por ADV | `liquidity_constraint`, `universe_filter` |
| R-04 | Pesos ≥ 0, suma = 1, peso máx. 30% | Todos los optimizadores |
| R-05 | Sin look-ahead — rolling OOS estricto | `rolling_engine`, `backtest_engine` |
| R-06 | Bootstrap IC 95% Sharpe (1000 iter.) | `metrics_calculator` |
| R-07 | Reproducibilidad total (`seed=42`) | `random_replacement`, `metrics_calculator` |
| R-08 | Métricas completas (Sharpe, Sortino, MaxDD, Calmar, VaR, CVaR, Beta, IR) | `metrics_calculator` |
| R-09 | Sin overfitting (Ledoit-Wolf, OOS, IC bootstrap) | `covariance_estimators`, `backtest_engine` |
| R-10 | Modularidad y reproducibilidad del pipeline | Toda la arquitectura |

---

## Licencia
Carlos D. Suarez www.linkedin.com/in/carlos-david-suarez-data-corporate-finance-bi-business
Proyecto académico — Universidad Icesi. Uso interno.
