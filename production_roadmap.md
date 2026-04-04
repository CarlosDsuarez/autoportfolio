# Production Roadmap — Autoportfolio v1.0
**Arquitectura de despliegue, monitoreo y fases futuras**

---

## 1. Arquitectura de Despliegue Recomendada

### Opción A — AWS Lambda + EventBridge (Recomendada)

```
                     ┌─────────────────────────────────────┐
                     │           EventBridge (Cron)         │
                     │  Mensual: 1er día hábil / Trimestral │
                     └──────────────┬──────────────────────┘
                                    ↓
                     ┌─────────────────────────────────────┐
                     │         AWS Lambda                   │
                     │  - run_backtest (modo live)          │
                     │  - filter_universe_at_date           │
                     │  - portfolio_optimizer.optimize      │
                     │  - rebalancing_engine.rebalance      │
                     │  Timeout: 15 min | Memory: 1GB       │
                     └──────┬────────────────┬─────────────┘
                            ↓                ↓
              ┌─────────────────┐   ┌──────────────────────┐
              │   S3 Bucket     │   │   SNS / SES Alerts   │
              │  - weights.csv  │   │  - Drift alerts      │
              │  - nav.csv      │   │  - Solver failures   │
              │  - metrics.json │   │  - Anomaly emails    │
              └─────────────────┘   └──────────────────────┘
                            ↓
              ┌─────────────────────────────────────────────┐
              │           Broker API (Alpaca / IBKR)        │
              │  - execute_order_twap (órdenes TWAP)        │
              │  - Posiciones actuales → w_current           │
              └─────────────────────────────────────────────┘
```

**Ventajas**: sin servidor, pago por uso, escalable, integración nativa con CloudWatch.
**Costo estimado**: < USD 5/mes para frecuencia mensual.

### Opción B — Cron Job en EC2/VPS

```bash
# crontab -e
# Primer día hábil de cada mes a las 08:00 ET
0 8 1-7 * 1 /opt/conda/bin/python /app/run_live_rebalance.py >> /var/log/autoportfolio.log 2>&1
```

**Ventajas**: control total, sin límite de tiempo de ejecución, más barato para alta frecuencia.

---

## 2. Frecuencia de Re-Optimización Recomendada

| Régimen | Frecuencia | Justificación |
|---------|-----------|---------------|
| **Producción base** | Trimestral (Q) | Balance entre costos de TC y captura de señales |
| **Mercado volátil** | Mensual (M) | Si VIX > 30 → drift acelerado |
| **Drift trigger** | Ad-hoc | Si `max_drift > 5%` en cualquier activo → rebalanceo inmediato |
| **Reemplazo de activos** | Trimestral | Junto con re-optimización principal |

**Parámetros recomendados de producción**:
```python
ProductionConfig = BacktestConfig(
    opt_method="robust",          # Más estable ante incertidumbre
    rebalance_freq="Q",
    kappa=0.10,
    max_weight=0.25,              # Ligeramente más conservador
    turnover_limit=0.15,          # Límite más estricto en producción
    commission_bps=7.0,           # Incluir slippage real
    spread_bps=3.0,
    replacement_rule="momentum",
)
```

---

## 3. Sistema de Alertas de Anomalías

### 3.1 Alertas de Drift
```python
# Configurar en drift_monitor.py
DRIFT_ALERT_THRESHOLD = 0.05    # 5% drift en un activo → email
TOTAL_DRIFT_ALERT = 0.15        # 15% drift total → rebalanceo urgente
```

| Trigger | Acción | Canal |
|---------|--------|-------|
| `max_drift > 5%` | Log + email de advertencia | SNS → Email |
| `total_drift > 15%` | Rebalanceo inmediato | SNS → Lambda trigger |
| `max_drift > 10%` | Alerta crítica + revisión manual | PagerDuty / Slack |

### 3.2 Alertas de Convergencia del Solver
```python
# En portfolio_optimizer.py
if not result["converged"]:
    # Fallback: equal-weight entre activos activos
    send_alert("SOLVER_FAIL", ticker_list, date)
```

| Trigger | Acción |
|---------|--------|
| Solver no converge | Fallback a equal-weight + alerta |
| `solve_time > 60s` | Log warning (posible degradación) |
| 3 fallos consecutivos | Alerta crítica + revisión manual |

### 3.3 Alertas de Datos
| Trigger | Acción |
|---------|--------|
| `NaN rate > 5%` en precios | Excluir ticker + notificación |
| Precio cero o negativo | Excluir ticker de universo |
| yfinance timeout | Retry 3x con backoff exponencial |

### 3.4 Alertas de Métricas en Tiempo Real

```python
# Monitorear métricas rolling cada semana
ROLLING_SHARPE_ALERT = 0.0      # Sharpe 90d < 0 → revisión
MAX_DD_ALERT = -0.15            # MaxDD > 15% → alerta
```

---

## 4. Plan de Monitoreo Continuo

### Dashboard en producción (CloudWatch / Grafana)

| Métrica | Frecuencia | Umbral Alerta |
|---------|-----------|---------------|
| NAV diario | Diaria | Caída > 2% en un día |
| Sharpe rolling 90d | Semanal | < 0.0 |
| MaxDD rolling | Semanal | > -15% |
| Turnover por rebalanceo | Por rebalanceo | > 25% |
| N activos en universo | Por rebalanceo | < 5 |
| Solver convergencia | Por rebalanceo | Cualquier fallo |
| Calidad de datos (NaN%) | Diaria | > 2% |

### Reportes automáticos

```
Semanal:  Reporte de performance rolling (Sharpe, MaxDD, VaR)
Mensual:  Comparación vs benchmark (S&P500, 60/40)
Trimestral: Análisis completo de atribución + IC bootstrap
Anual:    Revisión de hiperparámetros y universo de activos
```

---

## 5. Cronograma Fases Futuras

### Fase 2 — Mejoras de Modelo (Meses 3–6)

| Tarea | Descripción | Prioridad |
|-------|-------------|-----------|
| Factor model returns | Fama-French 5 factores para μ | Alta |
| DCC-GARCH covariance | Covarianza dinámica condicional | Alta |
| Black-Litterman completo | Views de analistas + equilibrio de mercado | Media |
| Regime detection | HMM para detección de régimen Bull/Bear | Media |
| ESG constraints | Filtro por puntuación ESG | Baja |

### Fase 3 — Infraestructura Avanzada (Meses 6–12)

| Tarea | Descripción | Prioridad |
|-------|-------------|-----------|
| Broker integration | Alpaca / IBKR API real para ejecución | Alta |
| Multi-portfolio | Gestión de múltiples portafolios simultáneos | Media |
| Risk management layer | Stop-loss dinámico + circuit breakers | Alta |
| Reporting automático | PDF/HTML mensual automático vía email | Media |
| A/B testing framework | Comparar configuraciones en vivo | Baja |
| ML signal layer | Señales de ML para ajuste de μ | Baja |

---

## 6. Requisitos de Infraestructura en Producción

### Mínimos
- Python 3.10+ con todas las dependencias de `requirements.txt`
- Acceso a internet para yfinance (o feed de datos alternativo)
- 2 GB RAM, 10 GB disco (histórico + logs)
- Cuenta broker con API (Alpaca paper trading para pruebas)

### Recomendados
- AWS Lambda con layer de dependencias pre-empaquetado
- S3 para storage de pesos, NAV y logs
- CloudWatch para métricas y alertas
- Secrets Manager para credenciales del broker

### Variables de entorno necesarias
```bash
BROKER_API_KEY=xxx
BROKER_SECRET_KEY=xxx
BROKER_BASE_URL=https://paper-api.alpaca.markets  # paper trading
AWS_REGION=us-east-1
S3_BUCKET=autoportfolio-results
ALERT_EMAIL=pm@company.com
PORTFOLIO_VALUE=1000000  # USD
```

---

## 7. SLA y Objetivos de Servicio

| Métrica | Objetivo |
|---------|----------|
| Disponibilidad del pipeline | 99.5% en días de rebalanceo |
| Latencia de re-optimización | < 5 min (trimestral) |
| Latencia de ejecución de órdenes | < 30 min (TWAP 10 slices) |
| Tiempo de recuperación ante fallo | < 1 hora (fallback equal-weight) |
| Retención de datos históricos | 5 años rolling |

---

*Documento generado: Autoportfolio v1.0 — Universidad Icesi*
