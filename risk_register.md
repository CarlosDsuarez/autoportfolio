# Risk Register — Autoportfolio v1.0
**Riesgos Técnicos y Éticos | Mitigaciones Definidas**

---

## Formato del Registro

| Campo | Descripción |
|-------|-------------|
| **ID** | Identificador único (RT = Técnico, RE = Ético) |
| **Riesgo** | Descripción del riesgo |
| **Probabilidad** | Alta / Media / Baja |
| **Impacto** | Alto / Medio / Bajo |
| **Severidad** | P × I (1–9 escala) |
| **Control implementado** | Mitigación ya en el código |
| **Mitigación adicional** | Acciones recomendadas |
| **Estado** | Activo / Monitorear / Cerrado |

---

## Riesgos Técnicos (RT)

### RT-01 — Overfitting en la optimización

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Alta |
| **Impacto** | Alto |
| **Severidad** | 9/9 |
| **Descripción** | El optimizador puede aprender ruido histórico como señal, produciendo pesos que se desempeñan mal OOS. |
| **Control implementado** | ✅ Evaluación estrictamente OOS (R-05): `rolling_windows` con ventanas separadas train/test. ✅ Shrinkage Ledoit-Wolf en covarianza (R-09). ✅ Bootstrap IC 95% para significancia estadística (R-06). ✅ `max_weight=30%` limita concentración extrema (R-04). |
| **Mitigación adicional** | En producción: no re-optimizar hiperparámetros (kappa, window) con datos en-sample. Revisión anual de parámetros con nuevos datos. |
| **Estado** | Monitorear |

---

### RT-02 — Sesgo de selección del universo (Survivorship Bias)

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Media |
| **Impacto** | Alto |
| **Severidad** | 6/9 |
| **Descripción** | yfinance descarga solo tickers actuales. Activos que quebraron o fueron deslistados no aparecen, sobreestimando retornos esperados. |
| **Control implementado** | ✅ Filtro Amihud excluye activos ilíquidos (proxy para empresas en dificultades). ✅ `clean_prices` marca gaps excesivos. |
| **Mitigación adicional** | Para producción: usar feed de datos con datos point-in-time (Bloomberg, Compustat). Documentar el universo fijo al inicio del backtest. Agregar tickers históricos de S&P500 que luego salieron del índice. |
| **Estado** | Activo — riesgo conocido y documentado |

---

### RT-03 — Cambios de Régimen de Mercado

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Media |
| **Impacto** | Alto |
| **Severidad** | 6/9 |
| **Descripción** | Los parámetros estimados en un régimen (bull) se desempeñan mal en otro (crash, bear). La matriz de covarianza histórica subestima correlaciones en crisis. |
| **Control implementado** | ✅ Optimizador robusto con parámetro `kappa` (R-09): protege contra incertidumbre elipsoidal en μ. ✅ Ledoit-Wolf regulariza la covarianza. ✅ Experimento 4 analiza rendimiento por régimen. |
| **Mitigación adicional** | En producción: monitorear VIX > 30 como señal de régimen. Activar `kappa=0.25` automáticamente. Implementar detección de régimen vía HMM en Fase 2. |
| **Estado** | Monitorear |

---

### RT-04 — No Convergencia del Solver CVXPY/ECOS

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Baja |
| **Impacto** | Alto |
| **Severidad** | 3/9 |
| **Descripción** | En universos mal condicionados (covarianzas casi singulares, activos muy correlacionados) el solver puede no converger o devolver soluciones inestables. |
| **Control implementado** | ✅ Detección de estado: `result["converged"]` y `result["status"]`. ✅ Eigenvalue diagnostics en `covariance_estimators`. ✅ Ledoit-Wolf garantiza covarianza PSD. ✅ Fallback documentado en `portfolio_optimizer`. |
| **Mitigación adicional** | Implementar fallback automático a equal-weight cuando solver falla. Loguear todos los fallos a CloudWatch. Alertar PM si falla > 2 veces consecutivas. |
| **Estado** | Monitorear |

---

### RT-05 — Calidad de Datos de yfinance

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Media |
| **Impacto** | Medio |
| **Severidad** | 4/9 |
| **Descripción** | yfinance puede devolver datos erróneos (splits no ajustados, gaps, precios cero). Datos corruptos contaminan la estimación de μ y Σ. |
| **Control implementado** | ✅ `clean_prices`: forward-fill máx. 5 días, elimina columnas con >20% NaN. ✅ `filter_universe_at_date`: excluye activos con volumen < 100k USD. ✅ Tests de calidad en `test_data_pipeline.py`. |
| **Mitigación adicional** | En producción: validar precios contra fuente secundaria (Quandl, Alpha Vantage). Alertar si precio cambia > 20% en un día. Implementar detección de outliers. |
| **Estado** | Activo |

---

### RT-06 — Slippage y Costos Subestimados

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Media |
| **Impacto** | Medio |
| **Severidad** | 4/9 |
| **Descripción** | El modelo TWAP simplifica market impact. En activos ilíquidos o en periodos de stress, el impacto real puede ser 2-5x el modelado. |
| **Control implementado** | ✅ TWAP reduce impact por √N vs bloque. ✅ Restricción de liquidez limita órdenes a % del ADV. ✅ Experimento 2 mide sensibilidad a niveles de costo. |
| **Mitigación adicional** | En producción: calibrar `impact_coef` mensualmente contra datos reales de ejecución. Usar `commission_bps=10` como estimación conservadora. |
| **Estado** | Monitorear |

---

### RT-07 — Concentración de Riesgo

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Baja |
| **Impacto** | Alto |
| **Severidad** | 3/9 |
| **Descripción** | El optimizador MV puede concentrar la cartera en pocos activos si el universo tiene activos muy correlacionados o si μ tiene alta dispersión. |
| **Control implementado** | ✅ `max_weight=30%` por activo (R-04). ✅ Restricción de liquidez limita peso por ADV. ✅ Robustez reduce sensibilidad a μ extremos. |
| **Mitigación adicional** | Agregar restricción de número mínimo de activos: N_active >= 5. Monitorear HHI (Herfindahl-Hirschman Index) de la cartera. |
| **Estado** | Monitorear |

---

## Riesgos Éticos (RE)

### RE-01 — Impacto Sistémico de Estrategias Correlacionadas

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Baja |
| **Impacto** | Alto |
| **Severidad** | 3/9 |
| **Descripción** | Si muchos actores usan estrategias MV similares, los rebalanceos simultáneos pueden amplificar movimientos del mercado (flash crashes). |
| **Control implementado** | ✅ TWAP fragmenta órdenes (reduce impacto instantáneo). ✅ Restricción de liquidez limita órdenes a fracción del ADV. |
| **Mitigación adicional** | Monitorear volumen de órdenes como % del ADV del mercado. En producción con AUM > USD 10M: diversificar horarios de ejecución. Documentar límites de capacidad de la estrategia. |
| **Estado** | Monitorear |

---

### RE-02 — Cumplimiento Regulatorio de Trading Algorítmico

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Media (si se monetiza) |
| **Impacto** | Alto |
| **Severidad** | 6/9 |
| **Descripción** | En Colombia y EE.UU., el trading algorítmico con dinero de terceros requiere registro como asesor de inversiones (RIA en EE.UU., SFC en Colombia). Uso de datos de mercado puede requerir licencias. |
| **Control implementado** | ⚠️ Proyecto académico — sin gestión de dinero de terceros en v1.0. |
| **Mitigación adicional** | Antes de comercializar: consultar abogado especializado en regulación financiera. En EE.UU.: SEC RIA registration si AUM > USD 110M. En Colombia: registro ante SFC. yfinance para uso personal/educativo — para producción usar Bloomberg o Refinitiv. |
| **Estado** | Activo — documentado, pendiente revisión legal antes de producción |

---

### RE-03 — Transparencia y Explicabilidad

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Baja |
| **Impacto** | Medio |
| **Severidad** | 2/9 |
| **Descripción** | El proceso de optimización puede ser difícil de explicar a clientes finales ("black box"), generando desconfianza o problemas regulatorios de suitability. |
| **Control implementado** | ✅ Todos los parámetros configurables y documentados. ✅ `rebalance_log` registra cada decisión con motivo. ✅ Dashboard transparente con atribución de pesos. |
| **Mitigación adicional** | Preparar resumen en lenguaje no técnico de cada rebalanceo. Documentar en el informe técnico qué variables impulsan cada decisión. |
| **Estado** | Monitorear |

---

### RE-04 — Sesgo en Selección de Universo (Ethical)

| Campo | Detalle |
|-------|---------|
| **Probabilidad** | Baja |
| **Impacto** | Bajo |
| **Severidad** | 1/9 |
| **Descripción** | El universo de activos puede excluir sectores o geografías sistémicamente, generando concentración en sectores de alto rendimiento histórico (ej. tech). |
| **Control implementado** | ✅ Filtro por liquidez es objetivo (Amihud). ✅ Momento de reemplazo considera volatilidad además de retorno. |
| **Mitigación adicional** | En producción: diversificar universo por sectores GICS. Considerar restricciones de sector (ej. max 30% tech). Integrar scoring ESG en Fase 2. |
| **Estado** | Monitorear |

---

## Resumen Ejecutivo de Riesgos

| ID | Riesgo | Probabilidad | Impacto | Severidad | Estado |
|----|--------|-------------|---------|-----------|--------|
| RT-01 | Overfitting | Alta | Alto | 9/9 | Monitorear |
| RT-02 | Survivorship Bias | Media | Alto | 6/9 | **Activo** |
| RT-03 | Cambio de régimen | Media | Alto | 6/9 | Monitorear |
| RT-04 | No convergencia solver | Baja | Alto | 3/9 | Monitorear |
| RT-05 | Calidad datos yfinance | Media | Medio | 4/9 | **Activo** |
| RT-06 | Slippage subestimado | Media | Medio | 4/9 | Monitorear |
| RT-07 | Concentración riesgo | Baja | Alto | 3/9 | Monitorear |
| RE-01 | Impacto sistémico | Baja | Alto | 3/9 | Monitorear |
| RE-02 | Regulación trading algo | Media | Alto | 6/9 | **Activo** |
| RE-03 | Transparencia | Baja | Medio | 2/9 | Monitorear |
| RE-04 | Sesgo universo | Baja | Bajo | 1/9 | Monitorear |

**Riesgos bloqueantes para producción**: RT-02 (datos point-in-time), RE-02 (regulación)

---

## Revisión y Actualización

| Fecha | Autor | Cambios |
|-------|-------|---------|
| 2026-04-01 | Equipo Técnico | Registro inicial v1.0 |

*Próxima revisión recomendada: antes de go-live en producción*
