# Criterios de Selección del Universo de Activos

## Fecha de definición: 2025-01-02 (inicio del periodo de backtest)

## Universo actual

- **Fuente**: [`universe.csv`](universe.csv)
- **Tamaño**: **100** acciones de los principales componentes del S&P 500
  (aprox. mega/large-caps por capitalización y liquidez)
- **Cobertura**: 11 sectores GICS
- **Sin ETFs** en esta revisión (el universo es equity S&P 500 puro)

## Criterios de inclusión

### Acciones (objetivo: 100)
- **Universo base**: componentes principales del S&P 500
- **Market cap**: large / mega cap
- **Volumen 30d promedio**: >= 1M acciones/día (filtro adicional R-03 en runtime)
- **Diversificación sectorial**: los 11 sectores GICS representados
- **Historial**: >= 5 años de datos de precios ajustados disponibles (ideal)
- **Listado activo**: cotizando en NYSE o NASDAQ

### ETFs (opcional / legacy)
- Versiones anteriores del universo incluían >= 10 ETFs de equity, renta fija
  y commodities. La revisión a 100 tickers S&P 500 prioriza acciones.

## Mitigación de survivorship bias
- Universo definido al inicio del periodo (point-in-time)
- No se seleccionan activos basándose en desempeño futuro
- Si un activo es delistado durante el backtest, se marca como excluido
  en la ventana correspondiente (no se elimina retroactivamente)
- Documentar cualquier activo que salga del universo y la razón

## Notas
- El universo se actualiza solo en las ventanas de reemplazo (trimestral)
- Restricción R-03 (liquidez) aplica filtro dinámico adicional por ventana
- Cap superior por activo: 30% (R-04, configurable)
- Cargar con: `load_universe()` → `universe["ticker"].tolist()`
