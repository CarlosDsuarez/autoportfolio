# Criterios de Selección del Universo de Activos

## Fecha de definición: 2025-01-02 (inicio del periodo de backtest)

## Criterios de inclusión

### Acciones (mínimo 30)
- **Market cap**: >= USD 10B (large cap) al momento de selección
- **Volumen 30d promedio**: >= 1M acciones/día
- **Diversificación sectorial**: mínimo 8 de 11 sectores GICS representados
- **Historial**: >= 5 años de datos de precios ajustados disponibles
- **Listado activo**: cotizando en NYSE o NASDAQ al inicio del periodo

### ETFs (mínimo 10)
- **AUM**: >= USD 1B
- **Volumen 30d promedio**: >= 500K acciones/día
- **Diversificación**: cubrir equity (US, intl), renta fija, commodities, real estate
- **Historial**: >= 5 años de datos disponibles
- **Spread bid-ask**: <= 0.10% promedio

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
