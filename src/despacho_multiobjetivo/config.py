"""Rutas del proyecto y escenario base de las corridas.

Todas las rutas parten de la carpeta del proyecto (MODELOS), así los scripts funcionan
sin importar desde qué carpeta se ejecuten.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]          # src/despacho_multiobjetivo/config.py → MODELOS
RAW = RAIZ / "data" / "raw"                         # datos de la empresa
PROCESSED = RAIZ / "data" / "processed"             # resultados y cachés que genera el proyecto
EXTERNAL = RAIZ / "data" / "external"               # parámetros externos (tarifas, capacidades, regiones)

# Escenario base de la tesis: Mosquera, enero de 2026, contra lo pagado, espera máxima de 3 días
SEDE = "0111"
MES_INICIO = "2026-01-01"
MES_FIN = "2026-01-31"
MODO = "historico"   # la tesis mide contra lo pagado; en uso real es "operacion" (por defecto en scripts/main.py)
ESPERA_MAXIMA = 3

# Parámetros del solver (HiGHS) según el uso
GAP_REL_MENSUAL = 0                # Etapa 1: es la referencia y resuelve en segundos, se exige el óptimo exacto
GAP_REL_FRONTERA = 0.001           # Etapa 3: cada punto a menos de 0,1 % del costo
LIMITE_SEGUNDOS_FRONTERA = 600     # tiempo máximo por punto de la frontera
GAP_REL_VALIDACION = 0.001         # Etapa 2 en validar_ventanas_tiempo.py
LIMITE_SEGUNDOS_VALIDACION = 1800  # tiempo máximo por espera en validar_ventanas_tiempo.py

# Prefijo de los archivos de la frontera del escenario base (matriz, puntos, puntos a mano y gráfica)
PREFIJO_FRONTERA = str(PROCESSED / f"frontera_{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}")
