"""Arma las series diarias del histórico de una sede (por defecto 0111) y las guarda en data/processed/:
series_diarias_<sede>_total.csv (una fila por día) y series_diarias_<sede>_region.csv (una fila por día y región).

Lee la copia en parquet del Excel de despachos 2024–2026, le aplica la misma limpieza del modelo (process_shipments),
toma los despachos de la sede y les pone el flete actualizado a las tarifas de este año: el directo cotizado, buscado
solo en la caché (no abre Excel: lo que falta por cotizar queda contado en sin_flete_actualizado), o lo pagado más el
aumento de tarifa cuando el cotizado da menos (diarias.updated_freight; aumentos en el archivo privado).
Columnas: ver series/diarias.py.
Uso: .venv/Scripts/python.exe scripts/construir_series.py
"""
import pandas as pd

from despacho_multiobjetivo.config import PROCESSED
from despacho_multiobjetivo.datos import data_processing as dp
from despacho_multiobjetivo.datos_empresa import DATOS
from despacho_multiobjetivo.series import diarias

SEDE = "0111"
COPIA_DESPACHOS = PROCESSED / "despachos_2024_2026_crudo.parquet"


def select_shipments(despachos: pd.DataFrame, sede: str) -> pd.DataFrame:
    # Despachos de la sede con peso (lo mismo que exige get_shipments)
    return despachos[(despachos["sede"] == sede) & despachos["peso"].notna()].copy()


if __name__ == "__main__":
    # 1. Despachos de la sede con la limpieza del modelo y el flete actualizado de la caché
    despachos = select_shipments(dp.process_shipments(pd.read_parquet(COPIA_DESPACHOS)), SEDE)
    cotizado = diarias.cached_direct_freight(despachos)
    factores = diarias.tariff_factors(DATOS["aumento_tarifa_anual"])
    despachos["flete_actualizado"] = diarias.updated_freight(despachos, cotizado, factores)

    # 2. Series y archivos
    for nombre, por_region in (("total", False), ("region", True)):
        serie = diarias.daily_series(despachos, por_region=por_region)
        ruta = PROCESSED / f"series_diarias_{SEDE}_{nombre}.csv"
        serie.to_csv(ruta, index=False)
        print(f"{ruta.name}: {len(serie):,} filas")

    # 3. Resumen por año (para revisar que todo cuadre)
    total = diarias.daily_series(despachos)
    resumen = total.groupby(total["fecha"].dt.year).agg(
        dias=("fecha", "size"), pedidos=("pedidos", "sum"), kilos=("kilos", "sum"),
        flete_pagado=("flete_pagado", "sum"), sin_flete_pagado=("sin_flete_pagado", "sum"),
        flete_actualizado=("flete_actualizado", "sum"), sin_flete_actualizado=("sin_flete_actualizado", "sum"))
    print(resumen.to_string())
