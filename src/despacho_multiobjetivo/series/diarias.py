"""Series diarias del histórico de despachos para el análisis de series de tiempo.

Una fila por día calendario (domingos y festivos incluidos, en 0 si no hubo despachos), para la sede completa o por
región logística. El rango por defecto va del 8 de enero de 2024 al 25 de octubre de 2026: antes del 8 de enero el
archivo arranca incompleto y después del 25 de octubre faltan despachos de las transportadoras que cortan el 26
(decisiones de Johan del 2026-11-16). Los días se cuentan por fecha de captura.

Columnas de la serie:
  fecha, region (solo por región), pedidos (despachos del día), unidades, kilos,
  peso_promedio (kilos por pedido; vacío si no hubo pedidos),
  flete_pagado (lo que se pagó, con las tarifas de cada momento; sin los vacíos ni los ceros),
  sin_flete_pagado (pedidos del día sin flete pagado válido),
  flete_actualizado (a las tarifas de este año: el directo cotizado, o lo pagado más el aumento de tarifa cuando el
    cotizado da menos, ver updated_freight; vacío si no se pidió),
  sin_flete_actualizado (pedidos del día que todavía no tienen cotización)
"""
import numpy as np
import pandas as pd

from despacho_multiobjetivo.datos import data_processing as dp

SERIE_INICIO, SERIE_FIN = "2024-01-08", "2026-10-25"


def cached_direct_freight(despachos: pd.DataFrame) -> pd.Series:
    # Flete directo (sede → destino) cotizado de cada despacho, buscado SOLO en la caché del cotizador: nunca abre
    # Excel. Lo que no está cotizado queda vacío. Usa la misma llave redondeada de dp.quote.
    consultas = pd.DataFrame({"origen": despachos["ciudad_sede"].values, "destino": despachos["ciudad_destino"].values,
                              "peso": despachos["peso"].values, "unidades": despachos["unidades"].values})
    cache = dp._with_cache_key(dp._load_quote_cache()).drop_duplicates(subset=dp.LLAVE_CACHE, keep="last")
    cruce = dp._with_cache_key(consultas).merge(cache[dp.LLAVE_CACHE + ["flete"]], on=dp.LLAVE_CACHE, how="left")
    return pd.Series(cruce["flete"].values, index=despachos.index, name="flete_actualizado")


def updated_freight(despachos: pd.DataFrame, cotizado: pd.Series, aumentos: dict) -> pd.Series:
    # Flete actualizado a las tarifas de este año (regla de Johan del 2026-11-16). Por cada despacho de un año con
    # aumento definido: si el cotizado con las tarifas de este año es menor que lo que se pagó, el actualizado es lo
    # pagado × (1 + aumento de tarifa de su transportadora); si no, el cotizado. Años sin aumento definido y
    # despachos sin flete pagado: el cotizado. Sin cotización: vacío.
    # aumentos: {"año": {"ENVIA": aumento, "OTRAS": aumento}} (del archivo privado: son datos comerciales)
    actualizado = cotizado.copy()
    anio = despachos["fecha_captura"].dt.year.astype("Int64").astype(str)
    transportadora = despachos["transportadora"].astype("string").str.upper().str.strip().replace(dp.NOMBRE_TRANSPORTADORA)
    for anio_regla, aumento in aumentos.items():
        factor = np.where(transportadora == "ENVIA", 1 + aumento["ENVIA"], 1 + aumento["OTRAS"])
        pagado_con_aumento = despachos["valor_flete"] * factor
        sube = (anio == anio_regla) & (despachos["valor_flete"] > 0) & (cotizado < despachos["valor_flete"])
        actualizado = actualizado.where(~sube.fillna(False).astype(bool), pagado_con_aumento)
    return actualizado


def daily_series(despachos: pd.DataFrame, inicio: str = SERIE_INICIO, fin: str = SERIE_FIN,
                 por_region: bool = False) -> pd.DataFrame:
    # despachos: columnas fecha_captura, region_logistica, unidades, peso, valor_flete y, si ya se cotizó,
    # flete_actualizado. Los despachos sin fecha o fuera del rango no cuentan; los sin región cuentan en el total
    # pero no en la serie por región.

    # 1. Despachos del rango, con las marcas de lo que falta
    df = despachos[despachos["fecha_captura"].between(inicio, fin)].copy()
    df["fecha"] = df["fecha_captura"].dt.normalize()
    pagado_valido = df["valor_flete"].notna() & (df["valor_flete"] > 0)
    df["flete_pagado"] = df["valor_flete"].where(pagado_valido, 0.0)
    df["sin_flete_pagado"] = (~pagado_valido).astype(int)
    con_actualizado = "flete_actualizado" in df
    if con_actualizado:
        df["sin_flete_actualizado"] = df["flete_actualizado"].isna().astype(int)
        df["flete_actualizado"] = df["flete_actualizado"].fillna(0.0)

    # 2. Sumas por día (y región)
    claves = ["fecha", "region_logistica"] if por_region else ["fecha"]
    if por_region:
        df = df[df["region_logistica"].notna()]
    sumas = {"pedidos": ("peso", "size"), "unidades": ("unidades", "sum"), "kilos": ("peso", "sum"),
             "flete_pagado": ("flete_pagado", "sum"), "sin_flete_pagado": ("sin_flete_pagado", "sum")}
    if con_actualizado:
        sumas |= {"flete_actualizado": ("flete_actualizado", "sum"),
                  "sin_flete_actualizado": ("sin_flete_actualizado", "sum")}
    agregado = df.groupby(claves).agg(**sumas)

    # 3. Todos los días del rango (y todas las regiones), con 0 donde no hubo despachos
    dias = pd.date_range(inicio, fin, freq="D", name="fecha")
    if por_region:
        indice = pd.MultiIndex.from_product([dias, sorted(df["region_logistica"].unique())],
                                            names=["fecha", "region_logistica"])
    else:
        indice = dias
    serie = agregado.reindex(indice, fill_value=0).reset_index().rename(columns={"region_logistica": "region"})
    if por_region:
        serie = serie.sort_values(["region", "fecha"], kind="stable").reset_index(drop=True)

    # 4. Peso promedio y flete actualizado vacío si no se pidió
    serie["peso_promedio"] = serie["kilos"] / serie["pedidos"].replace(0, np.nan)
    if not con_actualizado:
        serie["flete_actualizado"] = np.nan
        serie["sin_flete_actualizado"] = np.nan
    columnas = (["fecha"] + (["region"] if por_region else []) +
                ["pedidos", "unidades", "kilos", "peso_promedio", "flete_pagado", "sin_flete_pagado",
                 "flete_actualizado", "sin_flete_actualizado"])
    return serie[columnas]
