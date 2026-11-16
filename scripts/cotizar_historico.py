"""Cotiza el histórico 2025-01 a 2026-10 de la sede 0111 (decisión del usuario del 2026-11-08: 2025 y 2026 para la
simulación principal) y guarda todo en la caché del cotizador (data/processed/cotizaciones_cache.parquet).

Lee la copia en parquet del Excel de despachos (data/processed/despachos_2024_2026_crudo.parquet) y aplica los mismos
pasos del cargador (process_shipments, get_shipments, rate_shipments, rate_ltl_direct) para armar las consultas:
triangulación (sede regional → destino) y directo (sede → destino). Cotiza solo lo que no está en la caché, por tandas
de 500 que se guardan al terminar cada una: si se corta, al volver a correrlo sigue donde quedó.
Abre Excel (oculto) con xlwings: no correr junto con otras corridas pesadas; cotiza unas 3.000 consultas por hora.
Mientras corre, pide a Windows no suspender el equipo por inactividad.
Uso: .venv/Scripts/python.exe scripts/cotizar_historico.py
"""
import ctypes
import os
import time

import pandas as pd

from despacho_multiobjetivo.config import EXTERNAL, PROCESSED
from despacho_multiobjetivo.datos import data_processing as dp

SEDE, DESDE, HASTA = "0111", "2025-01-01", "2026-10-31"
COPIA_DESPACHOS = PROCESSED / "despachos_2024_2026_crudo.parquet"


def keep_awake(activar: bool) -> None:
    # Pide a Windows no suspender por inactividad mientras corre (no aplica a cerrar la tapa)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if activar else 0))


def shipments_to_quote() -> pd.DataFrame:
    # Pedidos de la sede y el período, con los mismos pasos y filtros del cargador
    df = dp.clean_column_types(pd.read_parquet(COPIA_DESPACHOS))
    df = dp.fill_destination_city(df)
    df = dp.fill_units(df)
    df = dp.filter_cedis(df)
    df = dp.exclude_metro_area(df)
    df = dp.exclude_inbound_to_cedi(df)
    divipola = pd.read_excel(EXTERNAL / "divipola.xlsx")
    df["ciudad_destino"] = dp.normalize_city(df["ciudad_destino"])
    for columna in ("Municipio", "CentroPoblado", "Departamento"):
        divipola[columna] = dp.normalize_city(divipola[columna])
    df = dp.search_department(df, divipola)
    df = dp.map_region(df)
    return df[(df["sede"] == SEDE) & (df["mes"] >= DESDE) & (df["mes"] <= HASTA) & df["peso"].notna()].copy()


if __name__ == "__main__":
    keep_awake(True)
    try:
        inicio = time.perf_counter()
        df = shipments_to_quote()
        print(f"{len(df):,} pedidos de {SEDE} entre {DESDE[:7]} y {HASTA[:7]}", flush=True)

        # Triangulación: los que tienen sede regional, salvo los traslados que llegan directo a la sede del hub
        df["nodo_regional"] = df["region_logistica"].map(dp.load_regional_hubs())
        con_hub = df["nodo_regional"].notna()
        es_traslado = df["tipo_despacho"].astype(str).str.strip().str.startswith("TRASLADOS")
        llega_directo = es_traslado & con_hub & pd.Series(
            [sede in dp.SEDES_LLEGADA_HUB.get(hub, set()) for sede, hub in zip(df["sede_flete"], df["nodo_regional"])],
            index=df.index)
        a_triangular = df[con_hub & ~llega_directo]
        triangulacion = list(zip(a_triangular["nodo_regional"], a_triangular["ciudad_destino"],
                                 a_triangular["peso"], a_triangular["unidades"]))
        directo = list(zip(df["ciudad_sede"], df["ciudad_destino"], df["peso"], df["unidades"]))

        for nombre, consultas in (("triangulación", triangulacion), ("directo", directo)):
            print(f"Cotizando {nombre}: {len(consultas):,} consultas (solo se cotiza lo que falta en la caché)", flush=True)
            resultado = dp.quote(consultas)
            print(f"   {nombre} lista: {resultado['flete'].notna().sum():,} con flete · "
                  f"{(time.perf_counter() - inicio) / 3600:.1f} h", flush=True)
        print("Listo", flush=True)
    finally:
        keep_awake(False)
