import numpy as np
import pandas as pd
import pytest

from despacho_multiobjetivo.datos import data_processing as dp
from despacho_multiobjetivo.series import diarias


def _despachos():
    # Tres días: el 2 sin despachos; un despacho sin región y uno sin fecha
    return pd.DataFrame({
        "fecha_captura": pd.to_datetime(["2025-03-01", "2025-03-01", "2025-03-03", "2025-03-03", None]),
        "region_logistica": ["CENTRO", "ORIENTE", "CENTRO", None, "CENTRO"],
        "unidades": [2.0, 1.0, 4.0, 1.0, 9.0],
        "peso": [100.0, 50.0, 300.0, 10.0, 999.0],
        "valor_flete": [20_000.0, np.nan, 45_000.0, 0.0, 7.0],
        "flete_actualizado": [25_000.0, 12_000.0, np.nan, 3_000.0, 1.0],
    })


def test_serie_total_tiene_un_dia_por_fila_con_ceros_los_dias_sin_despachos():
    serie = diarias.daily_series(_despachos(), "2025-03-01", "2025-03-03")
    assert serie["fecha"].tolist() == list(pd.date_range("2025-03-01", "2025-03-03"))
    assert serie["pedidos"].tolist() == [2, 0, 2]
    assert serie["unidades"].tolist() == [3, 0, 5]
    assert serie["kilos"].tolist() == [150, 0, 310]


def test_peso_promedio_por_pedido_y_vacio_si_no_hay_pedidos():
    serie = diarias.daily_series(_despachos(), "2025-03-01", "2025-03-03")
    assert serie["peso_promedio"].iloc[0] == 75
    assert np.isnan(serie["peso_promedio"].iloc[1])


def test_flete_pagado_suma_solo_los_validos_y_cuenta_los_que_faltan():
    serie = diarias.daily_series(_despachos(), "2025-03-01", "2025-03-03")
    assert serie["flete_pagado"].tolist() == [20_000, 0, 45_000]
    assert serie["sin_flete_pagado"].tolist() == [1, 0, 1]     # el vacío del 1 y el cero del 3


def test_flete_actualizado_suma_lo_cotizado_y_cuenta_lo_que_falta():
    serie = diarias.daily_series(_despachos(), "2025-03-01", "2025-03-03")
    assert serie["flete_actualizado"].tolist() == [37_000, 0, 3_000]
    assert serie["sin_flete_actualizado"].tolist() == [0, 0, 1]


def test_sin_columna_de_flete_actualizado_queda_vacia():
    serie = diarias.daily_series(_despachos().drop(columns="flete_actualizado"), "2025-03-01", "2025-03-03")
    assert serie["flete_actualizado"].isna().all()


def test_despachos_fuera_del_rango_o_sin_fecha_no_cuentan():
    serie = diarias.daily_series(_despachos(), "2025-03-02", "2025-03-03")
    assert serie["pedidos"].tolist() == [0, 2]


def test_serie_por_region_tiene_todas_las_regiones_todos_los_dias():
    serie = diarias.daily_series(_despachos(), "2025-03-01", "2025-03-03", por_region=True)
    assert len(serie) == 2 * 3                                  # CENTRO y ORIENTE, tres días
    centro = serie[serie["region"] == "CENTRO"]
    assert centro["pedidos"].tolist() == [1, 0, 1]
    assert centro["kilos"].tolist() == [100, 0, 300]


def test_el_despacho_sin_region_cuenta_en_el_total_pero_no_en_las_regiones():
    despachos = _despachos()
    total = diarias.daily_series(despachos, "2025-03-01", "2025-03-03")
    regiones = diarias.daily_series(despachos, "2025-03-01", "2025-03-03", por_region=True)
    assert total["pedidos"].sum() == regiones["pedidos"].sum() + 1


def test_flete_directo_de_la_cache_sin_abrir_excel(monkeypatch):
    cache = pd.DataFrame({"origen": ["BOGOTA"], "destino": ["CALI"], "peso": [10.0], "unidades": [1.0],
                          "transportadora": ["ENVIA"], "flete": [15_000.0], "tiempo_entrega": [2.0],
                          "promesa_envia": [2.0]})
    monkeypatch.setattr(dp, "_load_quote_cache", lambda: cache)
    monkeypatch.setattr(dp, "_excel_quoter", lambda consultas: pytest.fail("no debe cotizar en Excel"))
    despachos = pd.DataFrame({"ciudad_sede": ["BOGOTA", "BOGOTA"], "ciudad_destino": ["CALI", "PASTO"],
                              "peso": [10.0000001, 10.0], "unidades": [1.0, 1.0]})
    fletes = diarias.cached_direct_freight(despachos)
    assert fletes.iloc[0] == 15_000                             # misma llave redondeada de la caché
    assert np.isnan(fletes.iloc[1])                             # sin cotizar todavía


AUMENTOS = {"2025": {"ENVIA": 0.10, "OTRAS": 0.05}}   # inventados (los reales están en el archivo privado)


def _guias():
    return pd.DataFrame({
        "fecha_captura": pd.to_datetime(["2025-05-02", "2025-05-02", "2025-05-02", "2026-05-02", "2025-05-02"]),
        "transportadora": ["COLVANES", "SERVIENTREGA", "COLVANES", "COLVANES", "FRACOR"],
        "valor_flete": [100_000.0, 100_000.0, 100_000.0, 100_000.0, np.nan],
    })


def test_si_el_cotizado_es_menor_que_lo_pagado_se_usa_lo_pagado_con_el_aumento():
    cotizado = pd.Series([80_000.0, 80_000.0, 120_000.0, 80_000.0, 80_000.0])
    actualizado = diarias.updated_freight(_guias(), cotizado, AUMENTOS)
    assert actualizado.iloc[0] == pytest.approx(110_000)       # ENVIA (Colvanes): + 10 %
    assert actualizado.iloc[1] == pytest.approx(105_000)       # otra transportadora: + 5 %
    assert actualizado.iloc[2] == 120_000                      # el cotizado ya es mayor: se queda


def test_anios_sin_aumento_definido_y_guias_sin_flete_pagado_quedan_con_el_cotizado():
    cotizado = pd.Series([80_000.0, 80_000.0, 120_000.0, 80_000.0, 80_000.0])
    actualizado = diarias.updated_freight(_guias(), cotizado, AUMENTOS)
    assert actualizado.iloc[3] == 80_000                       # 2026: sin regla definida
    assert actualizado.iloc[4] == 80_000                       # sin flete pagado


def test_sin_cotizacion_el_actualizado_queda_vacio():
    cotizado = pd.Series([np.nan] * 5)
    assert diarias.updated_freight(_guias(), cotizado, AUMENTOS).isna().all()


def test_procesar_despachos_acepta_un_dataframe_ya_cargado(monkeypatch):
    monkeypatch.setattr(dp, "load_shipments", lambda: pytest.fail("no debe leer el Excel de 2026"))
    llamado = {}

    def limpiar_tipos(df):
        llamado["filas"] = len(df)
        raise RuntimeError("alto")                             # basta con saber que usó el que se le pasó

    monkeypatch.setattr(dp, "clean_column_types", limpiar_tipos)
    with pytest.raises(RuntimeError):
        dp.process_shipments(pd.DataFrame({"a": [1, 2, 3]}))
    assert llamado["filas"] == 3
