from datetime import date

import numpy as np
import pandas as pd
import pytest

from despacho_multiobjetivo.reglas_negocio import opciones_salida as t

# Enero de 2026: lunes 12 festivo. Semana del 19 al 23 normal.


# ------------------------------------------------------------ operation_start

@pytest.mark.parametrize("captura, esperado", [
    (date(2026, 1, 16), date(2026, 1, 16)),   # viernes normal
    (date(2026, 1, 17), date(2026, 1, 17)),   # sábado: sí se despacha
    (date(2026, 1, 18), date(2026, 1, 19)),   # domingo → lunes
    (date(2026, 1, 12), date(2026, 1, 13)),   # lunes festivo → martes
    (date(2026, 1, 11), date(2026, 1, 13)),   # domingo antes de lunes festivo → martes
])
def test_inicio_operacion(captura, esperado):
    assert t.operation_start(captura) == esperado


# -------------------------------------------------------- possible_departure_days

@pytest.mark.parametrize("captura, espera_maxima, esperado", [
    (date(2026, 1, 16), 2, [date(2026, 1, 16), date(2026, 1, 17), date(2026, 1, 19)]),
    (date(2026, 1, 16), 0, [date(2026, 1, 16)]),
    (date(2026, 1, 16), 0.5, [date(2026, 1, 16), date(2026, 1, 17)]),
    (date(2026, 1, 19), 3, [date(2026, 1, 19), date(2026, 1, 20), date(2026, 1, 21), date(2026, 1, 22)]),
    (date(2026, 1, 9), 2, [date(2026, 1, 9), date(2026, 1, 10), date(2026, 1, 13)]),   # salta dom y lunes festivo
    (date(2026, 1, 18), 0, [date(2026, 1, 19)]),                                       # captura en domingo
    (date(2026, 1, 18), 1, [date(2026, 1, 19), date(2026, 1, 20)]),
])
def test_dias_salida_posibles(captura, espera_maxima, esperado):
    assert t.possible_departure_days(captura, espera_maxima) == esperado


# --------------------------------------------------------------- fecha_entrega

@pytest.mark.parametrize("salida, transito, manipulacion, ultimo_tramo, directo, esperado", [
    (date(2026, 1, 19), 1, 1, 1, False, date(2026, 1, 22)),   # descarga mar 20 → mié 21 → jue 22
    (date(2026, 1, 23), 1, 1, 1, False, date(2026, 1, 27)),   # llega y descarga sáb 24 → lun 26 → mar 27
    (date(2026, 1, 20), 2, 1, 1, False, date(2026, 1, 26)),   # descarga jue 22 → vie 23 → lun 26
    (date(2026, 1, 19), 1, 1, 1, True, date(2026, 1, 20)),    # traslado directo: entrega = descarga
    (date(2026, 1, 19), 1, 0, 1, False, date(2026, 1, 21)),   # sin manipulación
])
def test_fecha_entrega(salida, transito, manipulacion, ultimo_tramo, directo, esperado):
    assert t.delivery_date(salida, transito, manipulacion, ultimo_tramo, directo) == esperado


def test_fecha_entrega_acepta_booleano_de_numpy():
    assert t.delivery_date(date(2026, 1, 19), 1, 1, 1, np.bool_(True)) == date(2026, 1, 20)


# --------------------------------------------------------------------- retraso

@pytest.mark.parametrize("entrega, limite, esperado", [
    (date(2026, 1, 22), date(2026, 1, 23), 0.0),
    (date(2026, 1, 23), date(2026, 1, 23), 0.0),
    (date(2026, 1, 26), date(2026, 1, 23), 1.5),
    (date(2026, 1, 28), date(2026, 1, 23), 3.5),
    (date(2026, 1, 13), date(2026, 1, 9), 1.5),   # pasa por lunes festivo
])
def test_retraso(entrega, limite, esperado):
    assert t.delay(entrega, limite) == esperado


def test_retraso_con_sabado_no_habil():
    # Para medir el servicio el sábado no cuenta: de vie 23 a lun 26 es 1 día de retraso, no 1,5
    assert t.delay(date(2026, 1, 26), date(2026, 1, 23), valor_sabado=0) == 1.0
    assert t.delay(date(2026, 1, 28), date(2026, 1, 23), valor_sabado=0) == 3.0


# --------------------------------------------------- build_departure_options

TIEMPOS_FTL = {("0111", "MANIZALES"): np.int64(1), ("0111", "MEDELLIN"): np.int64(1)}


def _pedidos():
    base = {"sede": "0111", "sin_ruta_triangulacion": False, "llega_directo": False,
            "fecha_captura": pd.Timestamp("2026-01-19"), "promesa_estricta": 3.0, "tiempo_triangulacion": 1.0}
    filas = [
        {"id_pedido": "PEREIRA_0", "nodo_regional": "MANIZALES"},
        {"id_pedido": "TRASLADO_0", "nodo_regional": "MEDELLIN", "llega_directo": True, "tiempo_triangulacion": 0.0},
        {"id_pedido": "SIN_HUB_0", "nodo_regional": np.nan},
        {"id_pedido": "BIG_M_0", "nodo_regional": "MANIZALES", "sin_ruta_triangulacion": True},
        {"id_pedido": "SIN_TRANSITO_0", "nodo_regional": "BOGOTA"},
        {"id_pedido": "SIN_FECHA_0", "nodo_regional": "MANIZALES", "fecha_captura": pd.NaT},
    ]
    return pd.DataFrame([{**base, **f} for f in filas])


def test_construir_opciones_solo_incluye_pedidos_consolidables():
    opciones = t.build_departure_options(_pedidos(), TIEMPOS_FTL, espera_maxima=4)
    assert set(opciones["id_pedido"]) == {"PEREIRA_0", "TRASLADO_0"}
    assert list(opciones.columns) == t.COLUMNAS_OPCIONES


def test_construir_opciones_calcula_entrega_y_retraso_por_dia_de_salida():
    opciones = t.build_departure_options(_pedidos(), TIEMPOS_FTL, espera_maxima=4)
    pereira = opciones[opciones["id_pedido"] == "PEREIRA_0"].set_index("dia_salida")
    # captura lun 19, promesa estricta 3 (sábado no hábil) → límite jue 22
    assert (pereira["fecha_limite"] == date(2026, 1, 22)).all()
    assert list(pereira.index) == [date(2026, 1, d) for d in (19, 20, 21, 22, 23)]
    assert pereira.loc[date(2026, 1, 19), "fecha_entrega"] == date(2026, 1, 22)
    assert pereira.loc[date(2026, 1, 19), "retraso"] == 0.0     # llega justo el día límite
    assert pereira.loc[date(2026, 1, 23), "espera"] == 4.0
    # sale vie 23, llega y descarga sáb 24, manipulación hasta lun 26, último tramo → mar 27
    assert pereira.loc[date(2026, 1, 23), "fecha_entrega"] == date(2026, 1, 27)
    # de jue 22 a mar 27: vie, lun y mar = 3 días hábiles (el sábado no cuenta)
    assert pereira.loc[date(2026, 1, 23), "retraso"] == 3.0


def test_construir_opciones_traslado_directo_entrega_el_dia_de_descarga():
    opciones = t.build_departure_options(_pedidos(), TIEMPOS_FTL, espera_maxima=0)
    traslado = opciones[opciones["id_pedido"] == "TRASLADO_0"].iloc[0]
    assert traslado["dia_salida"] == date(2026, 1, 19)
    assert traslado["fecha_entrega"] == date(2026, 1, 20)


def test_construir_opciones_captura_en_domingo_cuenta_promesa_desde_el_lunes():
    pedidos = _pedidos().iloc[[0]].assign(fecha_captura=pd.Timestamp("2026-01-18"))
    opciones = t.build_departure_options(pedidos, TIEMPOS_FTL, espera_maxima=0)
    fila = opciones.iloc[0]
    assert fila["dia_salida"] == date(2026, 1, 19)
    assert fila["espera"] == 0.0
    assert fila["fecha_limite"] == date(2026, 1, 22)   # igual que capturado el lunes


def test_construir_opciones_sin_pedidos_consolidables_devuelve_tabla_vacia():
    pedidos = _pedidos().iloc[[2, 3]]
    opciones = t.build_departure_options(pedidos, TIEMPOS_FTL, espera_maxima=2)
    assert opciones.empty
    assert list(opciones.columns) == t.COLUMNAS_OPCIONES
