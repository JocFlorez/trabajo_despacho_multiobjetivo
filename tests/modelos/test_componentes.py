"""Piezas comunes de las Etapas 2 y 3: datos de cada opción y variables."""
from datetime import date

import pandas as pd
import pulp
import pytest

from despacho_multiobjetivo.modelos import componentes as comp

SEDE = "0111"
NODOS = {"ANTIOQUIA_CHOCO": "MEDELLIN", "ORIENTE": None}
LUNES, MARTES = date(2026, 1, 5), date(2026, 1, 6)


def _pedidos(filas):
    """filas: (id, tipo, región, peso, flete_sin_consolidar, flete_triangulacion)."""
    return pd.DataFrame(filas, columns=["id_pedido", "tipo", "region_logistica", "peso_kg",
                                        "flete_sin_consolidar", "flete_triangulacion"])


def _opciones(filas):
    """filas: (id, dia_salida)."""
    return pd.DataFrame(filas, columns=["id_pedido", "dia_salida"])


# ------------------------------------------------------------------- datos

def test_enriquecer_opciones_trae_los_datos_del_pedido():
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50)])
    opciones = comp.enrich_options(_opciones([("C_0", LUNES), ("C_0", MARTES)]), pedidos)
    assert len(opciones) == 2
    assert opciones[["tipo", "peso_kg", "region_logistica", "flete_triangulacion"]].iloc[0].tolist() == \
        ["CLIENTES", 1000, "ANTIOQUIA_CHOCO", 50]


def test_enriquecer_opciones_falla_si_el_pedido_esta_repetido():
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50),
                        ("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50)])
    with pytest.raises(pd.errors.MergeError):
        comp.enrich_options(_opciones([("C_0", LUNES)]), pedidos)


# -------------------------------------------------------------- variables

def test_x_por_pedido_y_dia_binaria_para_clientes_y_continua_para_el_resto():
    opciones = pd.DataFrame({"id_pedido": ["C_0", "C_0", "T_0"], "dia_salida": [LUNES, MARTES, LUNES],
                             "tipo": ["CLIENTES", "CLIENTES", "TRASLADOS"]})
    x = comp.create_variables_x(opciones)
    assert set(x) == {("C_0", LUNES), ("C_0", MARTES), ("T_0", LUNES)}
    # PuLP guarda las binarias como enteras acotadas entre 0 y 1
    assert x["C_0", LUNES].cat == pulp.const.LpInteger
    assert (x["C_0", LUNES].lowBound, x["C_0", LUNES].upBound) == (0, 1)
    assert x["T_0", LUNES].cat == pulp.const.LpContinuous
    assert (x["T_0", LUNES].lowBound, x["T_0", LUNES].upBound) == (0, 1)


def test_otros_binarios_y_traslados_al_gasto_continuos():
    opciones = pd.DataFrame({"id_pedido": ["O_0", "G_0"], "dia_salida": [LUNES, LUNES],
                             "tipo": ["OTROS", "TRASLADOS AL GASTO"]})
    x = comp.create_variables_x(opciones)
    assert x["O_0", LUNES].cat == pulp.const.LpInteger
    assert x["G_0", LUNES].cat == pulp.const.LpContinuous


def test_y_por_region_y_dia_solo_con_hub_y_tarifa():
    capacidades = {"Turbo": 4500, "Mula": 34000}
    tarifas = {("Turbo", SEDE, "MEDELLIN"): 2_100_000}
    regiones_dias = pd.DataFrame({"region_logistica": ["ANTIOQUIA_CHOCO", "ANTIOQUIA_CHOCO", "ORIENTE"],
                                  "dia_salida": [LUNES, MARTES, LUNES]})
    y = comp.create_variables_y(capacidades, regiones_dias, tarifas, NODOS, SEDE)
    # Mula no tiene tarifa y Oriente no tiene hub
    assert list(y) == [("Turbo", "ANTIOQUIA_CHOCO", LUNES), ("Turbo", "ANTIOQUIA_CHOCO", MARTES)]
