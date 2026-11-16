"""Etapa 3: mismas piezas que la Etapa 2, con dos objetivos (costo y retraso esperado)."""
from datetime import date

import pandas as pd
import pulp
import pytest

from despacho_multiobjetivo.modelos import model_multiobjetivo, model_ventanas

SEDE = "0111"
NODOS = {"ANTIOQUIA_CHOCO": "MEDELLIN", "ORIENTE": None}
CAPACIDADES = {"Turbo": 4500, "Mula": 34000}
TARIFAS = {("Turbo", SEDE, "MEDELLIN"): 1_000_000}
LUNES, MARTES = date(2026, 1, 5), date(2026, 1, 6)


def _pedidos():
    filas = [  # id, tipo, región, peso, flete directo, flete último tramo
        ("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 3000, 1_500_000, 100_000),
        ("C_1", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 600_000, 50_000),
        ("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 2000, 900_000, 80_000),
        ("O_0", "CLIENTES", "ORIENTE", 500, 300_000, 40_000),
    ]
    pedidos = pd.DataFrame(filas, columns=["id_pedido", "tipo", "region_logistica", "peso_kg",
                                           "flete_sin_consolidar", "flete_triangulacion"])
    return pedidos.assign(transportadora_directo="ENVIA", tipo_transportadora="PAQUETEO", sede=SEDE, llega_directo=False)


def _opciones():
    filas = [("C_0", LUNES, 0), ("C_0", MARTES, 1), ("C_1", MARTES, 0),
             ("T_0", LUNES, 0), ("T_0", MARTES, 0), ("O_0", LUNES, 0)]
    return pd.DataFrame(filas, columns=["id_pedido", "dia_salida", "retraso"])


def test_no_fija_objetivo_y_devuelve_costo_y_retraso():
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    assert problema.objective is None or len(problema.objective) == 0
    assert isinstance(z1, pulp.LpAffineExpression) and isinstance(z2, pulp.LpAffineExpression)


def test_mismas_variables_y_restricciones_que_la_etapa_2():
    p2, x2, u2, y2 = model_ventanas.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    p3, x3, u3, y3, _, _ = model_multiobjetivo.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    assert (list(x3), list(u3), list(y3)) == (list(x2), list(u2), list(y2))
    assert list(p3.constraints) == list(p2.constraints)


def test_min_costo_de_la_etapa_3_es_el_optimo_de_la_etapa_2():
    p2, _, _, _ = model_ventanas.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    p2.solve(pulp.HiGHS(msg=0))
    p3, _, _, _, z1, _ = model_multiobjetivo.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    p3.setObjective(z1)
    p3.solve(pulp.HiGHS(msg=0))
    # Mismo óptimo calculado a mano en test_version_pyomo: un Turbo el martes con C_0, C_1 y 1/4 de T_0
    assert pulp.value(z1) == pytest.approx(pulp.value(p2.objective)) == pytest.approx(2_145_000)


def test_bajar_el_retraso_cuesta_mas():
    p3, _, _, _, z1, z2 = model_multiobjetivo.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    p3.setObjective(z1)
    p3.solve(pulp.HiGHS(msg=0))
    costo_minimo, retraso_del_mas_barato = pulp.value(z1), pulp.value(z2)

    p3 += z2 <= retraso_del_mas_barato - 0.5, "epsilon"
    p3.setObjective(z1)
    p3.solve(pulp.HiGHS(msg=0))
    assert pulp.value(z2) <= retraso_del_mas_barato - 0.5 + 1e-6
    assert pulp.value(z1) > costo_minimo


def test_z2_es_el_castigo_esperado_promedio_por_pedido():
    from despacho_multiobjetivo.reglas_negocio import servicio
    p3, x, u, y, z1, z2 = model_multiobjetivo.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    p3.setObjective(z1)
    p3.solve(pulp.HiGHS(msg=0))
    d_dir = servicio.expected_delay_direct(_pedidos())
    d_opc = servicio.expected_delay_options(_opciones(), _pedidos())
    total = (sum(d_dir[i] * v.varValue for i, v in u.items())
             + sum(d_opc[llave] * v.varValue for llave, v in x.items()))
    assert pulp.value(z2) == pytest.approx(total / len(_pedidos()))   # 4 pedidos
