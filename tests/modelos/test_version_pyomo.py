"""El modelo en Pyomo debe ser el mismo que el de PuLP: mismo tamaño, mismos dominios y mismos óptimos."""
from datetime import date

import pandas as pd
import pulp
import pyomo.environ as pyo
import pytest

from despacho_multiobjetivo.modelos import model_multiobjetivo as mo
from despacho_multiobjetivo.modelos import version_pyomo as mp
from despacho_multiobjetivo.metodos.resolver_pyomo import cost_extreme, epsilon_point

SEDE = "0111"
NODOS = {"ANTIOQUIA_CHOCO": "MEDELLIN", "ORIENTE": None}
CAPACIDADES = {"Turbo": 4500, "Mula": 34000}
TARIFAS = {("Turbo", SEDE, "MEDELLIN"): 1_000_000}   # la mula no tiene tarifa: no se crea su variable
LUNES, MARTES = date(2026, 1, 5), date(2026, 1, 6)


def _pedidos():
    filas = [  # id, tipo, región, peso, flete directo, flete último tramo
        ("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 3000, 1_500_000, 100_000),
        ("C_1", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 600_000, 50_000),
        ("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 2000, 900_000, 80_000),
        ("O_0", "CLIENTES", "ORIENTE", 500, 300_000, 40_000),        # Oriente no tiene hub: va directo
    ]
    pedidos = pd.DataFrame(filas, columns=["id_pedido", "tipo", "region_logistica", "peso_kg",
                                           "flete_sin_consolidar", "flete_triangulacion"])
    return pedidos.assign(transportadora_directo="ENVIA", tipo_transportadora="PAQUETEO", sede=SEDE, llega_directo=False)


def _opciones():
    filas = [("C_0", LUNES, 0), ("C_0", MARTES, 1), ("C_1", MARTES, 0),
             ("T_0", LUNES, 0), ("T_0", MARTES, 0), ("O_0", LUNES, 0)]
    return pd.DataFrame(filas, columns=["id_pedido", "dia_salida", "retraso"])


def _modelos():
    problema, x, u, y, z1, z2 = mo.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    modelo = mp.build_model(_opciones(), _pedidos(), CAPACIDADES, TARIFAS, NODOS, SEDE)
    return (problema, x, u, y, z1, z2), modelo


def _resolver_pulp(problema, objetivo):
    problema.setObjective(objetivo)
    problema.solve(pulp.HiGHS(msg=0))
    assert pulp.LpStatus[problema.status] == "Optimal"


def test_mismo_tamano_que_pulp():
    (problema, x, u, y, _, _), modelo = _modelos()
    assert (len(modelo.x), len(modelo.u), len(modelo.y)) == (len(x), len(u), len(y)) == (6, 4, 2)
    restricciones_pyomo = sum(len(c) for c in modelo.component_objects(pyo.Constraint))
    assert restricciones_pyomo == len(problema.constraints)


def test_clientes_binarios_y_traslados_continuos():
    _, modelo = _modelos()
    assert modelo.x["C_0", LUNES].domain is pyo.Binary
    assert modelo.x["T_0", LUNES].domain is pyo.UnitInterval
    assert modelo.u["C_0"].domain is pyo.UnitInterval
    assert modelo.y["Turbo", "ANTIOQUIA_CHOCO", LUNES].domain is pyo.NonNegativeIntegers


def test_parametros_con_los_datos_del_pedido_y_documentados():
    _, modelo = _modelos()
    assert modelo.c_dir["C_0"] == 1_500_000 and modelo.c_tri["C_0"] == 100_000 and modelo.w["T_0"] == 2000
    assert modelo.q["Turbo"] == 4500 and modelo.f["Turbo", "ANTIOQUIA_CHOCO"] == 1_000_000
    assert not any(r == "ORIENTE" for _, r in modelo.VR)            # Oriente no tiene sede regional: no hay camión
    assert modelo.d["C_0", MARTES] > modelo.d["C_0", LUNES]         # salir un día después suma retraso
    for componente in modelo.component_objects((pyo.Set, pyo.Param, pyo.Var, pyo.Constraint)):
        if not componente.name.endswith("_index"):                  # índices internos que crea Pyomo
            assert componente.doc, f"{componente.name} sin descripción"


def test_costo_minimo_calculado_a_mano():
    # Martes: un Turbo con C_0 + C_1 (4.000 kg) y 1/4 de T_0 (500 kg) = 1.000.000 + 100.000 + 50.000 + 20.000;
    # el resto de T_0 (675.000) y O_0 (300.000) van directo. Total 2.145.000
    _, modelo = _modelos()
    resultado = cost_extreme(modelo, mostrar=False)
    assert resultado["estado"] == "convergenceCriteriaSatisfied"
    assert resultado["costo"] == pytest.approx(2_145_000)
    assert pyo.value(modelo.x["T_0", MARTES]) == pytest.approx(0.25)
    assert pyo.value(modelo.u["O_0"]) == pytest.approx(1)


def test_mismo_extremo_de_costo_que_pulp():
    (problema, _, _, _, z1, z2), modelo = _modelos()
    _resolver_pulp(problema, z1)
    resultado = cost_extreme(modelo, mostrar=False)
    assert resultado["costo"] == pytest.approx(pulp.value(z1))
    assert resultado["retraso"] == pytest.approx(pulp.value(z2))


def test_mismo_punto_epsilon_que_pulp():
    # Con un presupuesto de retraso menor al del plan más barato, las dos librerías deben llegar al mismo costo
    (problema, _, _, _, z1, z2), modelo = _modelos()
    _resolver_pulp(problema, z1)
    epsilon = pulp.value(z2) - 0.5
    problema += z2 <= epsilon * (1 + 1e-6), "epsilon"
    _resolver_pulp(problema, z1)

    resultado = epsilon_point(modelo, epsilon, mostrar=False)
    assert resultado["costo"] == pytest.approx(pulp.value(z1))
    assert resultado["retraso"] <= epsilon * (1 + 1e-6)
    assert not hasattr(modelo, "epsilon")   # el modelo queda como estaba
