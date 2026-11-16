import pandas as pd
import pulp
import pytest

from despacho_multiobjetivo.modelos import model_mensual

SEDE = "0111"
NODOS = {"ANTIOQUIA_CHOCO": "MEDELLIN", "ORIENTE": None}


def _pedidos(filas):
    """filas: (id, tipo, región, peso, flete_sin_consolidar, flete_triangulacion)."""
    return pd.DataFrame(filas, columns=["id_pedido", "tipo", "region_logistica", "peso_kg",
                                        "flete_sin_consolidar", "flete_triangulacion"])


def _resolver(pedidos, capacidades, tarifas, nodos=NODOS):
    problema, x, u, y = model_mensual.build_model(pedidos, capacidades, tarifas, nodos, SEDE)
    problema.solve(pulp.HiGHS(msg=0))
    assert pulp.LpStatus[problema.status] == "Optimal"
    return problema, x, u, y


# -------------------------------------------------------------- variables

def test_x_es_binaria_para_clientes_y_continua_para_el_resto():
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 10, 100, 10),
                        ("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 10, 100, 10)])
    x = model_mensual.create_variables_x(pedidos)
    # PuLP guarda las binarias como enteras acotadas entre 0 y 1
    assert x["C_0"].cat == pulp.const.LpInteger
    assert (x["C_0"].lowBound, x["C_0"].upBound) == (0, 1)
    assert x["T_0"].cat == pulp.const.LpContinuous
    assert (x["T_0"].lowBound, x["T_0"].upBound) == (0, 1)


def test_otros_van_completos_y_traslados_al_gasto_se_pueden_partir():
    pedidos = _pedidos([("O_0", "OTROS", "ANTIOQUIA_CHOCO", 10, 100, 10),
                        ("G_0", "TRASLADOS AL GASTO", "ANTIOQUIA_CHOCO", 10, 100, 10)])
    x = model_mensual.create_variables_x(pedidos)
    assert x["O_0"].cat == pulp.const.LpInteger
    assert x["G_0"].cat == pulp.const.LpContinuous


def test_y_solo_existe_con_hub_y_tarifa():
    capacidades = {"Turbo": 4500, "Mula": 34000}
    tarifas = {("Turbo", SEDE, "MEDELLIN"): 2_100_000}
    y = model_mensual.create_variables_y(capacidades, ["ANTIOQUIA_CHOCO", "ORIENTE"], tarifas, NODOS, SEDE)
    # Mula no tiene tarifa y Oriente no tiene hub
    assert list(y) == [("Turbo", "ANTIOQUIA_CHOCO")]


# ----------------------------------------------------- decisiones del modelo

def test_consolida_cuando_el_camion_sale_mas_barato():
    pedidos = _pedidos([(f"C_{i}", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50) for i in range(3)])
    problema, x, _, y = _resolver(pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 600})
    assert all(v.varValue == 1 for v in x.values())
    assert y[("Turbo", "ANTIOQUIA_CHOCO")].varValue == 1
    assert pulp.value(problema.objective) == pytest.approx(600 + 3 * 50)


def test_no_consolida_cuando_el_camion_no_se_paga():
    pedidos = _pedidos([(f"C_{i}", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50) for i in range(3)])
    problema, x, _, y = _resolver(pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 2000})
    assert all(v.varValue == 0 for v in x.values())
    assert y[("Turbo", "ANTIOQUIA_CHOCO")].varValue == 0
    assert pulp.value(problema.objective) == pytest.approx(3 * 500)


def test_escoge_el_vehiculo_mas_barato_para_el_peso():
    # 6.000 kg: 2 Turbos (2.000) o 1 Sencillo (1.500) -> Sencillo
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 3000, 5000, 0),
                        ("C_1", "CLIENTES", "ANTIOQUIA_CHOCO", 3000, 5000, 0)])
    capacidades = {"Turbo": 4500, "Sencillo": 8500}
    tarifas = {("Turbo", SEDE, "MEDELLIN"): 1000, ("Sencillo", SEDE, "MEDELLIN"): 1500}
    _, x, _, y = _resolver(pedidos, capacidades, tarifas)
    assert y[("Sencillo", "ANTIOQUIA_CHOCO")].varValue == 1
    assert y[("Turbo", "ANTIOQUIA_CHOCO")].varValue == 0


def test_region_sin_tarifa_no_puede_consolidar():
    pedidos = _pedidos([("C_0", "CLIENTES", "ORIENTE", 1000, 500_000, 0)])
    problema, x, _, y = _resolver(pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1})
    assert x["C_0"].varValue == 0
    assert y == {}


def test_traslado_puede_consolidarse_parcialmente():
    # 5.000 kg en un Turbo de 4.500: llena el camión con el 90 % y el resto va directo.
    # 1 Turbo: 1.000 + 0,1 * 2.000 = 1.200 < 2 Turbos (2.000) = todo directo (2.000)
    pedidos = _pedidos([("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 5000, 2000, 0)])
    problema, x, _, y = _resolver(pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1000})
    assert x["T_0"].varValue == pytest.approx(0.9)
    assert y[("Turbo", "ANTIOQUIA_CHOCO")].varValue == 1
    assert pulp.value(problema.objective) == pytest.approx(1200)


def test_el_peso_consolidado_no_supera_la_capacidad_comprada():
    pedidos = _pedidos([(f"C_{i}", "CLIENTES", "ANTIOQUIA_CHOCO", 2000, 3000, 0) for i in range(5)])
    _, x, _, y = _resolver(pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1000})
    peso = sum(2000 * v.varValue for v in x.values())
    assert peso <= 4500 * y[("Turbo", "ANTIOQUIA_CHOCO")].varValue + 1e-6


def test_big_m_evita_consolidar_un_pedido_sin_ruta():
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50_000_000),
                        ("C_1", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50)])
    _, x, _, _ = _resolver(pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 100})
    assert x["C_0"].varValue == 0
    assert x["C_1"].varValue == 1


# --------------------------------------------------- salida completa (variable u)

def test_salida_completa_cada_pedido_suma_uno():
    # 5.000 kg en un Turbo de 4.500: 90 % en camión y 10 % directo, el traslado sale completo
    pedidos = _pedidos([("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 5000, 2000, 0),
                        ("C_0", "CLIENTES", "ORIENTE", 1000, 500, 0)])
    problema, x, u, _ = _resolver(pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1000})
    assert sum(n.startswith("salida_completa_") for n in problema.constraints) == len(pedidos)
    assert u["T_0"].varValue == pytest.approx(0.1)
    assert u["T_0"].varValue + x["T_0"].varValue == pytest.approx(1)
    assert u["C_0"].varValue == pytest.approx(1)   # región sin camión: va completo directo
