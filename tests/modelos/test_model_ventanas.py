from datetime import date

import pandas as pd
import pulp
import pytest

from despacho_multiobjetivo.modelos import model_mensual
from despacho_multiobjetivo.modelos import model_ventanas as mv

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


def _resolver(opciones, pedidos, capacidades, tarifas, nodos=NODOS):
    problema, x, u, y = mv.build_model(opciones, pedidos, capacidades, tarifas, nodos, SEDE)
    problema.solve(pulp.HiGHS(msg=0))
    assert pulp.LpStatus[problema.status] == "Optimal"
    return problema, x, u, y


# ----------------------------------------------------- decisiones del modelo

def test_consolida_el_dia_que_junta_mas_pedidos():
    # C_0 solo sale el lunes, C_2 solo el martes, C_1 cualquiera: C_1 espera al martes con C_2.
    # El directo de C_2 es más caro que el de C_0, así que juntarse el martes es la única mejor opción
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50),
                        ("C_1", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50),
                        ("C_2", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 800, 50)])
    opciones = _opciones([("C_0", LUNES), ("C_1", LUNES), ("C_1", MARTES), ("C_2", MARTES)])
    problema, x, _, y = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 600})
    assert x["C_1", MARTES].varValue == 1 and x["C_2", MARTES].varValue == 1
    assert x["C_0", LUNES].varValue == 0
    assert y[("Turbo", "ANTIOQUIA_CHOCO", MARTES)].varValue == 1
    assert y[("Turbo", "ANTIOQUIA_CHOCO", LUNES)].varValue == 0
    assert pulp.value(problema.objective) == pytest.approx(600 + 2 * 50 + 500)


def test_un_pedido_sale_maximo_una_vez():
    # El camión se compra ambos días por C_0 y C_2; C_1 no puede subirse a los dos
    pedidos = _pedidos([(f"C_{i}", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 0) for i in range(3)])
    opciones = _opciones([("C_0", LUNES), ("C_1", LUNES), ("C_1", MARTES), ("C_2", MARTES)])
    _, x, _, _ = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 100})
    assert x["C_1", LUNES].varValue + x["C_1", MARTES].varValue == pytest.approx(1)


def test_no_consolida_cuando_el_camion_no_se_paga():
    pedidos = _pedidos([(f"C_{i}", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50) for i in range(3)])
    opciones = _opciones([(f"C_{i}", LUNES) for i in range(3)])
    problema, x, _, y = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 2000})
    assert all(v.varValue == 0 for v in x.values())
    assert pulp.value(problema.objective) == pytest.approx(3 * 500)


def test_pedido_sin_opciones_paga_siempre_el_directo():
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50),
                        ("C_1", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50),
                        ("SIN_OPC", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 700, None)])
    opciones = _opciones([("C_0", LUNES), ("C_1", LUNES)])
    problema, x, _, _ = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 100})
    assert not any(pedido == "SIN_OPC" for pedido, _ in x)
    assert pulp.value(problema.objective) == pytest.approx(100 + 2 * 50 + 700)


def test_la_capacidad_de_un_dia_no_sirve_para_otro():
    # 3.000 kg el lunes y 3.000 kg el martes: con un solo Turbo de 4.500 no alcanza, se compran dos
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 3000, 5000, 0),
                        ("C_1", "CLIENTES", "ANTIOQUIA_CHOCO", 3000, 5000, 0)])
    opciones = _opciones([("C_0", LUNES), ("C_1", MARTES)])
    problema, x, _, y = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1000})
    assert y[("Turbo", "ANTIOQUIA_CHOCO", LUNES)].varValue == 1
    assert y[("Turbo", "ANTIOQUIA_CHOCO", MARTES)].varValue == 1
    assert pulp.value(problema.objective) == pytest.approx(2 * 1000)


def test_region_sin_tarifa_no_puede_consolidar():
    pedidos = _pedidos([("C_0", "CLIENTES", "ORIENTE", 1000, 500_000, 0)])
    opciones = _opciones([("C_0", LUNES)])
    _, x, _, y = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1})
    assert x["C_0", LUNES].varValue == 0
    assert y == {}


def test_traslado_puede_consolidarse_parcialmente():
    # 5.000 kg en un Turbo de 4.500: llena el camión con el 90 % y el resto va directo.
    # 1 Turbo: 1.000 + 0,1 * 2.000 = 1.200 < 2 Turbos (2.000) = todo directo (2.000)
    pedidos = _pedidos([("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 5000, 2000, 0)])
    opciones = _opciones([("T_0", LUNES)])
    problema, x, _, y = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1000})
    assert x["T_0", LUNES].varValue == pytest.approx(0.9)
    assert y[("Turbo", "ANTIOQUIA_CHOCO", LUNES)].varValue == 1
    assert pulp.value(problema.objective) == pytest.approx(1200)


def test_traslado_repartido_entre_dias_no_supera_el_total():
    # Camión barato los dos días: el traslado puede dividirse, pero en total sale una sola vez
    pedidos = _pedidos([("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 9000, 5000, 0)])
    opciones = _opciones([("T_0", LUNES), ("T_0", MARTES)])
    problema, x, _, _ = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1000})
    assert x["T_0", LUNES].varValue + x["T_0", MARTES].varValue == pytest.approx(1)
    assert pulp.value(problema.objective) == pytest.approx(2 * 1000)


# ------------------------------------------------------ restricción de enlace

def test_enlace_una_restriccion_por_opcion():
    pedidos = _pedidos([(f"C_{i}", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50) for i in range(3)])
    opciones = _opciones([("C_0", LUNES), ("C_1", LUNES), ("C_1", MARTES), ("C_2", MARTES)])
    problema, _, _, _ = mv.build_model(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 600}, NODOS, SEDE)
    assert sum(nombre.startswith("enlace_") for nombre in problema.constraints) == len(opciones)


def test_enlace_la_relajacion_no_compra_fracciones_de_camion():
    # 100 kg con directo de 300.000 y un Turbo de 1.900.000: en la realidad va directo.
    # Sin enlace, la relajación compraría 100/4.500 de Turbo (41.900) y su cota sería 91.900.
    # Con enlace debe pagar el camión completo, así que la cota ya es el óptimo real: 300.000
    pedidos = _pedidos([("C_0", "CLIENTES", "SUR_OCCIDENTE", 100, 300_000, 50_000)])
    nodos = {"SUR_OCCIDENTE": "CALI"}
    problema, _, _, _ = mv.build_model(_opciones([("C_0", LUNES)]), pedidos, {"Turbo": 4500},
                                       {("Turbo", SEDE, "CALI"): 1_900_000}, nodos, SEDE)
    for variable in problema.variables():
        variable.cat = pulp.const.LpContinuous
    problema.solve(pulp.HiGHS(msg=0))
    assert pulp.value(problema.objective) == pytest.approx(300_000)


# ------------------------------------------------- completitud (variable u)

def test_salida_completa_una_restriccion_por_pedido():
    # También los pedidos sin opciones de salida tienen su u y su restricción
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50),
                        ("SIN_OPC", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 700, None)])
    problema, _, u, _ = mv.build_model(_opciones([("C_0", LUNES), ("C_0", MARTES)]), pedidos, {"Turbo": 4500},
                                       {("Turbo", SEDE, "MEDELLIN"): 600}, NODOS, SEDE)
    assert set(u) == {"C_0", "SIN_OPC"}
    assert sum(nombre.startswith("salida_completa_") for nombre in problema.constraints) == len(pedidos)


def test_pedido_sin_opciones_va_completo_directo():
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50),
                        ("SIN_OPC", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 700, None)])
    _, _, u, _ = _resolver(_opciones([("C_0", LUNES)]), pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 100})
    assert u["SIN_OPC"].varValue == pytest.approx(1)


def test_traslado_partido_se_despacha_completo():
    # 5.000 kg y un Turbo de 4.500: el 90 % va en camión y el 10 % directo; entre los dos suman el pedido
    pedidos = _pedidos([("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 5000, 2000, 0)])
    _, x, u, _ = _resolver(_opciones([("T_0", LUNES)]), pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 1000})
    assert u["T_0"].varValue == pytest.approx(0.1)
    assert u["T_0"].varValue + x["T_0", LUNES].varValue == pytest.approx(1)


def test_cliente_va_completo_por_una_sola_via():
    # Un cliente: o todo en camión un día, o todo directo; u queda en 0 o 1
    pedidos = _pedidos([(f"C_{i}", "CLIENTES", "ANTIOQUIA_CHOCO", 1000, 500, 50) for i in range(3)])
    opciones = _opciones([("C_0", LUNES), ("C_1", LUNES), ("C_1", MARTES), ("C_2", MARTES)])
    _, x, u, _ = _resolver(opciones, pedidos, {"Turbo": 4500}, {("Turbo", SEDE, "MEDELLIN"): 600})
    for pedido in ("C_0", "C_1", "C_2"):
        salidas = [v.varValue for (i, t), v in x.items() if i == pedido]
        assert u[pedido].varValue in (pytest.approx(0), pytest.approx(1))
        assert u[pedido].varValue + sum(salidas) == pytest.approx(1)


# ------------------------------------------------ equivalencia con el modelo mensual

def test_con_un_solo_dia_da_lo_mismo_que_el_modelo_mensual():
    # Si todos los pedidos solo pueden salir el mismo día, el modelo con ventanas es el mensual
    pedidos = _pedidos([("C_0", "CLIENTES", "ANTIOQUIA_CHOCO", 2000, 3000, 100),
                        ("C_1", "CLIENTES", "ANTIOQUIA_CHOCO", 2500, 2000, 200),
                        ("C_2", "CLIENTES", "ANTIOQUIA_CHOCO", 1500, 1800, 50),
                        ("T_0", "TRASLADOS", "ANTIOQUIA_CHOCO", 6000, 4000, 0),
                        ("C_3", "CLIENTES", "ORIENTE", 1000, 900, 10)])
    capacidades = {"Turbo": 4500, "Sencillo": 8500}
    tarifas = {("Turbo", SEDE, "MEDELLIN"): 2500, ("Sencillo", SEDE, "MEDELLIN"): 3500}

    problema_mensual, _, _, _ = model_mensual.build_model(pedidos, capacidades, tarifas, NODOS, SEDE)
    problema_mensual.solve(pulp.HiGHS(msg=0))
    problema_ventanas, _, _, _ = _resolver(_opciones([(p, LUNES) for p in pedidos["id_pedido"]]), pedidos, capacidades, tarifas)

    assert pulp.value(problema_ventanas.objective) == pytest.approx(pulp.value(problema_mensual.objective))
