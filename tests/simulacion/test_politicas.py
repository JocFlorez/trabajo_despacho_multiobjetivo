"""Política C del Monte Carlo (fase 4): la tabú busca la decisión de hoy y la califica con _lote en N futuros."""
from datetime import date

import pandas as pd
import pytest

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import truck_cost_table
from despacho_multiobjetivo.simulacion.politicas import (best_today_decision, decision_orders, direct_delay,
                                                        make_policy_c, policy_tables)
from tests.metodos import instancia_pequena as pequena
from tests.simulacion.test_decision import JUE, LUN, MAR, MIE, _pedido

TABLA = truck_cost_table(pequena.CAPACIDADES, pequena.TARIFAS, pequena.SEDE, "MEDELLIN", 20_000)
ANT = "ANTIOQUIA_CHOCO"


# ---------------------------------------------------------------- tablas y pedidos de la decisión

def test_tablas_solo_para_regiones_con_sede_regional():
    tablas = policy_tables(pequena.CAPACIDADES, pequena.TARIFAS, pequena.NODOS, pequena.SEDE, peso_maximo=10_000)
    assert set(tablas) == {ANT}
    assert len(tablas[ANT][0]) == 10_001


def _conocidos_pequenos(fraccion_t0=1.0):
    pedidos, _ = pequena.datos()
    return pedidos.assign(fecha_captura=pd.to_datetime([pequena.LUN] * 4), dia_llegada=pequena.LUN,
                          ultimo_dia=pequena.MIE, fraccion=[1.0, 1.0, fraccion_t0, 1.0])


def test_pedidos_de_la_decision_por_region_con_sus_castigos():
    instancia = pequena.instancia()
    por_region = decision_orders(_conocidos_pequenos(), instancia["opciones"]["castigo"], 3, {ANT})
    assert set(por_region) == {ANT}                                   # O_0 (Oriente) no tiene camión
    pedidos = {p["id"]: p for p in por_region[ANT]}
    assert set(pedidos) == {"C_0", "C_1", "T_0"}
    assert set(pedidos["C_0"]["castigo_camion"]) == {pequena.LUN, pequena.MAR}   # sus opciones de camión
    assert pedidos["C_0"]["flete_directo"] == 1_500_000 and pedidos["C_0"]["flete_tramo"] == 100_000
    directo = pedidos["C_0"]["castigo_directo"]
    assert directo[pequena.LUN] < directo[pequena.MAR] < directo[pequena.MIE]   # esperar castiga más


def test_la_fraccion_pendiente_reduce_peso_fletes_y_castigos():
    instancia = pequena.instancia()
    completo = {p["id"]: p for p in decision_orders(_conocidos_pequenos(), instancia["opciones"]["castigo"], 3, {ANT})[ANT]}
    cuarto = {p["id"]: p for p in decision_orders(_conocidos_pequenos(0.25), instancia["opciones"]["castigo"], 3, {ANT})[ANT]}
    assert cuarto["T_0"]["peso"] == pytest.approx(0.25 * completo["T_0"]["peso"])
    assert cuarto["T_0"]["flete_directo"] == pytest.approx(0.25 * completo["T_0"]["flete_directo"])


# ---------------------------------------------------------------- la mejor decisión de hoy (una región)

def test_espera_si_en_los_futuros_suele_completarse_el_camion():
    # En 9 de 10 futuros llega D el martes y A + D pagan una Turbo: esperar sale más barato en promedio
    a = _pedido("A", 2000, 1_500_000)
    d = _pedido("D", 2000, 1_500_000, llegada=MAR, ultimo_dia=MIE, dias_camion=(MAR, MIE))
    futuros = [[d]] * 9 + [[]]
    decision = best_today_decision([a], futuros, LUN, [MAR, MIE], TABLA, epsilon=10.0, max_segundos=5)
    assert decision == {"A": "esperar"}


def test_sin_nada_que_esperar_sale_directo_hoy():
    # Ningún futuro trae carga: esperar cuesta lo mismo y llega más tarde. Gana directo hoy
    decision = best_today_decision([_pedido("A", 2000, 1_500_000)], [[]] * 10, LUN, [MAR, MIE], TABLA,
                                   epsilon=10.0, max_segundos=5)
    assert decision == {"A": "directo"}


def test_si_hoy_la_carga_ya_paga_el_camion_sale_hoy():
    a, b = _pedido("A", 2000, 1_500_000), _pedido("B", 2000, 1_500_000)
    decision = best_today_decision([a, b], [[]] * 10, LUN, [MAR, MIE], TABLA, epsilon=10.0, max_segundos=5)
    assert decision == {"A": "camion", "B": "camion"}


def test_un_limite_de_retraso_estricto_impide_esperar():
    # Esperar ahorraría, pero el retraso promedio se pasaría de un ε muy bajo: sale directo
    a = _pedido("A", 2000, 1_500_000)
    d = _pedido("D", 2000, 1_500_000, llegada=MAR, ultimo_dia=MIE, dias_camion=(MAR, MIE))
    decision = best_today_decision([a], [[d]] * 5 + [[]] * 5, LUN, [MAR, MIE], TABLA, epsilon=0.6, max_segundos=5)
    assert decision["A"] != "esperar"


def test_solo_propone_decisiones_posibles():
    # Sin opción de camión hoy y venciendo hoy: la única opción es directo
    a = _pedido("A", 2000, 1_500_000, ultimo_dia=LUN, dias_camion=(MAR,))
    assert best_today_decision([a], [[]], LUN, [], TABLA, epsilon=10.0, max_segundos=5) == {"A": "directo"}


# ---------------------------------------------------------------- la política completa

def test_la_politica_decide_sobre_todos_los_conocidos():
    # Todos vencen hoy (sin ventana ni futuros): cada conocido recibe una decisión válida;
    # O_0 (sin camión) va directo
    instancia = pequena.instancia()
    tablas = policy_tables(pequena.CAPACIDADES, pequena.TARIFAS, pequena.NODOS, pequena.SEDE, peso_maximo=10_000)
    conocidos = _conocidos_pequenos().assign(ultimo_dia=pequena.LUN)
    politica = make_policy_c(pool={}, instancia=instancia, tiempos_ftl={}, espera_maxima=3, factor_servicio=1.0,
                             n_futuros=10, tablas=tablas, max_segundos=5)
    decision = politica(conocidos, pequena.LUN)
    assert set(decision) == {"C_0", "C_1", "T_0", "O_0"}
    assert set(decision.values()) <= {"directo", "camion", "esperar"}
    assert decision["O_0"] == "directo"


# ---------------------------------------------------------------- límite de servicio relativo por región (corrección A)
# Un ε global aplicado en cada región castigaba a las regiones cuyas transportadoras cumplen peor: para bajar el
# retraso compraba camiones que no se pagan. El límite de cada región y día es su propio servicio con todo directo.

def test_el_limite_es_el_servicio_de_los_conocidos_si_hoy_salen_directo():
    # A (castigo 1,0 directo hoy) y B (ya esperó 2 días: 3,0). Lo que pase con los pedidos futuros no entra
    a, b = _pedido("A", 2000, 1_500_000, llegada=MIE, ultimo_dia=JUE), _pedido("B", 2000, 1_500_000)
    d = _pedido("D", 2000, 1_500_000, llegada=JUE, ultimo_dia=JUE, dias_camion=(JUE,))
    assert direct_delay([a, b], [[d], []], MIE, [JUE], TABLA) == pytest.approx((1.0 + 3.0) / 2)


def test_lo_que_ya_espero_cuenta_su_espera_en_el_retraso_con_todo_directo():
    # A llegó el lunes y hoy es miércoles: si sale directo hoy, ya lleva 2 días de espera (castigo 3,0)
    assert direct_delay([_pedido("A", 2000, 1_500_000)], [[]], MIE, [], TABLA) == pytest.approx(3.0)


def test_con_el_limite_relativo_no_compra_camiones_que_no_se_pagan():
    # Región que con todo directo ya está por encima de un ε global: con el límite relativo, un camión de un solo
    # pedido (que no se paga) no se compra solo para bajar el retraso
    a = _pedido("A", 500, 100_000, ultimo_dia=LUN)
    limite = 1.0 * direct_delay([a], [[]], LUN, [], TABLA)
    assert best_today_decision([a], [[]], LUN, [], TABLA, epsilon=limite, max_segundos=5) == {"A": "directo"}


def test_si_la_regla_hace_esperar_a_los_futuros_no_compra_camiones_que_no_se_pagan():
    # Los futuros traen pedidos que R1 hace esperar (más retraso). Con el límite calculado con la misma regla,
    # mandar A directo cumple, y un camión de un solo pedido no se compra para compensar
    a = _pedido("A", 500, 100_000)
    d = _pedido("D", 500, 100_000, llegada=MAR, ultimo_dia=MIE, dias_camion=(MAR, MIE))
    futuros = [[d]] * 10
    limite = direct_delay([a], futuros, LUN, [MAR, MIE], TABLA)
    assert best_today_decision([a], futuros, LUN, [MAR, MIE], TABLA, epsilon=limite, max_segundos=5)["A"] != "camion"


def test_no_queda_atrapada_en_un_camion_que_no_se_paga():
    # Cinco pedidos pequeños que juntos no pagan una Turbo: aunque "todos al camión" baje el retraso, devolverlos uno
    # por uno sube el costo (el camión se paga igual). Con el movimiento "todos directo" sale de esa trampa
    pedidos = [_pedido(f"P{i}", 100, 50_000) for i in range(5)]
    limite = direct_delay(pedidos, [[]], LUN, [MAR, MIE], TABLA)
    decision = best_today_decision(pedidos, [[]], LUN, [MAR, MIE], TABLA, epsilon=limite, max_segundos=5)
    assert set(decision.values()) == {"directo"}
