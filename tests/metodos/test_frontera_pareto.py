"""Extremos, matriz de pagos, punto epsilon y frontera con un problema que se resuelve a mano.

Una sola decisión x entre 0 y 1: consolidar más (x → 1) baja el costo pero sube el retraso.
    Z1 = 10 − 5x (costo)      Z2 = 2x (retraso)
Más barato: x = 1 → (costo 5, retraso 2). Mejor servicio: x = 0 → (costo 10, retraso 0).
"""
import pulp
import pytest

from despacho_multiobjetivo.metodos import frontera_pareto as fp


def _problema():
    problema = pulp.LpProblem("toy", pulp.LpMinimize)
    x = pulp.LpVariable("x", lowBound=0, upBound=1)
    problema += x >= 0, "dominio"
    z1 = 10 - 5 * x
    z2 = 2 * x
    return problema, z1, z2


def test_resolver_devuelve_las_mismas_columnas_que_el_resolver_de_pyomo():
    problema, z1, z2 = _problema()
    resultado = fp.solve(problema, z1, z1, z2, mostrar=False)
    assert set(resultado) == {"solver", "estado", "costo", "retraso", "cota", "segundos_resolver"}
    assert resultado["estado"] == "Optimal"


def test_extremos_y_matriz_de_pagos():
    problema, z1, z2 = _problema()
    costo = fp.cost_extreme(problema, z1, z2, mostrar=False)
    retraso = fp.delay_extreme(problema, z1, z2, mostrar=False)
    assert (costo["costo"], costo["retraso"]) == pytest.approx((5, 2))
    assert (retraso["costo"], retraso["retraso"]) == pytest.approx((10, 0))

    matriz = fp.payoff_matrix(costo, retraso)
    assert matriz.loc["Ideal"].tolist() == pytest.approx([5, 0])    # lo mejor de cada objetivo
    assert matriz.loc["Nadir"].tolist() == pytest.approx([10, 2])   # lo peor de la matriz
    assert "fijar_retraso_minimo" not in problema.constraints        # el extremo de retraso deja el modelo como estaba


def test_punto_epsilon_es_el_mas_barato_que_no_supera_el_presupuesto():
    problema, z1, z2 = _problema()
    resultado = fp.epsilon_point(problema, z1, z2, epsilon=1, mostrar=False)
    assert resultado["epsilon"] == 1
    assert (resultado["costo"], resultado["retraso"]) == pytest.approx((7.5, 1))
    assert "epsilon_point" not in problema.constraints


def test_frontera_recorre_el_rango_de_forma_pareja():
    problema, z1, z2 = _problema()
    matriz = fp.payoff_matrix(fp.cost_extreme(problema, z1, z2, mostrar=False),
                              fp.delay_extreme(problema, z1, z2, mostrar=False))
    puntos = fp.pareto_frontier(problema, z1, z2, matriz, n_puntos=3, mostrar=False)
    assert puntos["epsilon"].tolist() == pytest.approx([0, 1, 2])
    assert puntos["costo"].tolist() == pytest.approx([10, 7.5, 5])
    assert list(problema.constraints) == ["dominio"]                 # la restricción epsilon se quita al final


def test_frontera_con_presupuestos_propios_y_metricas():
    problema, z1, z2 = _problema()
    matriz = fp.payoff_matrix({"costo": 5, "retraso": 2}, {"costo": 10, "retraso": 0})
    puntos = fp.pareto_frontier(problema, z1, z2, matriz, epsilons=[0.5], metricas=lambda: {"extra": 1}, mostrar=False)
    assert puntos[["epsilon", "costo", "extra"]].iloc[0].tolist() == pytest.approx([0.5, 8.75, 1])
