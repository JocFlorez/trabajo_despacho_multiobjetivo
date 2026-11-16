import pulp
import pytest

from despacho_multiobjetivo.metodos import solver


def test_resolver_devuelve_el_problema_si_es_optimo():
    problema = pulp.LpProblem("factible", pulp.LpMinimize)
    v = pulp.LpVariable("v", lowBound=2)
    problema += v
    resultado = solver.solve(problema)
    assert pulp.LpStatus[resultado.status] == "Optimal"
    assert v.varValue == 2


def test_resolver_falla_si_no_hay_solucion_optima():
    problema = pulp.LpProblem("infactible", pulp.LpMinimize)
    v = pulp.LpVariable("v", lowBound=0)
    problema += v
    problema += v >= 2
    problema += v <= 1
    with pytest.raises(ValueError, match="no encontró una solución óptima"):
        solver.solve(problema)
