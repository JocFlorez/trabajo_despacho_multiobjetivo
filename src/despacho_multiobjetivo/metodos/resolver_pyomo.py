"""Resolver el modelo de Pyomo con distintos solvers, con los mismos criterios que la versión en PuLP."""
import time

import pyomo.environ as pyo
from pyomo.contrib.solver.common.factory import SolverFactory

from despacho_multiobjetivo.modelos.version_pyomo import set_objective
from despacho_multiobjetivo.config import GAP_REL_FRONTERA, LIMITE_SEGUNDOS_FRONTERA


def solve(modelo: pyo.ConcreteModel, objetivo, solver: str = "highs", mostrar: bool = True,
          opciones_solver: dict | None = None) -> dict:
    # Fija qué se minimiza, resuelve y devuelve estado, valores de z1 y z2, cota del solver y tiempo
    # opciones_solver: opciones propias del solver, p. ej. {"random_seed": 3} en HiGHS
    set_objective(modelo, objetivo)
    inicio = time.perf_counter()
    resultado = SolverFactory(solver).solve(modelo, tee=mostrar, rel_gap=GAP_REL_FRONTERA, time_limit=LIMITE_SEGUNDOS_FRONTERA,
                                            solver_options=opciones_solver or {})
    return {
        "solver": solver,
        "estado": resultado.termination_condition.name,
        "costo": pyo.value(modelo.z1),
        "retraso": pyo.value(modelo.z2),
        "cota": resultado.objective_bound,
        "segundos_resolver": time.perf_counter() - inicio,
    }


def cost_extreme(modelo: pyo.ConcreteModel, solver: str = "highs", mostrar: bool = True,
                 opciones_solver: dict | None = None) -> dict:
    return solve(modelo, modelo.z1, solver, mostrar, opciones_solver)


def epsilon_point(modelo: pyo.ConcreteModel, epsilon: float, solver: str = "highs", mostrar: bool = True,
                  opciones_solver: dict | None = None) -> dict:
    # Mínimo costo con un presupuesto de retraso: z2 <= epsilon (restricción epsilon pura, sin premio por la holgura)
    if hasattr(modelo, "epsilon"):
        modelo.del_component(modelo.epsilon)
    modelo.epsilon = pyo.Constraint(expr=modelo.z2 <= epsilon * (1 + 1e-6))
    resultado = solve(modelo, modelo.z1, solver, mostrar, opciones_solver)
    modelo.del_component(modelo.epsilon)   # deja el modelo como estaba
    return {"epsilon": epsilon, **resultado}
