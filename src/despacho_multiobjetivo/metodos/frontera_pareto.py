import time

import pulp
import pandas as pd

from despacho_multiobjetivo.config import GAP_REL_FRONTERA, LIMITE_SEGUNDOS_FRONTERA


def solve(problema: pulp.LpProblem, objetivo: pulp.LpAffineExpression, z1: pulp.LpAffineExpression,
          z2: pulp.LpAffineExpression, mostrar: bool = True, opciones_solver: dict | None = None) -> dict:
    # Fija qué se minimiza, resuelve y devuelve lo mismo que el resolver de Pyomo (metodos/resolver_pyomo.py):
    # estado de HiGHS, costo (z1), retraso (z2), cota del objetivo minimizado y tiempo de solución.
    # mostrar: imprime el registro del solver · opciones_solver: opciones de HiGHS, p. ej. {"random_seed": 3}
    problema.setObjective(objetivo)
    inicio = time.perf_counter()
    problema.solve(pulp.HiGHS(msg=mostrar, gapRel=GAP_REL_FRONTERA, timeLimit=LIMITE_SEGUNDOS_FRONTERA,
                              **(opciones_solver or {})))
    highs = problema.solverModel
    return {
        "solver": "highs",
        # PuLP marca "Optimal" también cuando HiGHS se detiene por tiempo: se usa el estado de HiGHS
        "estado": highs.modelStatusToString(highs.getModelStatus()),
        "costo": pulp.value(z1),
        "retraso": pulp.value(z2),
        "cota": highs.getInfo().mip_dual_bound,
        "segundos_resolver": time.perf_counter() - inicio,
    }

def cost_extreme(problema: pulp.LpProblem, z1: pulp.LpAffineExpression, z2: pulp.LpAffineExpression,
                 mostrar: bool = True, opciones_solver: dict | None = None) -> dict:
    return solve(problema, z1, z1, z2, mostrar, opciones_solver)

def delay_extreme(problema: pulp.LpProblem, z1: pulp.LpAffineExpression, z2: pulp.LpAffineExpression,
                  mostrar: bool = True, opciones_solver: dict | None = None) -> dict:
    # Lexicográfico: primero el menor retraso posible y, con ese retraso fijo, el menor costo
    retraso_minimo = solve(problema, z2, z1, z2, mostrar, opciones_solver)["retraso"]

    problema += z2 <= retraso_minimo * (1 + 1e-6), "fijar_retraso_minimo"
    resultado = solve(problema, z1, z1, z2, mostrar, opciones_solver)

    del problema.constraints["fijar_retraso_minimo"]
    return resultado

def epsilon_point(problema: pulp.LpProblem, z1: pulp.LpAffineExpression, z2: pulp.LpAffineExpression,
                  epsilon: float, mostrar: bool = True, opciones_solver: dict | None = None) -> dict:
    # Mínimo costo con un presupuesto de retraso: z2 <= epsilon (sin el premio por la holgura de la frontera)
    problema += z2 <= epsilon * (1 + 1e-6), "epsilon_point"
    resultado = solve(problema, z1, z1, z2, mostrar, opciones_solver)
    del problema.constraints["epsilon_point"]   # deja el modelo como estaba
    return {"epsilon": epsilon, **resultado}

def payoff_matrix(resultado_costo: dict, resultado_retraso: dict) -> pd.DataFrame:
    matriz = pd.DataFrame(
        {
            "costo": [resultado_costo["costo"], resultado_retraso["costo"]],
            "retraso": [resultado_costo["retraso"], resultado_retraso["retraso"]]
        },
        index=["Optimizar Z1 (costo)", "Optimizar Z2 (retraso)"]
    )
    matriz.loc["Ideal"] = matriz.min()
    matriz.loc["Nadir"] = matriz.max()
    return matriz

def pareto_frontier(problema: pulp.LpProblem, z1: pulp.LpAffineExpression, z2: pulp.LpAffineExpression,
                    matriz: pd.DataFrame, n_puntos: int = 10, epsilons: list | None = None, metricas=None,
             mostrar: bool = True) -> pd.DataFrame:
    # Método ε-restricción: minimizar el costo con un presupuesto de retraso (epsilon) que recorre
    # de forma pareja el rango entre el Ideal (mejor servicio) y el Nadir (plan más barato).
    # epsilons: si se pasa una lista, se usan esos presupuestos en lugar de la grilla pareja.
    # metricas: función sin argumentos que se llama después de cada punto; su diccionario se agrega a la fila
    ideal_retraso = matriz.loc["Ideal", "retraso"]
    rango_retraso = matriz.loc["Nadir", "retraso"] - ideal_retraso
    rango_costo = matriz.loc["Nadir", "costo"] - matriz.loc["Ideal", "costo"]
    if epsilons is None:
        epsilons = [ideal_retraso + rango_retraso * k / (n_puntos - 1) for k in range(n_puntos)]

    # a) Restricción epsilon: retraso + holgura = epsilon (la holgura mide cuánto sobra del presupuesto)
    holgura = pulp.LpVariable("holgura_retraso", lowBound=0)
    problema += z2 + holgura == matriz.loc["Nadir", "retraso"], "epsilon"

    # b) Objetivo aumentado: costo menos un premio pequeño por la holgura, para que entre
    #    dos planes del mismo costo gane el de menos retraso (Mavrotas, 2009)
    objetivo = z1 - 1e-3 * rango_costo * holgura / rango_retraso

    # c) Un modelo por cada presupuesto de retraso
    puntos = []
    for k, eps in enumerate(epsilons):
        problema.constraints["epsilon"].changeRHS(eps * (1 + 1e-6))   # margen numérico: el primer punto es justo el mínimo
        fila = {"epsilon": eps, **solve(problema, objetivo, z1, z2, mostrar)}
        if metricas:
            fila.update(metricas())
        puntos.append(fila)
        print(f"Punto {k + 1}/{len(epsilons)}: epsilon {eps:,.1f} | {fila['estado']} | costo {fila['costo']:,.0f} | retraso {fila['retraso']:,.1f}")

    del problema.constraints["epsilon"]   # deja el modelo como estaba
    return pd.DataFrame(puntos)
