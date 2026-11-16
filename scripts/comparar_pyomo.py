"""Comparación PuLP vs Pyomo con el mismo modelo multiobjetivo y los mismos datos.

Para cada librería: tiempo de armado del modelo, extremo de costo (min Z1) y un punto de la frontera
(min Z1 con Z2 <= retraso de la operación actual). Si el modelo está bien traducido, los costos coinciden
dentro del gap. Los resolvers de las dos librerías devuelven las mismas columnas.
Resultado en data/processed/comparacion_pulp_pyomo_{sede}_{mes}_E{espera}.csv
"""
import time

import pandas as pd

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.modelos import version_pyomo as modelo_pyomo
from despacho_multiobjetivo.metodos import frontera_pareto, resolver_pyomo

SOLVERS_PYOMO = ["highs"]     # cuando haya licencia: "gurobi_direct"
RUTA_RESULTADOS = PROCESSED / f"comparacion_pulp_pyomo_{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}.csv"


if __name__ == "__main__":
    inicio_total = time.perf_counter()
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)   # mismas entradas para las dos
    epsilon = servicio.expected_delay_direct(pedidos).mean()   # días de retraso ponderados por pedido de la operación actual
    corrida_epsilon = f"min Z1, Z2 <= {epsilon:,.3f}"
    filas = []

    # ---------------------------------------------------------------- PuLP
    print("\n=== PuLP: armando el modelo")
    inicio = time.perf_counter()
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                          nodos_regionales, SEDE)
    armado = time.perf_counter() - inicio
    print(f"PuLP armado en {armado:.1f} s · {len(problema.variables())} variables · {len(problema.constraints)} restricciones")
    print("PuLP: extremo de costo")
    filas.append({"libreria": "PuLP", "corrida": "min Z1", "segundos_armar": armado,
                  **frontera_pareto.cost_extreme(problema, z1, z2)})
    print(f"PuLP: {corrida_epsilon}")
    resultado = frontera_pareto.epsilon_point(problema, z1, z2, epsilon)
    resultado.pop("epsilon")
    filas.append({"libreria": "PuLP", "corrida": corrida_epsilon, "segundos_armar": armado, **resultado})

    # ---------------------------------------------------------------- Pyomo
    print("\n=== Pyomo: armando el modelo")
    inicio = time.perf_counter()
    modelo = modelo_pyomo.build_model(opciones, pedidos, capacidades, tarifas_ftl, nodos_regionales, SEDE)
    armado = time.perf_counter() - inicio
    print(f"Pyomo armado en {armado:.1f} s · {modelo.nvariables()} variables · {modelo.nconstraints()} restricciones")
    for solver in SOLVERS_PYOMO:
        print(f"Pyomo + {solver}: extremo de costo")
        filas.append({"libreria": "Pyomo", "corrida": "min Z1", "segundos_armar": armado,
                      **resolver_pyomo.cost_extreme(modelo, solver)})
        print(f"Pyomo + {solver}: {corrida_epsilon}")
        resultado = resolver_pyomo.epsilon_point(modelo, epsilon, solver)
        resultado.pop("epsilon")
        filas.append({"libreria": "Pyomo", "corrida": corrida_epsilon, "segundos_armar": armado, **resultado})

    # ---------------------------------------------------------------- comparación
    tabla = pd.DataFrame(filas)
    tabla["gap"] = (tabla["costo"] - tabla["cota"]) / tabla["costo"]
    referencia = tabla[tabla["libreria"] == "PuLP"].set_index("corrida")["costo"]
    tabla["dif_vs_pulp"] = tabla["costo"] / tabla["corrida"].map(referencia) - 1
    tabla.to_csv(RUTA_RESULTADOS, index=False)

    print("\n" + tabla.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\nResultados en {RUTA_RESULTADOS} ({(time.perf_counter() - inicio_total) / 60:.1f} min) · sede {SEDE} · {MES_INICIO[:7]}")
    print("Si dif_vs_pulp está dentro del gap (0,1 %), las dos librerías llegan al mismo plan.")
