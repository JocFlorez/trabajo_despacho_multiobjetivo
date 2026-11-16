"""Valida el evaluador de las metaheurísticas con los datos reales: debe dar el mismo Z1 y Z2 que el MILP.

Resuelve con el MILP el punto de la frontera con el castigo por retraso de la operación actual, convierte su
solución en un plan (plan_from_milp) y la evalúa con evaluate_plan, con y sin filtro de retraso en el relleno.
También mide cuánto tarda una evaluación, que es lo que decide cuántos planes puede probar una metaheurística.
Resultado en data/processed/validacion_evaluador_{sede}_{mes}_E{espera}.csv
"""
import time

import pandas as pd

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.metodos import frontera_pareto
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import evaluate_plan, plan_from_milp, prepare_instance

RUTA_RESULTADOS = PROCESSED / f"validacion_evaluador_{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}.csv"


if __name__ == "__main__":
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
    epsilon = servicio.expected_delay_direct(pedidos).mean()   # castigo por retraso de la operación actual

    print(f"MILP: mínimo costo con Z2 <= {epsilon:.3f} (castigo de la operación actual)")
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                                nodos_regionales, SEDE)
    milp = frontera_pareto.epsilon_point(problema, z1, z2, epsilon, mostrar=False)
    plan = plan_from_milp(x)
    partidos = sum(1 for variable in x.values() if 1e-6 < (variable.varValue or 0) < 1 - 1e-6)
    print(f"   {milp['estado']} en {milp['segundos_resolver']:.1f} s · costo {milp['costo']:,.0f} · Z2 {milp['retraso']:.4f}"
          f" · {len(plan)} pedidos completos en camión · {partidos} traslados partidos")

    print("Evaluador")
    inicio = time.perf_counter()
    instancia = prepare_instance(pedidos, opciones, capacidades, tarifas_ftl, nodos_regionales, SEDE)
    segundos_instancia = time.perf_counter() - inicio

    filas = [{"metodo": "MILP", "z1": milp["costo"], "z2": milp["retraso"], "segundos": milp["segundos_resolver"]}]
    for filtrar in (False, True):
        inicio = time.perf_counter()
        resultado = evaluate_plan(plan, instancia, filtrar_retraso=filtrar)
        segundos = time.perf_counter() - inicio
        filas.append({"metodo": f"evaluador (filtrar_retraso={filtrar})", "z1": resultado["z1"], "z2": resultado["z2"],
                      "segundos": segundos, "traslados_rellenados": len(resultado["relleno"])})

    tabla = pd.DataFrame(filas)
    tabla["dif_z1"] = tabla["z1"] - milp["costo"]
    tabla["dif_z1_pct"] = tabla["dif_z1"] / milp["costo"]
    tabla["dif_z2"] = tabla["z2"] - milp["retraso"]
    tabla.to_csv(RUTA_RESULTADOS, index=False)

    pd.set_option("display.width", 200)
    print("\n" + tabla.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))
    print(f"\nInstancia preparada en {segundos_instancia:.1f} s (una sola vez). Resultados en {RUTA_RESULTADOS}")
