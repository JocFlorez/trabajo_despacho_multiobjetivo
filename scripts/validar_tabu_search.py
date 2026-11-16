"""Búsqueda tabú con datos reales en un punto de la frontera, frente al óptimo del MILP.

ε = castigo por retraso de la operación actual. La tabú arranca desde cada plan de ARRANQUES:
  - el óptimo del MILP en un ε vecino (factor × ε): la "semilla" acordada en la reunión del 7 de octubre;
    con 97 % el plan cumple pero es más caro (la tabú debe bajar costo) y con 103 % es más barato pero se pasa
    del límite (la tabú debe recuperar servicio). Vecinos cercanos, como en una frontera con grilla fina: mover
    un pedido cambia Z2 en ~0,0003 y con vecinos lejanos (86 % y 111 %) no alcanzan los movimientos en 10 min
  - o todo directo (factor None): mide qué tan bien busca sin ayuda del MILP
Sirve para medir la brecha frente al MILP, cuántas iteraciones alcanzan en el tiempo y si el castigo oscila
alrededor del límite.
Resultados en data/processed/:
  tabu_{sede}_{mes}_E{espera}_{arranque}_historial.csv y .png   historial y curva de convergencia de cada arranque
  tabu_{sede}_{mes}_E{espera}_resumen.csv                        una fila por método (MILP y cada arranque)
"""
import time

import pandas as pd

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.metodos import frontera_pareto
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import evaluate_plan, plan_from_milp, prepare_instance
from despacho_multiobjetivo.metodos.metaheuristicas.tabu import tabu_search
from despacho_multiobjetivo.reportes.metaheuristicas import plot_convergence

PREFIJO = PROCESSED / f"tabu_{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}"
SEMILLA = 0
# Arranques: {nombre: factor del ε vecino cuyo óptimo del MILP es el plan inicial}; None = todo directo
# (la corrida desde todo directo del 18 de octubre quedó en tabu_{sede}_{mes}_E{espera}_historial.csv)
ARRANQUES = {"milp097": 0.97, "milp103": 1.03}


if __name__ == "__main__":
    # 1. Datos y opciones de salida
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)

    # 2. ε = castigo promedio por pedido de la operación actual
    epsilon = servicio.expected_delay_direct(pedidos).mean()

    # 3. Referencia: el óptimo del MILP en el mismo ε
    print(f"MILP: mínimo costo con Z2 <= {epsilon:.4f}")
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                                nodos_regionales, SEDE)
    milp = frontera_pareto.epsilon_point(problema, z1, z2, epsilon, mostrar=False)
    print(f"   {milp['estado']} en {milp['segundos_resolver']:.1f} s · costo {milp['costo']:,.0f} · Z2 {milp['retraso']:.4f}")
    filas = [{"metodo": "MILP", "z1": milp["costo"], "z2": milp["retraso"], "segundos": milp["segundos_resolver"]}]

    # 4. Instancia del evaluador (una sola vez)
    inicio = time.perf_counter()
    instancia = prepare_instance(pedidos, opciones, capacidades, tarifas_ftl, nodos_regionales, SEDE)
    print(f"Instancia preparada en {time.perf_counter() - inicio:.1f} s")

    for nombre, factor in ARRANQUES.items():
        # 5. Plan inicial: el óptimo del MILP en el ε vecino, o todo directo
        if factor is None:
            descripcion, plan_inicial = "todo directo", {}
        else:
            descripcion = f"el MILP con epsilon al {factor:.0%} ({factor * epsilon:.4f})"   # sin ε: la consola de Windows no la imprime
            vecino = frontera_pareto.epsilon_point(problema, z1, z2, factor * epsilon, mostrar=False)
            plan_inicial = plan_from_milp(x)   # se toma enseguida: la próxima resolución cambia x
            print(f"\nMILP vecino: {vecino['estado']} en {vecino['segundos_resolver']:.1f} s · costo {vecino['costo']:,.0f}")
        inicial = evaluate_plan(plan_inicial, instancia)
        print(f"Búsqueda tabú desde {descripcion} (semilla {SEMILLA}): arranca con costo {inicial['z1']:,.0f} · "
              f"Z2 {inicial['z2']:.4f} ({'cumple' if inicial['z2'] <= epsilon + 1e-6 else 'se pasa del límite'})")

        # 6. Tabú
        tabu = tabu_search(instancia, epsilon, plan_inicial=plan_inicial, semilla=SEMILLA)
        historial = tabu["historial"]
        brecha = (tabu["z1"] - milp["costo"]) / milp["costo"] if tabu["factible"] else None
        if tabu["factible"]:
            print(f"   Tabú: costo {tabu['z1']:,.0f} · Z2 {tabu['z2']:.4f} · {tabu['segundos']:.0f} s · brecha {brecha:.2%}")
        else:
            print("   Tabú: no encontró un plan que cumpla el límite")
        print(f"   {tabu['iteraciones']} iteraciones · {tabu['evaluaciones']:,} evaluaciones · "
              f"{historial['cumple'].mean():.0%} de las iteraciones cumplen el límite")
        filas.append({"metodo": f"tabú desde {descripcion}", "z1": tabu["z1"], "z2": tabu["z2"],
                      "segundos": tabu["segundos"], "brecha": brecha, "z1_inicial": inicial["z1"],
                      "z2_inicial": inicial["z2"], "iteraciones": tabu["iteraciones"],
                      "evaluaciones": tabu["evaluaciones"], "pct_cumple": historial["cumple"].mean()})

        # 7. Historial y curva de convergencia
        ruta = PREFIJO.with_name(f"{PREFIJO.name}_{nombre}_historial.csv")
        historial.to_csv(ruta, index=False)
        plot_convergence(historial, epsilon, milp["costo"],
                         f"Búsqueda tabú · {SEDE} {MES_INICIO[:7]} · ε = {epsilon:.3f} · desde {descripcion}",
                         str(ruta.with_suffix(".png")))
        print(f"   Historial y curva en {ruta.with_suffix('')}.csv/.png")

    # 8. Resumen
    resumen = pd.DataFrame(filas)
    ruta_resumen = PREFIJO.with_name(f"{PREFIJO.name}_resumen.csv")
    resumen.to_csv(ruta_resumen, index=False)
    pd.set_option("display.width", 200)
    print("\n" + resumen.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\nResumen en {ruta_resumen}")
