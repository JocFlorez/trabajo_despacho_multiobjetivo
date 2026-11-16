"""Algoritmo genético NSGA-II con datos reales, frente al MILP y a la búsqueda tabú.

Corre dos variantes con el mismo tiempo y la misma población:
  - sin semillas del MILP: la población inicial solo trae "todo directo" y planes aleatorios
  - con semillas del MILP: además, los óptimos del MILP en dos puntos vecinos (92 % y 115 % del retraso actual),
    que no son los puntos de comparación (si no, el genético los tendría desde el inicio)
y compara el plan más barato de cada frente en los puntos del codo (86, 100, 111 y 120 % del retraso actual) con el
óptimo del MILP de esos puntos (frontera_..._manuales.csv, de frontera_puntos_manuales.py).
Se repite con varias semillas aleatorias (--semillas 0 1 2 …) para saber si el resultado depende de la suerte. Las
semillas ya corridas no se repiten: el resumen junta todas las que estén en data/processed.
Resultados en data/processed/:
  genetico_{sede}_{mes}_E{espera}_{variante}_s{semilla}_frente.csv / _historial.csv   frente y curva de cada corrida
  genetico_{sede}_{mes}_E{espera}_comparacion_s{semilla}.csv                          brecha en cada punto, por semilla
  genetico_{sede}_{mes}_E{espera}_frentes_s{semilla}.png                              frentes y frontera del MILP
  genetico_{sede}_{mes}_E{espera}_resumen_semillas.csv                                mediana, mínimo y máximo de la brecha
"""
import argparse
import re
import time

import matplotlib
matplotlib.use("Agg")   # guarda la imagen sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd

from despacho_multiobjetivo.config import (SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED,
                                           PREFIJO_FRONTERA)
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.metodos import frontera_pareto
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import evaluate_plan, plan_from_milp, prepare_instance
from despacho_multiobjetivo.metodos.metaheuristicas.genetico import best_for_epsilon, run_nsga2

PREFIJO = PROCESSED / f"genetico_{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}"
FACTORES_SEMILLA = [0.92, 1.15]                # puntos vecinos del MILP que siembran la variante "con semillas"
PORCENTAJES_COMPARACION = [0.86, 1.00, 1.11, 1.20]


def milp_frontier() -> pd.DataFrame:
    # Frontera del MILP (puntos de main_multiobjetivo + puntos del codo), sin puntos dominados, para la gráfica
    puntos = pd.concat([pd.read_csv(f"{PREFIJO_FRONTERA}_puntos.csv"), pd.read_csv(f"{PREFIJO_FRONTERA}_manuales.csv")])
    puntos = puntos.sort_values("retraso")
    no_dominado = [not ((puntos["retraso"] <= r) & (puntos["costo"] < c - 1)).any()
                   for r, c in zip(puntos["retraso"], puntos["costo"])]
    return puntos[no_dominado]


def plot_fronts(frentes: dict, frontera: pd.DataFrame, actual: dict, ruta: str) -> None:
    # Frentes del genético frente a la frontera del MILP, en la zona que importa (alrededor del servicio actual)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(frontera["retraso"], frontera["costo"] / 1e6, "o-", color="darkred", label="MILP (óptimo de cada punto)")
    colores = {"sin_semillas": "steelblue", "con_semillas": "seagreen"}
    nombres = {"sin_semillas": "Genético sin semillas del MILP", "con_semillas": "Genético con semillas del MILP"}
    for variante, frente in frentes.items():
        ax.scatter(frente["z2"], frente["z1"] / 1e6, s=14, color=colores[variante], label=nombres[variante])
    ax.scatter(actual["retraso"], actual["costo"] / 1e6, marker="s", s=120, color="gray",
               label="Operación actual (todo directo)", zorder=5)
    ax.set_xlim(0.7, 2.5)
    ax.set_ylim(200, 300)
    ax.set_xlabel("Días de retraso ponderados por pedido   (← mejor servicio  ·  peor servicio →)", fontsize=12)
    ax.set_ylabel("Costo del mes (millones de pesos)", fontsize=12)
    ax.set_title(f"Genético NSGA-II frente al MILP · {SEDE} {MES_INICIO[:7]} · E = {ESPERA_MAXIMA}", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSGA-II con datos reales frente al MILP")
    parser.add_argument("--segundos", type=float, default=600, help="tiempo de cada variante (por defecto 600)")
    parser.add_argument("--poblacion", type=int, default=100, help="individuos por generación (por defecto 100)")
    parser.add_argument("--semillas", type=int, nargs="+", default=[0], help="semillas aleatorias a correr")
    args = parser.parse_args()
    inicio_total = time.perf_counter()

    # 1. Datos, instancia del evaluador y retraso actual
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
    instancia = prepare_instance(pedidos, opciones, capacidades, tarifas_ftl, nodos_regionales, SEDE)
    retraso_actual = servicio.expected_delay_direct(pedidos).mean()
    actual = evaluate_plan({}, instancia)

    # 2. Semillas: óptimos del MILP en dos puntos vecinos a los de comparación
    print("MILP de los puntos semilla")
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                                nodos_regionales, SEDE)
    semillas_milp = []
    for factor in FACTORES_SEMILLA:
        punto = frontera_pareto.epsilon_point(problema, z1, z2, factor * retraso_actual, mostrar=False)
        semillas_milp.append(plan_from_milp(x))   # se toma enseguida: la próxima resolución cambia x
        print(f"   {factor:.0%}: {punto['estado']} en {punto['segundos_resolver']:.0f} s · costo {punto['costo']:,.0f}")

    # 3. Las dos variantes del genético con cada semilla, con el mismo tiempo y la misma población
    variantes = {"sin_semillas": [{}], "con_semillas": [{}] + semillas_milp}
    manuales = pd.read_csv(f"{PREFIJO_FRONTERA}_manuales.csv")
    for semilla in args.semillas:
        resultados = {}
        for variante, semillas in variantes.items():
            print(f"\nSemilla {semilla} · NSGA-II {variante.replace('_', ' ')} · población {args.poblacion} · "
                  f"{args.segundos:.0f} s")
            resultado = run_nsga2(instancia, semillas=semillas, pop_size=args.poblacion, max_segundos=args.segundos,
                                  semilla=semilla)
            resultados[variante] = resultado
            resultado["frente"][["z1", "z2"]].to_csv(f"{PREFIJO}_{variante}_s{semilla}_frente.csv", index=False)
            resultado["historial"].to_csv(f"{PREFIJO}_{variante}_s{semilla}_historial.csv", index=False)
            print(f"   {resultado['generaciones']} generaciones · {len(resultado['frente'])} planes en el frente")

        # 4. Comparación en los puntos del codo contra el MILP
        filas = []
        for porcentaje in PORCENTAJES_COMPARACION:
            epsilon = porcentaje * retraso_actual
            milp = manuales.loc[(manuales["pct_retraso_actual"] - porcentaje).abs().idxmin()]
            fila = {"semilla": semilla, "punto": f"{porcentaje:.0%}", "epsilon": epsilon, "costo_milp": milp["costo"]}
            for variante, resultado in resultados.items():
                mejor = best_for_epsilon(resultado, epsilon)
                fila[f"costo_{variante}"] = mejor["z1"] if mejor else None
                fila[f"brecha_{variante}"] = (mejor["z1"] - milp["costo"]) / milp["costo"] if mejor else None
            filas.append(fila)
        comparacion = pd.DataFrame(filas)
        comparacion.to_csv(f"{PREFIJO}_comparacion_s{semilla}.csv", index=False)
        print(comparacion.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

        # 5. Gráfica de los frentes de esta semilla
        plot_fronts({v: r["frente"] for v, r in resultados.items()}, milp_frontier(),
                    {"costo": actual["z1"], "retraso": actual["z2"]}, f"{PREFIJO}_frentes_s{semilla}.png")

    # 6. Resumen con todas las semillas corridas (las de esta ejecución y las anteriores)
    tablas = []
    for archivo in sorted(PROCESSED.glob(f"{PREFIJO.name}_comparacion_s*.csv")):
        tabla = pd.read_csv(archivo)
        if "semilla" not in tabla:
            tabla.insert(0, "semilla", int(re.search(r"_s(\d+)\.csv$", archivo.name).group(1)))
        tablas.append(tabla)
    todas = pd.concat(tablas, ignore_index=True)
    largo = todas.melt(id_vars=["semilla", "punto"], value_vars=["brecha_sin_semillas", "brecha_con_semillas"],
                       var_name="variante", value_name="brecha")
    largo["variante"] = largo["variante"].str.replace("brecha_", "")
    resumen = (largo.groupby(["variante", "punto"])["brecha"]
               .agg(semillas="count", mediana="median", minimo="min", maximo="max").reset_index())
    resumen.to_csv(f"{PREFIJO}_resumen_semillas.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nResumen de la brecha frente al MILP (semillas: " + ", ".join(map(str, sorted(todas["semilla"].unique()))) + ")")
    print(resumen.to_string(index=False, float_format=lambda v: f"{v:.2%}"))
    print(f"\nResultados en {PREFIJO}_*.csv ({(time.perf_counter() - inicio_total) / 60:.1f} min)")
