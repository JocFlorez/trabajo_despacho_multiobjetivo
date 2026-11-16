"""Comparación justa de las metaheurísticas frente al MILP: búsqueda tabú, genético con operadores propios y genético
con los operadores genéricos de pymoo.

Las tres arrancan de las mismas semillas del MILP (óptimos al 92 % y al 115 % del retraso actual, vecinos a los puntos
de comparación) y se miden en los mismos puntos del codo (86, 100, 111 y 120 % del retraso actual), con las mismas
semillas aleatorias y 10 minutos por corrida.
  - genético propio: lo corre validar_genetico.py (aquí solo se lee su genetico_..._comparacion_s*.csv)
  - genético genérico (--parte generico): cruce SBX y mutación polinomial con redondeo, sin reparación; misma
    población inicial. Mide cuánto aportan el cruce por región y la reparación
  - tabú (--parte tabu): un punto por corrida, desde la semilla del MILP más cercana. Como el genético resuelve todos
    los puntos a la vez, la tabú se reporta con dos presupuestos tomados del mismo historial: 10 min por punto y
    2,5 min por punto (los mismos 10 min en total que el genético para los 4 puntos)
  - resumen (--parte resumen): junta todo y calcula mediana, mínimo y máximo de la brecha por método y punto
Corrección del 2 de noviembre: las corridas anteriores le daban al genético las semillas pasadas por el cromosoma
(traslados partidos completos en su día de mayor fracción, 4 a 8 % más caras) y a la tabú las originales. Por eso:
  - --parte gen0: repite los dos genéticos con semillas, con los traslados partidos en gen 0 (encode actual)
  - --parte tabu_codificada: tabú desde esa misma semilla codificada, 2,5 min por punto en 100, 111 y 120 %
  - --parte tabu_tenure5: tabú como la de --parte tabu, con 5 iteraciones de memoria en vez de 10
Las corridas ya guardadas no se repiten, así que las partes se pueden correr en paralelo o por tandas.
Resultados en data/processed/:
  genetico_generico_{sede}_{mes}_E{espera}_*      lo mismo que validar_genetico.py, con operadores genéricos
  tabu_justa_{sede}_{mes}_E{espera}_*             historial de cada corrida y resultado por semilla
  comparacion_metaheuristicas_{sede}_{mes}_E{espera}_resumen.csv / _corridas.csv
"""
import argparse
import re
import time

import pandas as pd

from despacho_multiobjetivo.config import (SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED,
                                           PREFIJO_FRONTERA)
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.metodos import frontera_pareto
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import plan_from_milp, prepare_instance
from despacho_multiobjetivo.metodos.metaheuristicas.genetico import (best_for_epsilon, chromosome_layout, decode,
                                                                    encode, run_nsga2)
from despacho_multiobjetivo.metodos.metaheuristicas.tabu import tabu_search

ESCENARIO = f"{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}"
PREFIJO_PROPIO = PROCESSED / f"genetico_{ESCENARIO}"
PREFIJO_GENERICO = PROCESSED / f"genetico_generico_{ESCENARIO}"
PREFIJO_TABU = PROCESSED / f"tabu_justa_{ESCENARIO}"
PREFIJO_RESUMEN = PROCESSED / f"comparacion_metaheuristicas_{ESCENARIO}"
PREFIJOS_GEN0 = {"propios": PROCESSED / f"genetico_gen0_{ESCENARIO}",
                 "genericos": PROCESSED / f"genetico_generico_gen0_{ESCENARIO}"}
PREFIJO_TABU_CODIFICADA = PROCESSED / f"tabu_codificada_{ESCENARIO}"
PREFIJO_TABU_TENURE5 = PROCESSED / f"tabu_tenure5_{ESCENARIO}"
FACTORES_SEMILLA = [0.92, 1.15]
PORCENTAJES = [0.86, 1.00, 1.11, 1.20]
SEGUNDOS = 600
SEGUNDOS_POR_PUNTO_MISMO_TOTAL = SEGUNDOS / len(PORCENTAJES)   # 150 s: el genético reparte 600 s entre 4 puntos


def milp_costs() -> dict:
    # Óptimo del MILP en cada punto de comparación (de frontera_puntos_manuales.py)
    manuales = pd.read_csv(f"{PREFIJO_FRONTERA}_manuales.csv")
    return {p: manuales.loc[(manuales["pct_retraso_actual"] - p).abs().idxmin(), "costo"] for p in PORCENTAJES}


def run_generic(instancia, retraso_actual, semillas_milp, semillas_aleatorias) -> None:
    # Genético con operadores genéricos, sin y con semillas del MILP, como validar_genetico.py
    costos = milp_costs()
    variantes = {"sin_semillas": [{}], "con_semillas": [{}] + semillas_milp}
    for semilla in semillas_aleatorias:
        ruta = f"{PREFIJO_GENERICO}_comparacion_s{semilla}.csv"
        if PROCESSED.joinpath(f"{PREFIJO_GENERICO.name}_comparacion_s{semilla}.csv").exists():
            print(f"Genético genérico, semilla {semilla}: ya corrida")
            continue
        resultados = {}
        for variante, semillas in variantes.items():
            print(f"Genético genérico · semilla {semilla} · {variante.replace('_', ' ')}")
            resultado = run_nsga2(instancia, semillas=semillas, max_segundos=SEGUNDOS, semilla=semilla,
                                  operadores="genericos")
            resultado["frente"][["z1", "z2"]].to_csv(f"{PREFIJO_GENERICO}_{variante}_s{semilla}_frente.csv", index=False)
            resultado["historial"].to_csv(f"{PREFIJO_GENERICO}_{variante}_s{semilla}_historial.csv", index=False)
            resultados[variante] = resultado
            print(f"   {resultado['generaciones']} generaciones · {len(resultado['frente'])} planes en el frente")
        filas = []
        for porcentaje in PORCENTAJES:
            fila = {"semilla": semilla, "punto": f"{porcentaje:.0%}", "costo_milp": costos[porcentaje]}
            for variante, resultado in resultados.items():
                mejor = best_for_epsilon(resultado, porcentaje * retraso_actual)
                fila[f"costo_{variante}"] = mejor["z1"] if mejor else None
                fila[f"brecha_{variante}"] = (mejor["z1"] - costos[porcentaje]) / costos[porcentaje] if mejor else None
            filas.append(fila)
        pd.DataFrame(filas).to_csv(ruta, index=False)


def run_tabu(instancia, retraso_actual, semillas_milp, semillas_aleatorias, prefijo=PREFIJO_TABU, tenure=10) -> None:
    # Tabú en cada punto desde la semilla del MILP más cercana; 600 s por corrida, y el mejor a los 150 s del historial
    costos = milp_costs()
    for semilla in semillas_aleatorias:
        ruta = PROCESSED / f"{prefijo.name}_s{semilla}.csv"
        if ruta.exists():
            print(f"Tabú, semilla {semilla}: ya corrida")
            continue
        filas = []
        for porcentaje in PORCENTAJES:
            k = min(range(len(FACTORES_SEMILLA)), key=lambda j: abs(FACTORES_SEMILLA[j] - porcentaje))
            print(f"Tabú · semilla {semilla} · punto {porcentaje:.0%} · desde el MILP al {FACTORES_SEMILLA[k]:.0%}")
            tabu = tabu_search(instancia, porcentaje * retraso_actual, plan_inicial=semillas_milp[k], semilla=semilla,
                               max_segundos=SEGUNDOS, tenure=tenure, mostrar=False)
            historial = tabu["historial"]
            historial.to_csv(f"{prefijo}_p{round(porcentaje * 100)}_s{semilla}_historial.csv", index=False)
            a_tiempo = historial.loc[historial["segundos"] <= SEGUNDOS_POR_PUNTO_MISMO_TOTAL, "z1_mejor"].dropna()
            costo_corto = a_tiempo.iloc[-1] if len(a_tiempo) else None
            filas.append({"semilla": semilla, "punto": f"{porcentaje:.0%}", "costo_milp": costos[porcentaje],
                          "arranque": f"{FACTORES_SEMILLA[k]:.0%}", "costo_600s": tabu["z1"],
                          "brecha_600s": (tabu["z1"] - costos[porcentaje]) / costos[porcentaje] if tabu["factible"] else None,
                          "costo_150s": costo_corto,
                          "brecha_150s": (costo_corto - costos[porcentaje]) / costos[porcentaje] if costo_corto else None,
                          "segundos": tabu["segundos"], "iteraciones": tabu["iteraciones"]})
            print(f"   {tabu['iteraciones']} iteraciones · {tabu['segundos']:.0f} s · costo {tabu['z1']}")
        pd.DataFrame(filas).to_csv(ruta, index=False)


def run_genetics_gen0(instancia, retraso_actual, semillas_milp, semillas_aleatorias) -> None:
    # Los dos genéticos con semillas del MILP, con los traslados partidos en gen 0
    costos = milp_costs()
    for semilla in semillas_aleatorias:
        for operadores, prefijo in PREFIJOS_GEN0.items():
            ruta = PROCESSED / f"{prefijo.name}_comparacion_s{semilla}.csv"
            if ruta.exists():
                print(f"Genético {operadores}, semilla {semilla}: ya corrida")
                continue
            print(f"Genético {operadores} · semilla {semilla} · con semillas, partidos en gen 0")
            resultado = run_nsga2(instancia, semillas=[{}] + semillas_milp, max_segundos=SEGUNDOS, semilla=semilla,
                                  operadores=operadores)
            resultado["frente"][["z1", "z2"]].to_csv(f"{prefijo}_s{semilla}_frente.csv", index=False)
            resultado["historial"].to_csv(f"{prefijo}_s{semilla}_historial.csv", index=False)
            print(f"   {resultado['generaciones']} generaciones · {len(resultado['frente'])} planes en el frente")
            filas = []
            for porcentaje in PORCENTAJES:
                mejor = best_for_epsilon(resultado, porcentaje * retraso_actual)
                filas.append({"semilla": semilla, "punto": f"{porcentaje:.0%}", "costo_milp": costos[porcentaje],
                              "costo": mejor["z1"] if mejor else None,
                              "brecha": (mejor["z1"] - costos[porcentaje]) / costos[porcentaje] if mejor else None})
            pd.DataFrame(filas).to_csv(ruta, index=False)


def run_tabu_coded(instancia, retraso_actual, semillas_milp, semillas_aleatorias) -> None:
    # Tabú desde la misma semilla que recibe el genético (pasada por el cromosoma), con el mismo total de 10 min:
    # mide qué parte del resultado de la tabú es búsqueda y qué parte viene heredada de la semilla con fracciones
    costos = milp_costs()
    layout = chromosome_layout(instancia)
    codificadas = [decode(encode(plan, layout), layout) for plan in semillas_milp]
    for semilla in semillas_aleatorias:
        ruta = PROCESSED / f"{PREFIJO_TABU_CODIFICADA.name}_s{semilla}.csv"
        if ruta.exists():
            print(f"Tabú desde semilla codificada, semilla {semilla}: ya corrida")
            continue
        filas = []
        for porcentaje in PORCENTAJES[1:]:
            k = min(range(len(FACTORES_SEMILLA)), key=lambda j: abs(FACTORES_SEMILLA[j] - porcentaje))
            print(f"Tabú codificada · semilla {semilla} · punto {porcentaje:.0%} · desde {FACTORES_SEMILLA[k]:.0%}")
            tabu = tabu_search(instancia, porcentaje * retraso_actual, plan_inicial=codificadas[k], semilla=semilla,
                               max_segundos=SEGUNDOS_POR_PUNTO_MISMO_TOTAL, mostrar=False)
            tabu["historial"].to_csv(f"{PREFIJO_TABU_CODIFICADA}_p{round(porcentaje * 100)}_s{semilla}_historial.csv",
                                     index=False)
            filas.append({"semilla": semilla, "punto": f"{porcentaje:.0%}", "costo_milp": costos[porcentaje],
                          "arranque": f"{FACTORES_SEMILLA[k]:.0%}", "costo": tabu["z1"],
                          "brecha": (tabu["z1"] - costos[porcentaje]) / costos[porcentaje] if tabu["factible"] else None,
                          "segundos": tabu["segundos"], "iteraciones": tabu["iteraciones"]})
            print(f"   {tabu['iteraciones']} iteraciones · {tabu['segundos']:.0f} s · costo {tabu['z1']}")
        pd.DataFrame(filas).to_csv(ruta, index=False)


def read_runs(patron: str, columnas: dict) -> list:
    # Lee los resultados por semilla de un método y los pasa a formato largo: método, semilla, punto, brecha
    filas = []
    for archivo in sorted(PROCESSED.glob(patron)):
        tabla = pd.read_csv(archivo)
        if "semilla" not in tabla:
            tabla.insert(0, "semilla", int(re.search(r"_s(\d+)\.csv$", archivo.name).group(1)))
        for columna, metodo in columnas.items():
            filas += [{"metodo": metodo, "semilla": f.semilla, "punto": f.punto, "brecha": getattr(f, columna)}
                      for f in tabla.itertuples()]
    return filas


def summarize() -> pd.DataFrame:
    filas = (read_runs(f"{PREFIJO_PROPIO.name}_comparacion_s*.csv",
                       {"brecha_sin_semillas": "genético propio sin semillas MILP",
                        "brecha_con_semillas": "genético propio con semillas MILP (día de mayor fracción)"})
             + read_runs(f"{PREFIJO_GENERICO.name}_comparacion_s*.csv",
                         {"brecha_sin_semillas": "genético genérico sin semillas MILP",
                          "brecha_con_semillas": "genético genérico con semillas MILP (día de mayor fracción)"})
             + read_runs(f"{PREFIJO_TABU.name}_s*.csv",
                         {"brecha_600s": "tabú con semillas MILP (10 min por punto)",
                          "brecha_150s": "tabú con semillas MILP (2,5 min por punto)"})
             + read_runs(f"{PREFIJOS_GEN0['propios'].name}_comparacion_s*.csv",
                         {"brecha": "genético propio con semillas MILP (partidos en gen 0)"})
             + read_runs(f"{PREFIJOS_GEN0['genericos'].name}_comparacion_s*.csv",
                         {"brecha": "genético genérico con semillas MILP (partidos en gen 0)"})
             + read_runs(f"{PREFIJO_TABU_TENURE5.name}_s*.csv",
                         {"brecha_600s": "tabú con tenure 5 (10 min por punto)",
                          "brecha_150s": "tabú con tenure 5 (2,5 min por punto)"})
             + read_runs(f"{PREFIJO_TABU_CODIFICADA.name}_s*.csv",
                         {"brecha": "tabú desde semilla codificada (2,5 min por punto)"}))
    corridas = pd.DataFrame(filas)
    corridas.to_csv(f"{PREFIJO_RESUMEN}_corridas.csv", index=False)
    resumen = (corridas.groupby(["metodo", "punto"])["brecha"]
               .agg(semillas="count", mediana="median", minimo="min", maximo="max").reset_index())
    resumen.to_csv(f"{PREFIJO_RESUMEN}_resumen.csv", index=False)
    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparación justa de tabú y genéticos frente al MILP")
    parser.add_argument("--parte", choices=["generico", "tabu", "gen0", "tabu_codificada", "tabu_tenure5", "resumen"], required=True)
    parser.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    inicio = time.perf_counter()

    if args.parte != "resumen":
        print("Cargando datos")
        pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
        opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
        instancia = prepare_instance(pedidos, opciones, capacidades, tarifas_ftl, nodos_regionales, SEDE)
        retraso_actual = servicio.expected_delay_direct(pedidos).mean()

        print("MILP de los puntos semilla (los mismos de validar_genetico.py)")
        problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                                    nodos_regionales, SEDE)
        semillas_milp = []
        for factor in FACTORES_SEMILLA:
            frontera_pareto.epsilon_point(problema, z1, z2, factor * retraso_actual, mostrar=False)
            semillas_milp.append(plan_from_milp(x))   # se toma enseguida: la próxima resolución cambia x

        corridas = {"generico": run_generic, "tabu": run_tabu, "gen0": run_genetics_gen0,
                    "tabu_codificada": run_tabu_coded,
                    "tabu_tenure5": lambda *datos: run_tabu(*datos, prefijo=PREFIJO_TABU_TENURE5, tenure=5)}
        corridas[args.parte](instancia, retraso_actual, semillas_milp, args.semillas)

    resumen = summarize()
    pd.set_option("display.width", 220)
    print("\nBrecha frente al MILP (mediana, mínimo y máximo entre semillas)")
    print(resumen.to_string(index=False, float_format=lambda v: f"{v:.2%}"))
    print(f"\n{(time.perf_counter() - inicio) / 60:.1f} min · resumen en {PREFIJO_RESUMEN}_resumen.csv")
