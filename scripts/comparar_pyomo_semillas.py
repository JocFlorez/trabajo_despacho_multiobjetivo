"""¿Pyomo es más rápido que PuLP o fue suerte del branch and bound? Repite la comparación con varias semillas.

El branch and bound de HiGHS depende del orden en que recorre el árbol; la semilla aleatoria (random_seed)
cambia ese recorrido sin cambiar el modelo. Con cada semilla se resuelven las mismas dos corridas de
comparar_pyomo.py (min Z1 y min Z1 con Z2 <= retraso actual) en PuLP y en Pyomo, alternando la librería que va primero.
Si una librería es más rápida en la mediana de todas las semillas, la diferencia no es azar.

Resultados (se guardan después de cada corrida, así no se pierden si se detiene con Ctrl+C):
  data/processed/comparacion_semillas_{sede}_{mes}_E{espera}.csv          una fila por corrida
  data/processed/comparacion_semillas_{sede}_{mes}_E{espera}_resumen.csv  mediana, mínimo y máximo por librería
"""
import time

import pandas as pd

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.modelos import version_pyomo as modelo_pyomo
from despacho_multiobjetivo.metodos import frontera_pareto, resolver_pyomo

SEMILLAS = [0, 1, 2, 3, 4]    # 0 es la semilla por defecto de HiGHS (la de comparar_pyomo.py)
PREFIJO = str(PROCESSED / f"comparacion_semillas_{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}")


def run_case(libreria: str, problema, z1, z2, modelo, semilla: int, epsilon: float | None) -> dict:
    # Misma corrida en cualquiera de las dos librerías: los resolvers devuelven las mismas columnas
    opciones = {"random_seed": semilla}
    if libreria == "PuLP":
        if epsilon is None:
            return frontera_pareto.cost_extreme(problema, z1, z2, mostrar=False, opciones_solver=opciones)
        resultado = frontera_pareto.epsilon_point(problema, z1, z2, epsilon, mostrar=False, opciones_solver=opciones)
    else:
        if epsilon is None:
            return resolver_pyomo.cost_extreme(modelo, mostrar=False, opciones_solver=opciones)
        resultado = resolver_pyomo.epsilon_point(modelo, epsilon, mostrar=False, opciones_solver=opciones)
    resultado.pop("epsilon")
    return resultado


def summarize(tabla: pd.DataFrame) -> pd.DataFrame:
    resumen = (tabla.groupby(["corrida", "libreria"])
               .agg(corridas=("segundos_resolver", "size"),
                    mediana_s=("segundos_resolver", "median"),
                    minimo_s=("segundos_resolver", "min"),
                    maximo_s=("segundos_resolver", "max"),
                    costo_mediano=("costo", "median"),
                    gap_mediano=("gap", "median"))
               .reset_index())
    return resumen


if __name__ == "__main__":
    inicio_total = time.perf_counter()
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
    epsilon = servicio.expected_delay_direct(pedidos).mean()   # días de retraso ponderados por pedido de la operación actual

    print("Armando los dos modelos")
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                          nodos_regionales, SEDE)
    modelo = modelo_pyomo.build_model(opciones, pedidos, capacidades, tarifas_ftl, nodos_regionales, SEDE)

    corridas = [("min Z1", None), (f"min Z1, Z2 <= {epsilon:,.3f}", epsilon)]
    total = len(SEMILLAS) * len(corridas) * 2
    filas = []
    for k, semilla in enumerate(SEMILLAS):
        # Se alterna qué librería va primero, para que ninguna se beneficie siempre del mismo estado del equipo
        librerias = ["PuLP", "Pyomo"] if k % 2 == 0 else ["Pyomo", "PuLP"]
        for nombre_corrida, eps in corridas:
            for libreria in librerias:
                resultado = run_case(libreria, problema, z1, z2, modelo, semilla, eps)
                filas.append({"semilla": semilla, "corrida": nombre_corrida, "libreria": libreria, **resultado})
                print(f"[{len(filas)}/{total}] semilla {semilla} | {nombre_corrida} | {libreria:5} | "
                      f"{resultado['segundos_resolver']:6.1f} s | costo {resultado['costo']:,.0f} | {resultado['estado']}")

                tabla = pd.DataFrame(filas)
                tabla["gap"] = (tabla["costo"] - tabla["cota"]) / tabla["costo"]
                tabla.to_csv(f"{PREFIJO}.csv", index=False)
                summarize(tabla).to_csv(f"{PREFIJO}_resumen.csv", index=False)

    resumen = summarize(tabla)
    print("\n" + resumen.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    for nombre_corrida, _ in corridas:
        mediana = resumen[resumen["corrida"] == nombre_corrida].set_index("libreria")["mediana_s"]
        print(f"{nombre_corrida}: mediana PuLP {mediana['PuLP']:.1f} s vs Pyomo {mediana['Pyomo']:.1f} s "
              f"→ PuLP / Pyomo = {mediana['PuLP'] / mediana['Pyomo']:.2f}")
    print(f"\nResultados en {PREFIJO}.csv y {PREFIJO}_resumen.csv ({(time.perf_counter() - inicio_total) / 60:.1f} min)")
