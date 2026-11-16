"""Calibración del memético: 4 configuraciones con datos reales, semilla 0 y 10 minutos cada una.

Arranca de las mismas semillas del MILP que los genéticos (92 % y 115 %, traslados partidos en gen 0) y compara cada
configuración con el MILP en 86, 100, 111 y 120 % del retraso actual. La vara es el genético propio de
comparar_metaheuristicas.py --parte gen0.
Resultados en data/processed/:
  memetico_calibracion_{sede}_{mes}_E{espera}_{configuracion}_frente.csv / _historial.csv
  memetico_calibracion_{sede}_{mes}_E{espera}_resumen.csv
"""
import pandas as pd

from despacho_multiobjetivo.config import (SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED,
                                           PREFIJO_FRONTERA)
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.metodos import frontera_pareto
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import plan_from_milp, prepare_instance
from despacho_multiobjetivo.metodos.metaheuristicas.genetico import best_for_epsilon
from despacho_multiobjetivo.metodos.metaheuristicas.memetico import run_memetic
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio

PREFIJO = PROCESSED / f"memetico_calibracion_{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}"
FACTORES_SEMILLA = [0.92, 1.15]
PORCENTAJES = [0.86, 1.00, 1.11, 1.20]
CONFIGURACIONES = {"base_10_20_3": {"individuos": 10, "tamano_muestra": 20, "max_sin_mejora": 3},
                   "menos_individuos_5_20_3": {"individuos": 5, "tamano_muestra": 20, "max_sin_mejora": 3},
                   "menos_vecinos_10_10_3": {"individuos": 10, "tamano_muestra": 10, "max_sin_mejora": 3},
                   "sin_memoria_10_20_1": {"individuos": 10, "tamano_muestra": 20, "max_sin_mejora": 1}}


def calibrate(instancia: dict, semillas: list, retraso_actual: float, costos: dict, prefijo=PREFIJO,
              configuraciones: dict = CONFIGURACIONES, max_segundos: float = 600, semilla: int = 0) -> pd.DataFrame:
    # Corre cada configuración y devuelve una fila por configuración y punto; guarda frente e historial de cada una
    filas = []
    for nombre, parametros in configuraciones.items():
        print(f"Memético · {nombre} · {max_segundos:.0f} s")
        resultado = run_memetic(instancia, semillas=semillas, max_segundos=max_segundos, semilla=semilla, **parametros)
        resultado["frente"][["z1", "z2"]].to_csv(f"{prefijo}_{nombre}_frente.csv", index=False)
        resultado["historial"].to_csv(f"{prefijo}_{nombre}_historial.csv", index=False)
        print(f"   {resultado['generaciones']} generaciones · {len(resultado['frente'])} planes · {resultado['segundos']:.0f} s")
        for porcentaje in PORCENTAJES:
            mejor = best_for_epsilon(resultado, porcentaje * retraso_actual)
            costo = costos[porcentaje]
            filas.append({"configuracion": nombre, "generaciones": resultado["generaciones"],
                          "planes": len(resultado["frente"]), "punto": f"{porcentaje:.0%}", "costo_milp": costo,
                          "costo": mejor["z1"] if mejor else None,
                          "brecha": (mejor["z1"] - costo) / costo if mejor else None})
    resumen = pd.DataFrame(filas)
    resumen.to_csv(f"{prefijo}_resumen.csv", index=False)
    return resumen


if __name__ == "__main__":
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
    instancia = prepare_instance(pedidos, opciones, capacidades, tarifas_ftl, nodos_regionales, SEDE)
    retraso_actual = servicio.expected_delay_direct(pedidos).mean()

    print("MILP de los puntos semilla")
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                                nodos_regionales, SEDE)
    semillas_milp = []
    for factor in FACTORES_SEMILLA:
        frontera_pareto.epsilon_point(problema, z1, z2, factor * retraso_actual, mostrar=False)
        semillas_milp.append(plan_from_milp(x))   # se toma enseguida: la próxima resolución cambia x

    manuales = pd.read_csv(f"{PREFIJO_FRONTERA}_manuales.csv")
    costos = {p: manuales.loc[(manuales["pct_retraso_actual"] - p).abs().idxmin(), "costo"] for p in PORCENTAJES}
    resumen = calibrate(instancia, [{}] + semillas_milp, retraso_actual, costos)
    pd.set_option("display.width", 200)
    print(resumen.pivot(index="configuracion", columns="punto", values="brecha").to_string(float_format=lambda v: f"{v:.2%}"))
