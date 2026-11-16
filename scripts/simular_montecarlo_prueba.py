"""Monte Carlo de PRUEBA con la política C (como el taller), sede 0111, enero de 2026.

Rotulado como prueba de funcionamiento: los futuros que imagina la política se sortean de los mismos días de enero
que se simulan (contaminación aceptada por el usuario el 2026-11-08); los resultados de la tesis saldrán del
histórico cotizado. Decisiones del usuario: servicio al 100 % del actual, castigo fuerte, N = 10 futuros, 10 s por
decisión diaria. Corrección A (2026-11-08): el límite de cada región y día es su propio retraso con todo directo
(factor 1,0), no el ε global, que obligaba a comprar camiones que no se pagan.

1. Enero real: la política decide día a día sin conocer el futuro; se compara con todo directo y con el MILP con el
   mes conocido (226,6 M en el punto 100 %).
2. Meses simulados: cada mes se arma sorteando, para cada día de enero, un día del histórico del mismo tramo y día
   de la semana (generar_demanda); se corre la política y todo directo, y se guarda una fila por mes.
Se puede retomar: los meses ya guardados no se repiten. Mientras corre, pide a Windows no suspender el equipo por
inactividad (no cambia ninguna configuración y se quita al terminar).
Uso: .venv/Scripts/python.exe scripts/simular_montecarlo_prueba.py [--meses 100] [--futuros 10] [--segundos 10]
Resultados: data/processed/montecarlo_prueba_C_0111_2026-01_E3_meses.csv y _registro_enero_real.csv
"""
import argparse
import ctypes
import os
import random
import time

import pandas as pd

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED, PREFIJO_FRONTERA
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import prepare_instance
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.simulacion.escenarios import build_day_pool, generar_demanda
from despacho_multiobjetivo.simulacion.politicas import make_policy_c, policy_tables
from despacho_multiobjetivo.simulacion.simulador import costo_escenario, decide_all_direct, simular_escenario

PREFIJO = PROCESSED / f"montecarlo_prueba_C_{SEDE}_{MES_INICIO[:7]}_E{ESPERA_MAXIMA}"


def keep_awake(activar: bool) -> None:
    # Pide a Windows no suspender por inactividad mientras corre (no aplica a cerrar la tapa)
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if activar else 0))


def run_month(pedidos_mes: pd.DataFrame, etiqueta: str, datos: dict, semilla: int, args) -> tuple:
    # Corre la política C y todo directo sobre un mes; devuelve (fila de resultados, registro de la política)
    inicio = time.perf_counter()
    opciones = opciones_salida.build_departure_options(pedidos_mes, datos["tiempos_ftl"], ESPERA_MAXIMA)
    instancia = prepare_instance(pedidos_mes, opciones, datos["capacidades"], datos["tarifas_ftl"],
                                 datos["nodos"], SEDE)
    politica = make_policy_c(datos["pool"], instancia, datos["tiempos_ftl"], ESPERA_MAXIMA, args.factor,
                             args.futuros, datos["tablas"], semilla=semilla, max_segundos=args.segundos)
    registro = simular_escenario(pedidos_mes, politica, ESPERA_MAXIMA, instancia)
    costo = costo_escenario(registro, pedidos_mes, instancia)
    directo = costo_escenario(simular_escenario(pedidos_mes, decide_all_direct, ESPERA_MAXIMA), pedidos_mes, instancia)
    en_camion = registro.loc[registro["modo"] == "camion", "fraccion"].sum()
    esperaron = (registro["dia_salida"] > registro["dia_llegada"]).sum()
    fila = {"mes": etiqueta, "pedidos": len(pedidos_mes), "costo_politica": costo["z1"], "z2_politica": costo["z2"],
            "costo_todo_directo": directo["z1"], "z2_todo_directo": directo["z2"],
            "ahorro": 1 - costo["z1"] / directo["z1"], "pedidos_en_camion": en_camion,
            "salidas_despues_de_esperar": int(esperaron), "minutos": (time.perf_counter() - inicio) / 60}
    return fila, registro


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo de prueba con la política C")
    parser.add_argument("--meses", type=int, default=100, help="meses simulados (M)")
    parser.add_argument("--futuros", type=int, default=10, help="futuros por decisión diaria (N)")
    parser.add_argument("--segundos", type=float, default=10, help="segundos por decisión diaria")
    parser.add_argument("--semilla", type=int, default=0)
    parser.add_argument("--factor", type=float, default=1.0, help="servicio de cada región: factor × todo directo")
    args = parser.parse_args()
    keep_awake(True)
    try:
        print("Cargando datos", flush=True)
        pedidos, nodos, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
        datos = {"pool": build_day_pool(pedidos), "tiempos_ftl": tiempos_ftl, "capacidades": capacidades,
                 "tarifas_ftl": tarifas_ftl, "nodos": nodos, "epsilon": servicio.expected_delay_direct(pedidos).mean(),
                 "tablas": policy_tables(capacidades, tarifas_ftl, nodos, SEDE)}
        print(f"servicio: {args.factor:.0%} del de todo directo en cada región (Z2 actual del mes {datos['epsilon']:.4f}) · "
              f"N = {args.futuros} · {args.segundos:.0f} s por día", flush=True)

        ruta_meses = PROCESSED / f"{PREFIJO.name}_meses.csv"
        hechos = pd.read_csv(ruta_meses) if ruta_meses.exists() else pd.DataFrame(columns=["mes"])
        filas = hechos.to_dict("records")

        # 1. Enero real
        if "enero_real" not in set(hechos["mes"]):
            print("Enero real con la política C", flush=True)
            fila, registro = run_month(pedidos, "enero_real", datos, args.semilla, args)
            registro.to_csv(PROCESSED / f"{PREFIJO.name}_registro_enero_real.csv", index=False)
            filas.append(fila)
            pd.DataFrame(filas).to_csv(ruta_meses, index=False)
            manuales = pd.read_csv(f"{PREFIJO_FRONTERA}_manuales.csv")
            milp = manuales.loc[(manuales["pct_retraso_actual"] - 1.0).abs().idxmin(), "costo"]
            print(f"   enero real: política C {fila['costo_politica']:,.0f} (Z2 {fila['z2_politica']:.4f}) · "
                  f"todo directo {fila['costo_todo_directo']:,.0f} · MILP con el mes conocido {milp:,.0f} · "
                  f"ahorro {fila['ahorro']:.2%} · {fila['minutos']:.1f} min", flush=True)

        # 2. Meses simulados con días de enero (mismo tramo y día de la semana)
        fechas = sorted(datos["pool"])
        meses = generar_demanda(datos["pool"], fechas, args.meses, random.Random(args.semilla + 1))
        for k, pedidos_mes in enumerate(meses):
            etiqueta = f"simulado_{k}"
            if etiqueta in set(pd.DataFrame(filas)["mes"]):
                continue
            fila, _ = run_month(pedidos_mes, etiqueta, datos, args.semilla + 100 + k, args)
            filas.append(fila)
            pd.DataFrame(filas).to_csv(ruta_meses, index=False)
            print(f"   mes simulado {k + 1}/{args.meses}: ahorro {fila['ahorro']:.2%} · Z2 {fila['z2_politica']:.4f} "
                  f"(todo directo {fila['z2_todo_directo']:.4f}) · {fila['minutos']:.1f} min", flush=True)
        print("Listo", flush=True)
    finally:
        keep_awake(False)
