"""Valida el simulador de horizonte rodante con el mes real: la política "todo directo" debe reproducir la referencia
(en 0111-enero, modo histórico: el costo pagado y el retraso de la operación actual).

Recorre el mes con simular_escenario y decide_all_direct, calcula lo ejecutado con costo_escenario y revisa:
  - costo (Z1) y días de retraso ponderados (Z2) iguales a los de todo directo
  - cada pedido con fecha sale completo: sus fracciones suman 1 (ninguno se pierde ni se repite)
  - todo sale directo el mismo día en que llega
La referencia depende de MODO (config.py): en modo histórico es lo pagado (la operación actual); en modo
operación, todo directo con el cotizador.
Uso: .venv/Scripts/python.exe scripts/validar_simulador.py
"""
import sys
import time

import pandas as pd

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import prepare_instance
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.simulacion.simulador import costo_escenario, decide_all_direct, simular_escenario

TOLERANCIA = 1e-6


def reference_name(modo: str) -> str:
    # Contra qué se compara: el costo directo sale de la carga de datos según el modo
    return "operación actual (lo pagado)" if modo == "historico" else "todo directo cotizado"


def validate(pedidos: pd.DataFrame, instancia: dict, espera_maxima: float) -> list:
    # Corre el simulador con todo directo y devuelve una fila por revisión: revision, esperado, obtenido, cumple
    registro = simular_escenario(pedidos, decide_all_direct, espera_maxima)
    simulado = costo_escenario(registro, pedidos, instancia)

    # 1. Referencia: todo directo el primer día (lo pagado o lo cotizado, según el modo de la carga)
    costo_actual = pedidos["flete_sin_consolidar"].sum()
    retraso_actual = servicio.expected_delay_direct(pedidos).mean()

    # 2. Pedidos que el simulador debía despachar: los que tienen fecha de captura
    con_fecha = sorted(pedidos.loc[pedidos["fecha_captura"].notna(), "id_pedido"])
    fracciones = registro.groupby("id_pedido")["fraccion"].sum()
    completos = sorted(fracciones.index) == con_fecha and bool(((fracciones - 1).abs() < 1e-9).all())
    mismo_dia = bool((registro["modo"] == "directo").all() and (registro["dia_salida"] == registro["dia_llegada"]).all())

    return [
        {"revision": "costo (Z1)", "esperado": costo_actual, "obtenido": simulado["z1"],
         "cumple": abs(simulado["z1"] - costo_actual) <= TOLERANCIA * costo_actual},
        {"revision": "días de retraso ponderados (Z2)", "esperado": retraso_actual, "obtenido": simulado["z2"],
         "cumple": abs(simulado["z2"] - retraso_actual) <= TOLERANCIA},
        {"revision": "cada pedido despachado completo", "esperado": len(con_fecha), "obtenido": len(fracciones),
         "cumple": completos},
        {"revision": "todo sale directo el día que llega", "esperado": True, "obtenido": mismo_dia, "cumple": mismo_dia},
    ]


if __name__ == "__main__":
    inicio = time.perf_counter()
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
    instancia = prepare_instance(pedidos, opciones, capacidades, tarifas_ftl, nodos_regionales, SEDE)

    print(f"Simulando {MES_INICIO[:7]} con todo directo (espera máxima {ESPERA_MAXIMA}); modo {MODO}: "
          f"la referencia es {reference_name(MODO)}")
    revisiones = pd.DataFrame(validate(pedidos, instancia, ESPERA_MAXIMA))
    pd.set_option("display.width", 200)
    print(revisiones.to_string(index=False))
    print(f"{time.perf_counter() - inicio:.0f} s")
    if not revisiones["cumple"].all():
        print(f"ADVERTENCIA: el simulador no reproduce la referencia: {reference_name(MODO)}")
        sys.exit(1)
    print(f"El simulador reproduce la referencia: {reference_name(MODO)}")
