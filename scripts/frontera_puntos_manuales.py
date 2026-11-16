"""Puntos de la frontera costo–servicio escogidos a mano (zona del codo y servicio actual).

Reutiliza la matriz de pagos guardada por main_multiobjetivo.py, así no recalcula los extremos.
Los presupuestos de retraso se definen como porcentaje de los días de retraso ponderados por pedido de la operación actual
(calculado de los datos), así siguen teniendo sentido si cambia la escala de Z2.
Cada punto incluye las métricas del plan: nivel de servicio, pedidos en camión, espera promedio y camiones.
"""
import time

import pandas as pd

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PREFIJO_FRONTERA as PREFIJO
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.metodos.frontera_pareto import pareto_frontier
from despacho_multiobjetivo.reportes.frontera import plan_metrics

# % de los días de retraso ponderados por pedido de la operación actual: 100 % = mismo nivel de hoy
# (en enero de 2026, con i = 0,5 y súper castigo 10, dan ≈ 0,872; 1,014; 1,125 y 1,216 días ponderados por pedido)
PORCENTAJES_RETRASO_ACTUAL = [0.86, 1.00, 1.11, 1.20]


if __name__ == "__main__":
    inicio = time.perf_counter()
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    total_pagado = pedidos["flete_sin_consolidar"].sum()
    retraso_actual = servicio.expected_delay_direct(pedidos).mean()   # promedio por pedido, como Z2
    epsilons = [p * retraso_actual for p in PORCENTAJES_RETRASO_ACTUAL]
    matriz = pd.read_csv(f"{PREFIJO}_matriz.csv", index_col=0)
    print(f"Matriz de pagos leída de {PREFIJO}_matriz.csv")

    print("Armando el modelo multiobjetivo")
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                          nodos_regionales, SEDE)
    prob_directo = servicio.on_time_probability_direct(pedidos)
    prob_opciones = servicio.on_time_probability_options(opciones, pedidos)
    dias_directo = servicio.expected_delay_direct(pedidos, castigo=False)
    dias_opciones = servicio.expected_delay_options(opciones, pedidos, castigo=False)

    print(f"\nOperación actual: {retraso_actual:,.3f} días de retraso ponderados por pedido · puntos a mano: "
          + ", ".join(f"{p:.0%} = {e:,.3f}" for p, e in zip(PORCENTAJES_RETRASO_ACTUAL, epsilons)))
    puntos = pareto_frontier(problema, z1, z2, matriz, epsilons=epsilons,
                             metricas=lambda: plan_metrics(pedidos, opciones, x, u, y, prob_directo, prob_opciones,
                                                           dias_directo, dias_opciones))
    puntos.insert(1, "pct_retraso_actual", PORCENTAJES_RETRASO_ACTUAL)
    puntos["ahorro"] = total_pagado - puntos["costo"]
    puntos.to_csv(f"{PREFIJO}_manuales.csv", index=False)
    print(puntos.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
    print(f"\nResultados en {PREFIJO}_manuales.csv ({(time.perf_counter() - inicio) / 60:.1f} min) · sede {SEDE} · {MES_INICIO[:7]}")
