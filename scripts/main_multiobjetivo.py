"""Frontera costo–servicio (Etapa 3): extremos, matriz de pagos y frontera de Pareto con el método ε-restricción.

Resultados en data/processed/: matriz de pagos y frontera en CSV, y la gráfica en PNG.
"""
import time

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PREFIJO_FRONTERA as PREFIJO
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.metodos.frontera_pareto import cost_extreme, delay_extreme, payoff_matrix, pareto_frontier
from despacho_multiobjetivo.reportes.frontera import plan_metrics
from despacho_multiobjetivo.reportes.frontera import plot_frontier

N_PUNTOS = 10


if __name__ == "__main__":
    inicio = time.perf_counter()
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    total_pagado = pedidos["flete_sin_consolidar"].sum()
    linea_base = {"costo": total_pagado, "retraso": servicio.expected_delay_direct(pedidos).mean()}
    print(f"Operación actual (todo directo): costo {linea_base['costo']:,.0f} | "
          f"{linea_base['retraso']:,.3f} días de retraso ponderados por pedido · "
          f"{servicio.expected_delay_direct(pedidos, castigo=False).mean():.3f} días hábiles de retraso por pedido · "
          f"{servicio.on_time_probability_direct(pedidos).mean():.1%} a tiempo")

    print("Armando el modelo multiobjetivo")
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, capacidades, tarifas_ftl,
                                                          nodos_regionales, SEDE)

    print("Extremo de costo")
    resultado_costo = cost_extreme(problema, z1, z2)
    print("Extremo de retraso")
    resultado_retraso = delay_extreme(problema, z1, z2)

    matriz = payoff_matrix(resultado_costo, resultado_retraso)
    matriz["ahorro"] = total_pagado - matriz["costo"]
    print("\nMatriz de pagos:")
    print(matriz.to_string(float_format=lambda v: f"{v:,.1f}"))
    matriz.to_csv(f"{PREFIJO}_matriz.csv")

    print(f"\nFrontera ({N_PUNTOS} puntos)")
    # Métricas en lenguaje natural para cada punto: % a tiempo, días de retraso por pedido, pedidos en camión…
    prob_directo = servicio.on_time_probability_direct(pedidos)
    prob_opciones = servicio.on_time_probability_options(opciones, pedidos)
    dias_directo = servicio.expected_delay_direct(pedidos, castigo=False)
    dias_opciones = servicio.expected_delay_options(opciones, pedidos, castigo=False)
    puntos = pareto_frontier(problema, z1, z2, matriz, N_PUNTOS,
                             metricas=lambda: plan_metrics(pedidos, opciones, x, u, y, prob_directo, prob_opciones,
                                                           dias_directo, dias_opciones))
    puntos["ahorro"] = total_pagado - puntos["costo"]
    puntos.to_csv(f"{PREFIJO}_puntos.csv", index=False)
    print(puntos.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    plot_frontier(matriz, puntos, linea_base,
                  f"Frontera costo–servicio · sede {SEDE} · {MES_INICIO[:7]} · espera máxima {ESPERA_MAXIMA}",
             f"{PREFIJO}.png")
    print(f"\nResultados en {PREFIJO}_matriz.csv, {PREFIJO}_puntos.csv y {PREFIJO}.png "
          f"({(time.perf_counter() - inicio) / 60:.1f} min)")
