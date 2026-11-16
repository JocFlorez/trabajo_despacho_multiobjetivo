"""Validación del modelo con ventanas de consolidación (Etapa 2) contra el modelo mensual (Etapa 1).

Corre el modelo mensual como referencia y luego el modelo con ventanas para varias
esperas máximas. Lo que debe cumplirse:
  - El ahorro nunca baja cuando aumenta la espera.
  - Con espera de un mes el ahorro se acerca al mensual, sin superarlo.
"""
import time

import pandas as pd
import pulp

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, PROCESSED, GAP_REL_VALIDACION, LIMITE_SEGUNDOS_VALIDACION
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida
from despacho_multiobjetivo.modelos import model_mensual, model_ventanas
from despacho_multiobjetivo.metodos import solver

ESPERAS = [3]
# El nombre lleva el rango de esperas para que una corrida nueva no sobrescriba las anteriores
RUTA_RESULTADOS = PROCESSED / f"validacion_ventanas_{SEDE}_{MES_INICIO[:7]}_E{ESPERAS[0]}-{ESPERAS[-1]}.csv"


def run_monthly(pedidos, capacidades, tarifas_ftl, nodos_regionales) -> float:
    # Misma resolución que la Etapa 1 (scripts/main.py): óptimo exacto; se detiene si no lo encuentra
    problema, _, _, _ = model_mensual.build_model(pedidos, capacidades, tarifas_ftl, nodos_regionales, SEDE)
    problema = solver.solve(problema)
    return pedidos["flete_sin_consolidar"].sum() - pulp.value(problema.objective)


def run_windows(pedidos, capacidades, tarifas_ftl, nodos_regionales, tiempos_ftl, espera_maxima) -> dict:
    inicio = time.perf_counter()

    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, espera_maxima)
    problema, x, u, y = model_ventanas.build_model(opciones, pedidos, capacidades,
                                             tarifas_ftl, nodos_regionales, SEDE)
    problema.solve(pulp.HiGHS(msg=True, timeLimit=LIMITE_SEGUNDOS_VALIDACION, gapRel=GAP_REL_VALIDACION))
    # HiGHS ve el costo total (el directo entra por u): su cota es el costo mínimo posible
    info = problema.solverModel.getInfo()

    salida_por_pedido = {}
    for (id_pedido, dia_salida), variable in x.items():
        salida_por_pedido[id_pedido] = salida_por_pedido.get(id_pedido, 0) + variable.varValue

    return {
        "espera": espera_maxima,
        # PuLP marca "Optimal" también cuando HiGHS se detiene por tiempo: se usa el estado de HiGHS
        "estado": problema.solverModel.modelStatusToString(problema.solverModel.getModelStatus()),
        "pedidos_con_opciones": opciones["id_pedido"].nunique(),
        "opciones": len(opciones),
        "pedidos_consolidados": sum(1 for total in salida_por_pedido.values() if total > 1e-6),
        "camiones": sum(variable.varValue for variable in y.values()),
        "ahorro": pedidos["flete_sin_consolidar"].sum() - pulp.value(problema.objective),
        "ahorro_maximo": pedidos["flete_sin_consolidar"].sum() - info.mip_dual_bound,
        "gap": info.mip_gap,
        "segundos": time.perf_counter() - inicio,
    }


if __name__ == "__main__":
    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    total = pedidos["flete_sin_consolidar"].sum()
    print(f"Pedidos: {len(pedidos)}   Flete sin consolidar: {total:,.0f}")

    print("Corriendo modelo mensual (referencia)")
    ahorro_mensual = run_monthly(pedidos, capacidades, tarifas_ftl, nodos_regionales)
    print(f"Ahorro mensual: {ahorro_mensual:,.0f} ({ahorro_mensual / total:.2%})\n")

    resultados = []
    for espera in ESPERAS:
        print(f"Corriendo ventanas con espera máxima {espera}")
        r = run_windows(pedidos, capacidades, tarifas_ftl, nodos_regionales, tiempos_ftl, espera)
        r["pct_total"] = r["ahorro"] / total
        r["pct_mensual"] = r["ahorro"] / ahorro_mensual
        resultados.append(r)
        print(f"   {r['estado']}  ahorro {r['ahorro']:,.0f} (máximo posible {r['ahorro_maximo']:,.0f}, gap {r['gap']:.2%})  "
              f"{r['pct_mensual']:.1%} del mensual  {r['segundos']:.1f} s")

        # Se guarda después de cada espera: si el script se detiene, lo ya calculado no se pierde
        tabla = pd.DataFrame(resultados)
        tabla.to_csv(RUTA_RESULTADOS, index=False)

    print(f"\nResultados guardados en {RUTA_RESULTADOS}")
    print(tabla.to_string(index=False))

    # Comprobaciones
    # Con gap, una espera mayor puede quedar un poco por debajo de la anterior. Solo es error si
    # ni siquiera su ahorro máximo posible alcanza el ahorro ya logrado con la espera menor
    ahorros = [r["ahorro"] for r in resultados]
    if any(despues["ahorro_maximo"] < antes["ahorro"] - 1 for antes, despues in zip(resultados, resultados[1:])):
        print("\nERROR: el ahorro bajó al aumentar la espera (más de lo que explica el gap)")
    if max(ahorros) > ahorro_mensual + 1:
        print("\nERROR: el modelo con ventanas ahorra más que el mensual")