"""Etapa 3: modelo multiobjetivo costo – servicio.

Usa las mismas variables y restricciones de la Etapa 2 y no fija objetivo: devuelve Z1 (costo)
y Z2 (días de retraso ponderados por pedido) para que el método de solución escoja qué minimizar y qué restringir.
"""
import pulp
import pandas as pd
from despacho_multiobjetivo.reglas_negocio import servicio
from despacho_multiobjetivo.modelos import componentes


def build_model(opciones: pd.DataFrame,
                pedidos: pd.DataFrame,
                capacidades: dict,
                tarifas_ftl: dict,
                nodos_regionales: dict,
                sede: str
                ) -> tuple[pulp.LpProblem, dict, dict, dict, pulp.LpAffineExpression, pulp.LpAffineExpression]:
    retraso_directo = servicio.expected_delay_direct(pedidos)
    retraso_opciones = servicio.expected_delay_options(opciones, pedidos)

    problema, opciones, x, u, y = componentes.build_base(opciones, pedidos, capacidades, tarifas_ftl,
                                                         nodos_regionales, sede, "Consolidacion_Multiobjetivo")

    z1 = componentes.objective_z1_cost(pedidos, opciones, x, u, y, tarifas_ftl, nodos_regionales, sede)
    z2 = componentes.objective_z2_delay(u, x, retraso_directo, retraso_opciones)
    return problema, x, u, y, z1, z2
