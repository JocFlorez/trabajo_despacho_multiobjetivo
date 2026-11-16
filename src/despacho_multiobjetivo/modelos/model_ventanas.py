"""Etapa 2: modelo con ventanas de consolidación. Cada pedido escoge el día en que sale; se minimiza el costo (Z1)."""
import pulp
import pandas as pd
from despacho_multiobjetivo.modelos import componentes


def build_model(opciones: pd.DataFrame,
                pedidos: pd.DataFrame,
                capacidades: dict,
                tarifas_ftl: dict,
                nodos_regionales: dict,
                sede: str
                ) -> tuple[pulp.LpProblem, dict, dict, dict]:
    problema, opciones, x, u, y = componentes.build_base(opciones, pedidos, capacidades, tarifas_ftl,
                                                         nodos_regionales, sede, "Consolidacion_Carga_Ventanas")
    problema += componentes.objective_z1_cost(pedidos, opciones, x, u, y, tarifas_ftl, nodos_regionales, sede)
    return problema, x, u, y
