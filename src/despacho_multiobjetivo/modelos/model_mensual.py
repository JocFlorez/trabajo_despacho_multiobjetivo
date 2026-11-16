import pulp
from despacho_multiobjetivo.datos.data_processing import TIPOS_SIN_PARTIR
import pandas as pd

##VARIABLES

def create_variables_x(pedidos: pd.DataFrame) -> dict:
    x={}
    for row in pedidos.itertuples():
        if row.tipo in TIPOS_SIN_PARTIR:
            x[row.id_pedido] =pulp.LpVariable(f"x_{row.id_pedido}",lowBound=0, upBound=1, cat="Binary")
        else:
            x[row.id_pedido] = pulp.LpVariable(f"x_{row.id_pedido}", lowBound=0, upBound=1, cat="Continuous")
    return x

def create_variables_y(capacidades: dict, regiones, tarifas_ftl: dict, nodos_regionales: dict, sede: str) -> dict: #cantidad de vehículos tipo v en la region r
    y={}
    for v in capacidades:
        for r in regiones:
            hub = nodos_regionales.get(r)
            if hub is not None and (v, sede, hub) in tarifas_ftl:
                y[(v, r)] = pulp.LpVariable(f"y_{v}_{r}", lowBound=0, cat="Integer")
    return y

def create_variables_u(pedidos: pd.DataFrame) -> dict: # Cuánto del pedido i va directo por paquetería
    u={}
    for id_pedido in pedidos["id_pedido"]:
        u[id_pedido] = pulp.LpVariable(f"u_{id_pedido}", lowBound=0, upBound=1, cat="Continuous")
    return u


##RESTRICCIONES

def capacity_constraint(problema: pulp.LpProblem, pedidos: pd.DataFrame, x: dict, y: dict, capacidades: dict) -> pulp.LpProblem:
    for r in pedidos["region_logistica"].unique():
        pedidos_r = pedidos[pedidos["region_logistica"] == r]
        peso_consolidado = pulp.lpSum(row.peso_kg*x[row.id_pedido] for row in pedidos_r.itertuples())
        capacidad_regional = pulp.lpSum(capacidades[v]*y[(v, r)] for v in capacidades if (v, r) in y)
        problema += peso_consolidado <= capacidad_regional, f"capacidad_{r}"
    return problema

def full_dispatch_constraint(problema: pulp.LpProblem, x: dict, u: dict) -> pulp.LpProblem:
    # Cada pedido se despacha completo: lo que va directo más lo que va en camión es todo el pedido
    for id_pedido, variable_directo in u.items():
        problema += variable_directo + x[id_pedido] == 1, f"salida_completa_{id_pedido}"
    return problema


#FUNCIÓN OBJETIVO

def direct_ltl_cost(pedidos: pd.DataFrame, u: dict) -> pulp.LpAffineExpression:
    # flete_sin_consolidar: lo pagado (modo histórico) o lo cotizado (modo operación)
    return pulp.lpSum(row.flete_sin_consolidar * u[row.id_pedido] for row in pedidos.itertuples())

def triangulation_cost(pedidos: pd.DataFrame, x: dict) -> pulp.LpAffineExpression:
    return pulp.lpSum(x[row.id_pedido] * row.flete_triangulacion for row in pedidos.itertuples())

def ftl_trunk_cost(y: dict, tarifas_ftl: dict, nodos_regionales: dict, sede: str) -> pulp.LpAffineExpression:
    return pulp.lpSum(tarifas_ftl[(v, sede, nodos_regionales[r])]* y[(v, r)] for (v, r) in y)

def objective_function(problema: pulp.LpProblem, pedidos: pd.DataFrame, x: dict, u: dict, y: dict, tarifas_ftl: dict, nodos_regionales: dict, sede: str) -> pulp.LpProblem:
    problema += direct_ltl_cost(pedidos, u) + triangulation_cost(pedidos, x) + ftl_trunk_cost(y, tarifas_ftl, nodos_regionales, sede)
    return problema


def build_model(pedidos: pd.DataFrame,
                capacidades: dict,
                tarifas_ftl: dict,
                nodos_regionales: dict,
                sede: str
                ) -> tuple[pulp.LpProblem, dict, dict, dict]:
    problema = pulp.LpProblem("Consolidacion_Carga", pulp.LpMinimize)

    x = create_variables_x(pedidos)
    u = create_variables_u(pedidos)
    regiones = pedidos["region_logistica"].unique()
    y = create_variables_y(capacidades, regiones, tarifas_ftl, nodos_regionales, sede)

    problema = full_dispatch_constraint(problema, x, u)
    problema = capacity_constraint(problema, pedidos, x, y, capacidades)
    problema = objective_function(problema, pedidos, x, u, y, tarifas_ftl, nodos_regionales, sede)

    return problema, x, u, y