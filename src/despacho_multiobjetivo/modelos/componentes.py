"""Piezas comunes de los modelos con ventanas de tiempo (Etapa 2) y multiobjetivo (Etapa 3).

Las variables, las restricciones y las funciones objetivo se escriben aquí una sola vez;
cada etapa las arma con build_base() y escoge qué objetivo usa.
"""
import pulp
import pandas as pd
from despacho_multiobjetivo.datos.data_processing import TIPOS_SIN_PARTIR

##VARIABLES

def create_variables_x(opciones: pd.DataFrame) -> dict: # Cuanto del pedido i sale en camion el dia t?
    x={}
    for row in opciones.itertuples():
        if row.tipo in TIPOS_SIN_PARTIR:
            x[row.id_pedido, row.dia_salida] = pulp.LpVariable(f"x_{row.id_pedido}_{row.dia_salida:%Y%m%d}",lowBound=0, upBound=1, cat="Binary")
        else:
            x[row.id_pedido, row.dia_salida] = pulp.LpVariable(f"x_{row.id_pedido}_{row.dia_salida:%Y%m%d}", lowBound=0, upBound=1, cat="Continuous")
    return x

def create_variables_y(capacidades: dict,regiones_dias: pd.DataFrame, tarifas_ftl: dict, nodos_regionales: dict, sede: str) -> dict: #cantidad de vehículos tipo v en la region r el dia t
    y={}
    for par in regiones_dias.itertuples():
        hub = nodos_regionales.get(par.region_logistica)
        for v in capacidades:
            if hub is not None and (v, sede, hub) in tarifas_ftl:
                y[(v, par.region_logistica, par.dia_salida)] = pulp.LpVariable(f"y_{v}_{par.region_logistica}_{par.dia_salida:%Y%m%d}", lowBound=0, cat="Integer")
    return y

def create_variables_u(pedidos: pd.DataFrame) -> dict:  # Cuanto del pedido i sale en paquetería (no consolidado)
    u={}
    for id_pedido in pedidos["id_pedido"]:
        u[id_pedido] = pulp.LpVariable(f"u_{id_pedido}", lowBound=0, upBound=1, cat="Continuous")
    return u


##RESTRICCIONES

def capacity_constraint(problema: pulp.LpProblem, opciones: pd.DataFrame, x: dict, y: dict, capacidades: dict) -> pulp.LpProblem:
    carga_por_region_dia = {}

    for row in opciones.itertuples():
        carga_por_region_dia.setdefault((row.region_logistica, row.dia_salida),[]).append(row.peso_kg * x[row.id_pedido, row.dia_salida])

    for (region, dia), cargas in carga_por_region_dia.items():
        capacidad = pulp.lpSum(capacidades[v] * y[v, region, dia] for v in capacidades if (v, region, dia) in y)
        problema += pulp.lpSum(cargas) <= capacidad, f"capacidad_{region}_{dia:%Y%m%d}"
    return problema

def full_dispatch_constraint(problema: pulp.LpProblem, x: dict, u: dict) -> pulp.LpProblem:
    salidas_por_pedido = {}
    for (id_pedido, dia_salida), variable in x.items():
        salidas_por_pedido.setdefault(id_pedido, []).append(variable)

    for id_pedido, variable_directo in u.items():
        problema += variable_directo + pulp.lpSum(salidas_por_pedido.get(id_pedido, [])) ==1, f"salida_completa_{id_pedido}"
    return problema

def linking_constraint(problema: pulp.LpProblem, opciones: pd.DataFrame, x: dict, y: dict) -> pulp.LpProblem:
    # Si el pedido i sale en camión el día t, ese día debe salir al menos un camión hacia su región.
    # No cambia el óptimo (los camiones son enteros), pero impide que la relajación lineal
    # compre fracciones pequeñas de camión: sube la cota inferior y cierra el gap más rápido
    camiones_por_region_dia = {}
    for (v, region, dia), variable in y.items():
        camiones_por_region_dia.setdefault((region, dia), []).append(variable)

    for row in opciones.itertuples():
        camiones = pulp.lpSum(camiones_por_region_dia.get((row.region_logistica, row.dia_salida), []))
        problema += x[row.id_pedido, row.dia_salida] <= camiones, f"enlace_{row.id_pedido}_{row.dia_salida:%Y%m%d}"
    return problema


#FUNCIÓN OBJETIVO z1

def direct_ltl_cost(pedidos: pd.DataFrame, u: dict) -> pulp.LpAffineExpression:
    return pulp.lpSum(row.flete_sin_consolidar * u[row.id_pedido] for row in pedidos.itertuples())

def triangulation_cost(opciones: pd.DataFrame, x: dict) -> pulp.LpAffineExpression:
    return pulp.lpSum(x[row.id_pedido, row.dia_salida] * row.flete_triangulacion for row in opciones.itertuples())

def ftl_trunk_cost(y: dict, tarifas_ftl: dict, nodos_regionales: dict, sede: str) -> pulp.LpAffineExpression:
    return pulp.lpSum(tarifas_ftl[(v, sede, nodos_regionales[r])]* y[(v, r, t)] for (v, r, t) in y)

def objective_z1_cost(pedidos: pd.DataFrame, opciones: pd.DataFrame, x: dict, u: dict, y: dict, tarifas_ftl: dict, nodos_regionales: dict, sede: str) -> pulp.LpAffineExpression:
    return direct_ltl_cost(pedidos, u) + triangulation_cost(opciones, x) + ftl_trunk_cost(y, tarifas_ftl, nodos_regionales, sede)

#FUNCION OBJETIVO Z2

def total_direct_delay(u: dict, retraso_directo: pd.Series) -> pulp.LpAffineExpression:
    return pulp.lpSum(retraso_directo[id_pedido]*variable for id_pedido, variable in u.items())

def consolidated_delay(x: dict, retraso_opciones: pd.Series) -> pulp.LpAffineExpression:
    return pulp.lpSum(retraso_opciones[llave]*variable for llave, variable in x.items())

def objective_z2_delay(u: dict, x: dict, retraso_directo: pd.Series, retraso_opciones: pd.Series) -> pulp.LpAffineExpression:
    # Días de retraso ponderados por pedido: el total del mes dividido entre el número de pedidos
    # (u tiene una variable por pedido). Dividir entre una constante no cambia los planes, solo la escala
    return (total_direct_delay(u, retraso_directo) + consolidated_delay(x, retraso_opciones)) / len(u)



def enrich_options(opciones: pd.DataFrame, pedidos: pd.DataFrame) -> pd.DataFrame:
    # Cada opción (pedido, día) hereda del pedido lo que necesita el modelo
    columnas = ["id_pedido", "tipo", "peso_kg", "region_logistica", "flete_triangulacion"]
    return opciones.merge(pedidos[columnas], on="id_pedido", how="left", validate="many_to_one")


def build_base(opciones: pd.DataFrame,
               pedidos: pd.DataFrame,
               capacidades: dict,
               tarifas_ftl: dict,
               nodos_regionales: dict,
               sede: str,
               nombre: str
               ) -> tuple[pulp.LpProblem, pd.DataFrame, dict, dict, dict]:
    # Variables x, y, u y restricciones de salida completa, capacidad y enlace, sin función objetivo.
    # Devuelve también las opciones enriquecidas, que las funciones objetivo necesitan
    problema = pulp.LpProblem(nombre, pulp.LpMinimize)

    opciones = enrich_options(opciones, pedidos)

    x = create_variables_x(opciones)
    regiones_dias = opciones[["region_logistica", "dia_salida"]].drop_duplicates()
    y = create_variables_y(capacidades, regiones_dias, tarifas_ftl, nodos_regionales, sede)
    u = create_variables_u(pedidos)

    problema = full_dispatch_constraint(problema, x, u)
    problema = capacity_constraint(problema, opciones, x, y, capacidades)
    problema = linking_constraint(problema, opciones, x, y)
    return problema, opciones, x, u, y
