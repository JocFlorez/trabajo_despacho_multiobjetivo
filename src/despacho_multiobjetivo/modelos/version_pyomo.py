"""Modelo multiobjetivo de consolidación (Etapa 3) escrito en Pyomo.

Es el mismo modelo de modelos/model_multiobjetivo.py (PuLP): mismas variables, mismas
restricciones y mismos objetivos Z1 (costo) y Z2 (días de retraso ponderados por pedido), armados con las mismas entradas.
Sirve para validar el modelo con otra librería y para comparar solvers.

Se escribe en el orden de la formulación, como en los notebooks del curso: conjuntos (Set), parámetros
(Param), variables (Var), restricciones y objetivos. Cada restricción es una regla suelta (como supply_rule
en el notebook) que Pyomo aplica a cada índice; lo que la regla necesita está dentro del modelo, incluidos
los conjuntos indexados como PEDIDOS_RT[r, t]. Cada componente lleva su descripción (doc).
Los nombres siguen la notación de la presentación: I, Iᶜ, Tᵢ, cᵢᵈⁱʳ, cᵢᵗʳⁱ, f_vr, q_v, wᵢ, dᵢᵈⁱʳ, d_it.
"""
import pandas as pd
import pyomo.environ as pyo

from despacho_multiobjetivo.datos.data_processing import TIPOS_SIN_PARTIR
from despacho_multiobjetivo.reglas_negocio import servicio
from despacho_multiobjetivo.modelos.componentes import enrich_options

##CONJUNTOS

def create_sets(modelo: pyo.ConcreteModel, opciones: pd.DataFrame, pedidos: pd.DataFrame, capacidades: dict,
                tarifas_ftl: dict, nodos_regionales: dict, sede: str) -> pyo.ConcreteModel:
    m = modelo
    m.I = pyo.Set(initialize=list(pedidos["id_pedido"]), doc="Pedidos del mes")
    m.IC = pyo.Set(initialize=list(opciones["id_pedido"].drop_duplicates()), doc="Pedidos que pueden ir en camión (Iᶜ)")
    m.OPCIONES = pyo.Set(initialize=list(zip(opciones["id_pedido"], opciones["dia_salida"])), dimen=2,
                         doc="Pedido i y día t en que puede salir en camión")
    m.V = pyo.Set(initialize=list(capacidades), doc="Tipos de vehículo")

    regiones_dias = opciones[["region_logistica", "dia_salida"]].drop_duplicates()
    m.REGIONES_DIAS = pyo.Set(initialize=list(zip(regiones_dias["region_logistica"], regiones_dias["dia_salida"])), dimen=2,
                              doc="Región r y día t con al menos un pedido que puede salir en camión")

    vehiculos_region = [(v, r) for r in regiones_dias["region_logistica"].drop_duplicates() for v in capacidades
                        if nodos_regionales.get(r) is not None and (v, sede, nodos_regionales[r]) in tarifas_ftl]
    m.VR = pyo.Set(initialize=vehiculos_region, dimen=2, doc="Vehículo v con tarifa de la sede a la sede regional de r")
    m.CAMIONES = pyo.Set(initialize=[(v, r, t) for r, t in m.REGIONES_DIAS for v in capacidades if (v, r) in m.VR],
                         dimen=3, doc="Camión tipo v hacia la región r el día t")

    # Conjuntos indexados: para cada índice, la lista de elementos que le corresponden
    dias_por_pedido = {i: [] for i in m.I}
    pedidos_por_region_dia = {(r, t): [] for r, t in m.REGIONES_DIAS}
    for row in opciones.itertuples():
        dias_por_pedido[row.id_pedido].append(row.dia_salida)
        pedidos_por_region_dia[row.region_logistica, row.dia_salida].append(row.id_pedido)
    vehiculos_por_region_dia = {(r, t): [v for v in capacidades if (v, r) in m.VR] for r, t in m.REGIONES_DIAS}

    m.T_I = pyo.Set(m.I, initialize=dias_por_pedido, doc="Días Tᵢ en que el pedido i puede salir en camión")
    m.PEDIDOS_RT = pyo.Set(m.REGIONES_DIAS, initialize=pedidos_por_region_dia,
                           doc="Pedidos que pueden salir hacia la región r el día t")
    m.VEHICULOS_RT = pyo.Set(m.REGIONES_DIAS, initialize=vehiculos_por_region_dia,
                             doc="Vehículos que pueden ir hacia la región r el día t")
    return modelo

##PARÁMETROS

def create_parameters(modelo: pyo.ConcreteModel, pedidos: pd.DataFrame, opciones: pd.DataFrame, capacidades: dict,
                      tarifas_ftl: dict, nodos_regionales: dict, sede: str,
                     retraso_directo: pd.Series, retraso_opciones: pd.Series) -> pyo.ConcreteModel:
    por_pedido = opciones.drop_duplicates("id_pedido").set_index("id_pedido")
    m = modelo
    m.c_dir = pyo.Param(m.I, initialize=dict(zip(pedidos["id_pedido"], pedidos["flete_sin_consolidar"])),
                        doc="Flete directo CEDI → destino del pedido i (cᵢᵈⁱʳ)")
    m.c_tri = pyo.Param(m.IC, initialize=por_pedido["flete_triangulacion"].to_dict(),
                        doc="Flete del último tramo sede regional → destino del pedido i (cᵢᵗʳⁱ)")
    m.w = pyo.Param(m.IC, initialize=por_pedido["peso_kg"].to_dict(), doc="Peso del pedido i en kg (wᵢ)")
    m.tipo = pyo.Param(m.IC, initialize=por_pedido["tipo"].to_dict(), within=pyo.Any,
                       doc="Tipo de pedido: clientes y otros no se parten, traslados sí")
    m.region = pyo.Param(m.IC, initialize=por_pedido["region_logistica"].to_dict(), within=pyo.Any,
                         doc="Región logística del pedido i, r(i)")
    m.q = pyo.Param(m.V, initialize=capacidades, doc="Capacidad del vehículo v en kg (q_v)")
    m.f = pyo.Param(m.VR, initialize={(v, r): tarifas_ftl[v, sede, nodos_regionales[r]] for v, r in m.VR},
                    doc="Tarifa de un camión v de la sede a la sede regional de r (f_vr)")
    m.d_dir = pyo.Param(m.I, initialize=retraso_directo.to_dict(),
                        doc="Castigo esperado por retraso si el pedido i va directo, en días equivalentes (dᵢᵈⁱʳ)")
    m.d = pyo.Param(m.OPCIONES, initialize=retraso_opciones.to_dict(),
                    doc="Castigo esperado por retraso si el pedido i sale en camión el día t, en días equivalentes (d_it)")
    return modelo

##VARIABLES

def x_domain_rule(m, i, t):
    # Clientes y otros no se parten (binaria); los traslados sí (continua entre 0 y 1)
    return pyo.Binary if m.tipo[i] in TIPOS_SIN_PARTIR else pyo.UnitInterval

def create_variables(modelo: pyo.ConcreteModel) -> pyo.ConcreteModel:
    modelo.x = pyo.Var(modelo.OPCIONES, domain=x_domain_rule, doc="Parte del pedido i que sale en camión el día t")
    modelo.u = pyo.Var(modelo.I, domain=pyo.UnitInterval, doc="Parte del pedido i que va directo")
    modelo.y = pyo.Var(modelo.CAMIONES, domain=pyo.NonNegativeIntegers, doc="Camiones tipo v a la región r el día t")
    return modelo

##RESTRICCIONES

def full_dispatch_rule(m, i):
    # uᵢ + Σ_t xᵢₜ = 1  ∀ i ∈ I
    return m.u[i] + pyo.quicksum(m.x[i, t] for t in m.T_I[i]) == 1

def capacity_rule(m, r, t):
    # Σ_i wᵢ·xᵢₜ ≤ Σ_v q_v·y_vrt  ∀ (r, t)
    return (pyo.quicksum(m.w[i] * m.x[i, t] for i in m.PEDIDOS_RT[r, t])
            <= pyo.quicksum(m.q[v] * m.y[v, r, t] for v in m.VEHICULOS_RT[r, t]))

def linking_rule(m, i, t):
    # xᵢₜ ≤ Σ_v y_v,r(i),t  ∀ (i, t)
    return m.x[i, t] <= pyo.quicksum(m.y[v, m.region[i], t] for v in m.VEHICULOS_RT[m.region[i], t])

def create_constraints(modelo: pyo.ConcreteModel) -> pyo.ConcreteModel:
    modelo.salida_completa = pyo.Constraint(modelo.I, rule=full_dispatch_rule,
                                            doc="Cada pedido sale completo: directo o en camión")
    modelo.capacidad = pyo.Constraint(modelo.REGIONES_DIAS, rule=capacity_rule,
                                      doc="La carga hacia la región r el día t cabe en los camiones comprados")
    modelo.enlace = pyo.Constraint(modelo.OPCIONES, rule=linking_rule,
                                   doc="Si el pedido i sale en camión el día t, sale al menos un camión hacia su región")
    return modelo

##FUNCIONES OBJETIVO

def z1_cost_rule(m):
    return (pyo.quicksum(m.c_dir[i] * m.u[i] for i in m.I)                     # despacho directo
            + pyo.quicksum(m.c_tri[i] * m.x[i, t] for i, t in m.OPCIONES)      # último tramo
            + pyo.quicksum(m.f[v, r] * m.y[v, r, t] for v, r, t in m.CAMIONES))  # camión troncal

def z2_delay_rule(m):
    # Promedio por pedido: el total del mes dividido entre el número de pedidos
    return (pyo.quicksum(m.d_dir[i] * m.u[i] for i in m.I)
            + pyo.quicksum(m.d[i, t] * m.x[i, t] for i, t in m.OPCIONES)) / len(m.I)

def create_objectives(modelo: pyo.ConcreteModel) -> pyo.ConcreteModel:
    # Se guardan como expresiones (no como Objective) para escoger después cuál se minimiza
    modelo.z1 = pyo.Expression(rule=z1_cost_rule, doc="Z1: costo total del mes")
    modelo.z2 = pyo.Expression(rule=z2_delay_rule, doc="Z2: días de retraso ponderados por pedido")
    return modelo


def set_objective(modelo: pyo.ConcreteModel, expresion) -> pyo.ConcreteModel:
    # Pyomo admite un solo objetivo activo: se borra el anterior y se pone el nuevo (equivale a setObjective de PuLP)
    if hasattr(modelo, "objetivo"):
        modelo.del_component(modelo.objetivo)
    modelo.objetivo = pyo.Objective(expr=expresion, sense=pyo.minimize)
    return modelo


def build_model(opciones: pd.DataFrame,
                pedidos: pd.DataFrame,
                capacidades: dict,
                tarifas_ftl: dict,
                nodos_regionales: dict,
                sede: str
                ) -> pyo.ConcreteModel:
    # Igual que en PuLP: sin objetivo; z1 y z2 quedan en el modelo para escoger cuál se minimiza
    modelo = pyo.ConcreteModel("Consolidacion_Multiobjetivo")

    retraso_directo = servicio.expected_delay_direct(pedidos)
    retraso_opciones = servicio.expected_delay_options(opciones, pedidos)

    opciones = enrich_options(opciones, pedidos)

    modelo = create_sets(modelo, opciones, pedidos, capacidades, tarifas_ftl, nodos_regionales, sede)
    modelo = create_parameters(modelo, pedidos, opciones, capacidades, tarifas_ftl, nodos_regionales, sede,
                               retraso_directo, retraso_opciones)
    modelo = create_variables(modelo)
    modelo = create_constraints(modelo)
    modelo = create_objectives(modelo)
    return modelo
