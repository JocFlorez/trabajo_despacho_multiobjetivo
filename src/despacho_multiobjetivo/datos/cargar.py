"""Carga de pedidos y parámetros: la usan todos los scripts (Etapa 1 en main.py, Etapas 2 y 3)."""
import pandas as pd

from despacho_multiobjetivo.datos import data_processing


def prepare_orders(sede: str, mes_inicio: str, mes_fin: str, modo: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Pedidos del periodo con su identificador, cotizados (directo y triangulación) y con su costo sin consolidar.
    # Devuelve (pedidos para el modelo, pedidos para revisión manual en modo operación, nodos regionales)
    pedidos = data_processing.get_shipments(sede, mes_inicio, mes_fin)
    pedidos = data_processing.add_unique_id(pedidos)
    nodos_regionales = data_processing.load_regional_hubs()
    pedidos = data_processing.rate_shipments(pedidos, nodos_regionales)
    pedidos = data_processing.rate_ltl_direct(pedidos)
    pedidos, revision_manual = data_processing.prepare_unconsolidated_cost(pedidos, modo)
    return pedidos, revision_manual, nodos_regionales


def load_data(sede: str, mes_inicio: str, mes_fin: str, modo: str) -> tuple[pd.DataFrame, dict, dict, dict, dict]:
    # Pedidos del periodo (ver prepare_orders) y los parámetros de camiones y tiempos de las Etapas 2 y 3
    pedidos, _, nodos_regionales = prepare_orders(sede, mes_inicio, mes_fin, modo)
    capacidades = data_processing.load_vehicle_capacities()
    tarifas_ftl = data_processing.load_ftl_rates()
    tiempos_ftl = data_processing.load_ftl_transit_times()
    return pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl
