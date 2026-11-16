"""Instancia pequeña compartida por las pruebas de las metaheurísticas (tabú y genético).

C_0 puede salir lunes o martes, C_1 solo el martes, T_0 (traslado) lunes o miércoles y O_0 es de Oriente,
que no tiene sede regional: siempre va directo.
"""
from datetime import date

import pandas as pd

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import prepare_instance

SEDE = "0111"
CAPACIDADES = {"Turbo": 4500, "Sencillo": 8500}
TARIFAS = {("Turbo", SEDE, "MEDELLIN"): 2_100_000, ("Sencillo", SEDE, "MEDELLIN"): 2_300_000}
NODOS = {"ANTIOQUIA_CHOCO": "MEDELLIN", "ORIENTE": None}   # Oriente no tiene sede regional: siempre va directo
LUN, MAR, MIE = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)


def datos():
    # C_0 puede salir lunes o martes, C_1 solo el martes, T_0 (traslado) lunes o miércoles y O_0 es de Oriente
    pedidos = pd.DataFrame({
        "id_pedido": ["C_0", "C_1", "T_0", "O_0"],
        "tipo": ["CLIENTES", "CLIENTES", "TRASLADOS", "CLIENTES"],
        "region_logistica": ["ANTIOQUIA_CHOCO", "ANTIOQUIA_CHOCO", "ANTIOQUIA_CHOCO", "ORIENTE"],
        "peso_kg": [3000, 1000, 2000, 500],
        "flete_sin_consolidar": [1_500_000, 600_000, 900_000, 300_000],
        "flete_triangulacion": [100_000, 50_000, 80_000, 40_000],
    }).assign(transportadora_directo="ENVIA", tipo_transportadora="PAQUETEO", sede=SEDE, llega_directo=False)
    opciones = pd.DataFrame([("C_0", LUN, 0), ("C_0", MAR, 1), ("C_1", MAR, 0),
                             ("T_0", LUN, 0), ("T_0", MIE, 0), ("O_0", LUN, 0)],
                            columns=["id_pedido", "dia_salida", "retraso"])
    return pedidos, opciones


def instancia():
    pedidos, opciones = datos()
    return prepare_instance(pedidos, opciones, CAPACIDADES, TARIFAS, NODOS, SEDE)
