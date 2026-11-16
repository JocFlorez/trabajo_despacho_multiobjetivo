import pandas as pd
from despacho_multiobjetivo import parametros
from despacho_multiobjetivo.reglas_negocio import calendario as cal
from datetime import date
from datetime import timedelta
from typing import Literal

def operation_start(captura: date) -> date:
    # Si se captura en domingo o festivo, el pedido entra a operación el siguiente día de despacho
    inicio = captura
    while cal.business_day_factor(inicio) == 0:
        inicio = inicio + timedelta(days=1)
    return inicio


def possible_departure_days(captura: date, espera_maxima: float) -> list[date]:
    inicio = operation_start(captura)

    fechas_posible_salida = []
    fecha = inicio

    while cal.business_days_between(inicio, fecha) <= espera_maxima:
        if cal.business_day_factor(fecha) > 0:
            fechas_posible_salida.append(fecha)
        fecha = fecha + timedelta(days=1)

    return fechas_posible_salida


def delivery_date(dia_salida: date, transito: int, manipulacion: float, ultimo_tramo: float, llega_directo: bool) -> date:
    descarga = cal.unloading_day(dia_salida, transito)

    if llega_directo:
        return descarga
    else:
        entrega_transportadora = cal.due_date(descarga, manipulacion)
        return cal.due_date(entrega_transportadora, ultimo_tramo)


def delay(entrega: date, limite: date, valor_sabado: Literal[0.5,0] = 0.5) -> float:
    if entrega <= limite:
        return 0.0
    return cal.business_days_between(limite, entrega, valor_sabado)


COLUMNAS_OPCIONES = ["id_pedido", "dia_salida", "espera", "fecha_entrega", "fecha_limite", "retraso"]

def build_departure_options(pedidos: pd.DataFrame, tiempos_ftl: dict, espera_maxima: float,
                            manipulacion: float | None = None) -> pd.DataFrame:
    """Una fila por cada pedido que puede ir en camión y cada día en que podría salir,
    con su fecha de entrega y su retraso frente a la promesa estricta (sin día de gracia,
    sábado no hábil). Los pedidos sin
    hub, en Big-M, sin tránsito sede→hub o sin fecha de captura no tienen filas:
    para ellos solo existe el despacho directo. Sin manipulacion se usa
    parametros.DIAS_MANIPULACION_HUB."""
    if manipulacion is None:
        manipulacion = parametros.DIAS_MANIPULACION_HUB
    tiene_transito = pd.Series(
        [(sede, hub) in tiempos_ftl for sede, hub in zip(pedidos["sede"], pedidos["nodo_regional"])],
        index=pedidos.index,
    )
    consolidables = pedidos[
        pedidos["nodo_regional"].notna()
        & ~pedidos["sin_ruta_triangulacion"]
        & pedidos["fecha_captura"].notna()
        & pedidos["tiempo_triangulacion"].notna()
        & tiene_transito
    ]

    opciones = []
    for pedido in consolidables.itertuples():
        inicio = operation_start(pd.Timestamp(pedido.fecha_captura).date())
        limite = cal.due_date(inicio, pedido.promesa_estricta, valor_sabado=0)
        transito = int(tiempos_ftl[(pedido.sede, pedido.nodo_regional)])  # timedelta no acepta enteros de numpy

        for salida in possible_departure_days(inicio, espera_maxima):
            entrega = delivery_date(salida, transito, manipulacion, pedido.tiempo_triangulacion, pedido.llega_directo)
            opciones.append({
                "id_pedido": pedido.id_pedido,
                "dia_salida": salida,
                "espera": cal.business_days_between(inicio, salida),
                "fecha_entrega": entrega,
                "fecha_limite": limite,
                "retraso": delay(entrega, limite, valor_sabado=0),
            })

    return pd.DataFrame(opciones, columns=COLUMNAS_OPCIONES)
