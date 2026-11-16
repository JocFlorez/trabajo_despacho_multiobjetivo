"""Escenarios de demanda para el Monte Carlo: remuestreo de días completos del histórico."""
import random
from datetime import date

import pandas as pd

from despacho_multiobjetivo.reglas_negocio.opciones_salida import operation_start

LIMITES_TRAMOS = (10, 20) # días 1-10, 11-20 y 21-fin; si cambia el cierre, se cambia aquí o al llamar las funciones

def build_day_pool(pedidos: pd.DataFrame) -> dict:
    #Pedidos del historico agrupados por el dia en que entran a operacion: {fecha: pedidos de ese dia}
    #Un dia completo es la unidad que se remuestrea, asi se conserva lo que llega junto

    #1. Un pedido sin fecha de captura no se puede ubicar en ningun día
    sin_fecha = pedidos["fecha_captura"].isna()
    if sin_fecha.any():
        print(f"ADVERTENCIA: {sin_fecha.sum()} pedidos sin fecha de captura no entran al remuestreo")
    con_fecha = pedidos[~sin_fecha]

    #2. Dia de operacion: lo capturado en domingo o festivo entra al siguiente día de despacho
    dias = con_fecha["fecha_captura"].map(lambda fecha: operation_start(pd.Timestamp(fecha).date()))

    #3. Un grupo por dia, en orden cronológico
    return {dia: grupo.copy() for dia, grupo in con_fecha.groupby(dias, sort=True)}

def relabel_day(pedidos_dia: pd.DataFrame, fecha_nueva: date, etiqueta: str) -> pd.DataFrame:
    #Un día del histórico convertido en un día simulado: mismos pedidos, otra fecha e identificadores nuevos.
    #Etiqueta: distingue cada sorteo (por ejemplo "f3d2"), porque el mismo día puede salir varias veces.
    #El día que recibe no se modifica,

    #1.  Copia con las filas numeradas desde 0 para poder juntar varios días después
    nuevo = pedidos_dia.copy().reset_index(drop=True)

    #2. Identificador único: el modelo y el evaluador necesitan uno por pedido
    nuevo["id_pedido"] = nuevo["id_pedido"].astype(str) + "@" + etiqueta

    #3. Fecha simulada: la promesa y los días de salida se calculan a partir de ella
    nuevo["fecha_captura"] = pd.Timestamp(fecha_nueva)

    #4. La entrega histórica ocurrió en otra fecha: no aplica al pedido simulado
    if "fecha_entrega" in nuevo:
        nuevo["fecha_entrega"] = pd.NaT

    return nuevo

def day_stratum(fecha: date, limites_tramos: tuple = LIMITES_TRAMOS) -> tuple:
    # A qué grupo pertenece un día: (tramo del mes, día de la semana). Lunes = 0
    tramo = sum(fecha.day > limite for limite in limites_tramos)
    return tramo, fecha.weekday()


def candidate_days(pool: dict, fecha: date, limites_tramos: tuple = LIMITES_TRAMOS) -> list:
    # Días del histórico que se pueden sortear para una fecha: los del mismo tramo y día de la semana.
    # Si no hay, se amplía primero a todo el tramo y luego a todo el histórico, avisando
    tramo, dia_semana = day_stratum(fecha, limites_tramos)

    # 1. Mismo tramo y mismo día de la semana
    mismo_estrato = [dia for dia in pool if day_stratum(dia, limites_tramos) == (tramo, dia_semana)]
    if mismo_estrato:
        return mismo_estrato

    # 2. Mismo tramo
    mismo_tramo = [dia for dia in pool if day_stratum(dia, limites_tramos)[0] == tramo]
    if mismo_tramo:
        print(f"ADVERTENCIA: {fecha} no tiene días de ese día de la semana en su tramo; se sortea en todo el tramo")
        return mismo_tramo

    # 3. Todo el histórico
    print(f"ADVERTENCIA: {fecha} no tiene días de su tramo en el histórico; se sortea entre todos los días")
    return list(pool)


def generar_demanda(pool: dict, fechas: list, n_futuros: int, rng: random.Random,
                       limites_tramos: tuple = LIMITES_TRAMOS) -> list:
    # n_futuros posibles para las fechas pedidas. Cada futuro es una tabla con los pedidos de todos sus días.
    # rng: random.Random(semilla), para que el sorteo se pueda repetir

    # 1. Días candidatos de cada fecha (una sola vez, para no repetir los avisos)
    candidatos = {fecha: candidate_days(pool, fecha, limites_tramos) for fecha in fechas}

    # 2. En cada futuro, un día sorteado por fecha, con su fecha e identificadores nuevos
    futuros = []
    for f in range(n_futuros):
        dias = [relabel_day(pool[rng.choice(candidatos[fecha])], fecha, f"f{f}d{d}") for d, fecha in enumerate(fechas)]
        futuros.append(pd.concat(dias, ignore_index=True))
    return futuros

