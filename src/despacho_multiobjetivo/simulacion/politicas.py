"""Políticas de decisión diaria del Monte Carlo (fase 4).

Política C (como el taller): cada día, para cada región, una tabú corta busca solo la decisión de hoy de los pedidos
conocidos (directo, camión o esperar). Cada decisión candidata se califica con _lote en N futuros sorteados del
histórico (los días siguientes se resuelven con la regla R1) y con obj_promedio (costo promedio, con castigo si el
retraso promedio pasa de ε). Se ejecuta solo lo de hoy; al día siguiente se vuelve a decidir.
El límite ε es relativo a cada región y día (corrección A, 2026-11-08): factor × el retraso de los conocidos si
hoy salieran todos directo, y se compara con el retraso que terminan teniendo esos mismos conocidos en cada futuro.
La decisión de hoy responde por los pedidos que decide; los futuros se juzgan cuando llegan. Un ε global aplicado
región por región obligaba a comprar camiones que no se pagan, y medir toda la ventana dejaba esperar de más.
"""
import random
import time

import pandas as pd

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import truck_cost_table
from despacho_multiobjetivo.reglas_negocio import calendario as cal
from despacho_multiobjetivo.reglas_negocio import servicio
from despacho_multiobjetivo.reglas_negocio.opciones_salida import (build_departure_options, operation_start,
                                                                   possible_departure_days)
from despacho_multiobjetivo.simulacion.decision import _lote, decision_window, obj_promedio
from despacho_multiobjetivo.simulacion.escenarios import generar_demanda

PESO_MAXIMO_TABLA = 60_000   # kg: los pendientes de varios días pueden superar la mayor carga de un solo día
PASO_CASTIGO = 0.1           # castigo fuerte (decidido): pasarse 10 % de ε cuesta tanto como la ventana todo directo


def policy_tables(capacidades: dict, tarifas_ftl: dict, nodos_regionales: dict, sede: str,
                  peso_maximo: float = PESO_MAXIMO_TABLA) -> dict:
    # Tabla de camiones de cada región con sede regional, hasta peso_maximo kg (la de la instancia llega solo
    # hasta la mayor carga de un día, y en la decisión se juntan pedidos de varios días)
    tablas = {}
    for region, hub in nodos_regionales.items():
        if hub is None or pd.isna(hub):
            continue
        tabla = truck_cost_table(capacidades, tarifas_ftl, sede, hub, peso_maximo)
        if tabla is not None:
            tablas[region] = tabla
    return tablas


def decision_orders(pedidos: pd.DataFrame, castigos: pd.Series, espera_maxima: float, regiones: set) -> dict:
    # Pedidos de la decisión (diccionarios, ver decision.py), agrupados por región: {region: [pedido, ...]}
    # pedidos: id_pedido, peso_kg, flete_sin_consolidar, flete_triangulacion, region_logistica, fecha_captura y las
    #   columnas de la transportadora; si las trae, dia_llegada, ultimo_dia y fraccion (la parte aún pendiente)
    # castigos: castigo de cada opción de camión, por (id_pedido, dia_salida)
    # Solo entran los pedidos de regiones con camión y con alguna opción de camión; los demás van directo.
    dias_camion = {}
    for (id_pedido, dia), castigo in castigos.items():
        dias_camion.setdefault(id_pedido, {})[dia] = castigo
    cumplimiento = servicio.on_time_probability_direct(pedidos).to_numpy()

    castigo_espera = {}   # (cumplimiento, días esperados) → castigo; se repite mucho entre pedidos
    por_region = {}
    for fila, prob in zip(pedidos.itertuples(index=False), cumplimiento):
        if fila.region_logistica not in regiones or fila.id_pedido not in dias_camion:
            continue
        llegada = getattr(fila, "dia_llegada", None) or operation_start(pd.Timestamp(fila.fecha_captura).date())
        posibles = possible_departure_days(llegada, espera_maxima)
        ultimo = getattr(fila, "ultimo_dia", None) or posibles[-1]
        f = getattr(fila, "fraccion", 1.0)

        directo = {}
        for dia in posibles:
            clave = (prob, cal.business_days_between(llegada, dia, valor_sabado=0))
            if clave not in castigo_espera:
                castigo_espera[clave] = servicio.expected_delay_penalty(*clave)
            directo[dia] = f * castigo_espera[clave]

        por_region.setdefault(fila.region_logistica, []).append({
            "id": fila.id_pedido, "peso": f * fila.peso_kg,
            "flete_directo": f * fila.flete_sin_consolidar, "flete_tramo": f * fila.flete_triangulacion,
            "llegada": llegada, "ultimo_dia": ultimo, "castigo_directo": directo,
            "castigo_camion": {dia: f * c for dia, c in dias_camion[fila.id_pedido].items()}})
    return por_region


def future_orders(futuro: pd.DataFrame, tiempos_ftl: dict, espera_maxima: float, regiones: set) -> dict:
    # Los pedidos de un futuro sorteado, como pedidos de la decisión agrupados por región
    opciones = build_departure_options(futuro, tiempos_ftl, espera_maxima)
    if opciones.empty:
        return {}
    return decision_orders(futuro, servicio.expected_delay_options(opciones, futuro), espera_maxima, regiones)


def direct_delay(conocidos: list, futuros: list, dia, dias: list, tabla: tuple) -> float:
    # Retraso promedio de los conocidos si hoy salieran todos directo (con la espera que ya llevan): el límite de
    # servicio de la decisión de hoy, medido con la misma cuenta que las candidatas
    _, retrasos = _lote(dict.fromkeys((p["id"] for p in conocidos), "directo"), conocidos, futuros or [[]], dia, dias,
                        tabla, solo_conocidos=True)
    return sum(retrasos) / len(retrasos)


def best_today_decision(conocidos: list, futuros: list, dia, dias: list, tabla: tuple, epsilon: float,
                        max_segundos: float, tenure: int = 3, max_sin_mejora: int = 5) -> dict:
    # Tabú corta sobre la decisión de hoy de los conocidos de una región. Cada candidata se califica con
    # obj_promedio(_lote(...)); en un empate gana la de menor retraso promedio y luego la que deja esperar menos.
    # Arranca de "todos directo", que cumple el límite. Vecinos: cambiar la decisión de un pedido, o mover a todos a
    # la vez (todos al camión, todos esperan, todos directo): por el costo fijo del camión, devolver los pedidos uno
    # por uno a directo sube el costo, y sin el movimiento de bloque la búsqueda queda atrapada en un camión que no
    # se paga.
    posibles = {p["id"]: [m for m in ("esperar", "directo", "camion")
                          if (m != "esperar" or p["ultimo_dia"] > dia) and (m != "camion" or dia in p["castigo_camion"])]
                for p in conocidos}
    base = sum(p["flete_directo"] for p in conocidos) + sum(
        sum(p["flete_directo"] for p in futuro) for futuro in futuros) / max(len(futuros), 1)
    penalizacion = base / (PASO_CASTIGO * epsilon)

    def calificar(decision):
        costos, retrasos = _lote(decision, conocidos, futuros, dia, dias, tabla, solo_conocidos=True)
        return (obj_promedio(costos, retrasos, epsilon, penalizacion), sum(retrasos) / len(retrasos),
                sum(modo == "esperar" for modo in decision.values()))

    todos_esperan = {i: ("esperar" if "esperar" in opciones else "directo") for i, opciones in posibles.items()}
    todos_directo = dict.fromkeys(posibles, "directo")
    actual = dict(todos_directo)
    mejor, nota_mejor = dict(actual), calificar(actual)
    tabu, sin_mejora, iteracion, inicio = {}, 0, 0, time.perf_counter()

    while sin_mejora < max_sin_mejora and time.perf_counter() - inicio < max_segundos:
        iteracion += 1
        vecinos = [(i, {**actual, i: modo}) for i, opciones in posibles.items() for modo in opciones if modo != actual[i]]
        con_camion = [i for i, opciones in posibles.items() if "camion" in opciones]
        if con_camion:
            vecinos.append(("todos_camion", {**actual, **dict.fromkeys(con_camion, "camion")}))
        vecinos.append(("todos_esperan", dict(todos_esperan)))
        vecinos.append(("todos_directo", dict(todos_directo)))
        calificados = [(calificar(vecino), clave, vecino) for clave, vecino in vecinos if vecino != actual]
        if not calificados:
            break
        admisibles = [c for c in calificados if tabu.get(c[1], 0) <= iteracion or c[0] < nota_mejor]
        nota, clave, actual = min(admisibles or calificados, key=lambda c: c[0])
        tabu[clave] = iteracion + tenure
        if nota < nota_mejor:
            mejor, nota_mejor, sin_mejora = dict(actual), nota, 0
        else:
            sin_mejora += 1
    return mejor


def make_policy_c(pool: dict, instancia: dict, tiempos_ftl: dict, espera_maxima: float, factor_servicio: float,
                  n_futuros: int, tablas: dict, semilla: int = 0, max_segundos: float = 10):
    # Política C lista para simular_escenario: politica(conocidos, dia) -> {id_pedido: decisión}
    # pool: banco de días (build_day_pool); instancia: la del mes simulado (sus opciones traen el castigo)
    # factor_servicio: ε de cada región y día = factor × su retraso con todo directo (1,0 = el mismo servicio de hoy)
    # tablas: policy_tables; max_segundos: tiempo de la decisión de cada día, repartido entre regiones
    rng = random.Random(semilla)
    regiones = set(tablas)
    castigos = instancia["opciones"]["castigo"]

    def politica(conocidos: pd.DataFrame, dia) -> dict:
        decision = dict.fromkeys(conocidos["id_pedido"], "directo")   # sin camión posible: directo
        por_region = decision_orders(conocidos, castigos, espera_maxima, regiones)
        todos = [pedido for lista in por_region.values() for pedido in lista]
        dias = decision_window(todos, dia)
        futuros = [future_orders(futuro, tiempos_ftl, espera_maxima, regiones)
                   for futuro in generar_demanda(pool, dias, n_futuros, rng)] if dias else []
        for region, lista in por_region.items():
            futuros_region = [futuro.get(region, []) for futuro in futuros] or [[]]
            segundos = max_segundos * len(lista) / len(todos)
            epsilon = factor_servicio * direct_delay(lista, futuros_region, dia, dias, tablas[region])
            decision.update(best_today_decision(lista, futuros_region, dia, dias, tablas[region], epsilon, segundos))
        return decision

    return politica
