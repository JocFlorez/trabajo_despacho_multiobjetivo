"""Búsqueda tabú: mínimo costo (Z1) con el castigo por retraso (Z2) acotado por ε, un punto de la frontera.

Parte de un plan (todo directo o la solución del MILP de un punto vecino) y en cada iteración prueba una muestra
aleatoria de movimientos, evaluados por región: de un pedido (subir, bajar o cambiar de día) o de un camión
completo (abrir, llenar, recortar o cerrar). El pedido o el camión que se mueve queda tabú unas iteraciones, salvo que
el movimiento dé el mejor plan encontrado (aspiración). El límite Z2 <= ε se maneja con un castigo adaptativo
(Cordeau, Gendreau y Laporte, 1997).
"""

import math
import random
import time

import numpy as np
import pandas as pd

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import (departure_days, evaluate_by_region,
                                                                    plan_departures, prepare_regions,
                                                                    reevaluate_region)

# Holgura para decidir si un plan cumple Z2 <= ε: el MILP deja Z2 pegado a ε con error de redondeo del solver
# (por ejemplo 0,8000008 con ε = 0,8), y ese plan sí cumple
TOLERANCIA_EPSILON = 1e-6

# Movimientos que cambian de una vez varios pedidos de una región en un día
TIPOS_CAMION = ("abrir", "llenar", "recortar", "cerrar")

def random_move(plan: dict, dias_por_pedido: dict, rng: random.Random) -> tuple:
    # Un movimiento al azar: (tipo, id_pedido, dia_nuevo)
    #   ("subir", id, dia)    el pedido va directo y pasa al camión de ese día
    #   ("bajar", id, None)   el pedido va en camión y pasa a directo
    #   ("cambiar", id, dia)  el pedido va en camión y pasa a otro de sus días
    # rng: random.Random(semilla), para que cada corrida se pueda repetir

    # 1. Un pedido al azar entre los que se pueden mover
    id_pedido = rng.choice(list(dias_por_pedido))
    dias = dias_por_pedido[id_pedido]

    # 2. Va directo: subirlo a uno de sus días
    if id_pedido not in plan:
        return ("subir", id_pedido, rng.choice(dias))

    # 3. Va en camión: bajarlo o pasarlo a otro de sus días
    salida = plan[id_pedido]
    if isinstance(salida, dict):
        otros = list(dias)                                  # traslado partido: queda completo en cualquiera de sus días
    else:
        otros = [dia for dia in dias if dia != salida]      # sale completo: su día actual no cuenta
    destino = rng.choice([None] + otros)
    if destino is None:
        return ("bajar", id_pedido, None)
    return ("cambiar", id_pedido, destino)

def truck_slots(instancia: dict) -> dict:
    # Camiones posibles por (region, dia), con lo que necesitan los movimientos de camión:
    #   pedidos: [(id_pedido, peso_kg, ahorro), ...] de mayor a menor ahorro por kilo,
    #            con ahorro = flete directo − flete del último tramo si sale ese día
    #   capacidades: de los vehículos que se usan hacia la sede regional, de menor a mayor
    #   costo y capacidad: la tabla de truck_cost_table de la región (costo y kg que caben para cada carga)
    # Solo cuentan las regiones con camión (las de instancia["tablas"]), igual que en departure_days.

    # 1. Opciones con camión y su ahorro por kilo, de mayor a menor
    opciones = instancia["opciones"]
    con_camion = (opciones[opciones["region_logistica"].isin(list(instancia["tablas"]))].reset_index()
                  .join(instancia["pedidos"]["flete_directo"], on="id_pedido"))
    con_camion["ahorro"] = con_camion["flete_directo"] - con_camion["flete_triangulacion"]
    con_camion["ahorro_kg"] = con_camion["ahorro"] / con_camion["peso_kg"]
    con_camion = con_camion.sort_values("ahorro_kg", ascending=False, kind="stable")

    # 2. Un registro por camión
    camiones = {}
    for (region, dia), grupo in con_camion.groupby(["region_logistica", "dia_salida"], sort=False):
        costo, capacidad, camion_metido = instancia["tablas"][region]
        camiones[region, dia] = {
            "pedidos": list(zip(grupo["id_pedido"], grupo["peso_kg"].astype(float), grupo["ahorro"].astype(float))),
            "capacidades": sorted({instancia["capacidades"][v] for v in camion_metido if v is not None}),
            "costo": costo, "capacidad": capacidad}
    return camiones

def random_truck_move(plan: dict, camiones: dict, rng: random.Random) -> tuple | None:
    # Un movimiento de camión al azar: (tipo, (region, dia), ids)
    #   ("abrir", camion, ids)    camión vacío: suben los directos que más ahorran por kilo, hasta la capacidad
    #                             de un vehículo escogido al azar (con Turbo: lo que cabe sube, el resto sigue directo)
    #   ("llenar", camion, ids)   camión que ya sale: suben los directos que más ahorran por kilo y caben en su espacio libre
    #   ("recortar", camion, ids) bajan los que menos ahorran por kilo hasta que la carga quepa en camiones más baratos
    #   ("cerrar", camion, ids)   bajan todos los que salen ese día hacia esa región
    # Devuelve None si en ese camión no hay nada que subir ni que bajar.
    # Por el costo fijo del camión, mover un pedido suelto casi nunca baja el costo: hay que mover varios a la vez.

    # 1. Un camión (región, día) al azar
    camion = rng.choice(list(camiones))
    _, dia = camion
    datos = camiones[camion]

    # 2. Quiénes salen en él (con los kg que ocupan), su carga, y quiénes van directo y ahorrarían subiendo
    #    (las dos listas quedan de mayor a menor ahorro por kilo, como en datos["pedidos"])
    en_camion, directos, carga = [], [], 0.0
    for id_pedido, peso, ahorro in datos["pedidos"]:
        salida = plan.get(id_pedido)
        if salida is None:
            if ahorro > 0:
                directos.append((id_pedido, peso))
        elif salida == dia or (isinstance(salida, dict) and dia in salida):
            kg = peso * salida[dia] if isinstance(salida, dict) else peso   # traslado partido: solo su fracción
            en_camion.append((id_pedido, kg))
            carga += kg
    w = math.ceil(carga - 1e-6)   # misma tolerancia que buy_trucks
    posibles = []

    # 3. Subir: si el camión está vacío, hasta la capacidad de un vehículo al azar; si ya sale, en su espacio libre
    if directos:
        if en_camion:
            tipo, libre = "llenar", datos["capacidad"][w] - carga
        else:
            tipo, libre = "abrir", rng.choice(datos["capacidades"])
        suben = []
        for id_pedido, peso in directos:
            if peso <= libre:
                suben.append(id_pedido)
                libre -= peso
        if suben:
            posibles.append((tipo, camion, tuple(suben)))

    # 4. Bajar: recortar hasta la mayor carga que se lleva con camiones más baratos, o cerrar
    if en_camion:
        meta = np.flatnonzero(datos["costo"][:w] < datos["costo"][w]).max() if w > 0 else 0
        if meta > 0:   # con meta 0 recortar sería cerrar
            bajan = []
            for id_pedido, kg in reversed(en_camion):   # de menor a mayor ahorro por kilo
                if carga <= meta:
                    break
                bajan.append(id_pedido)
                carga -= kg
            if len(bajan) < len(en_camion):
                posibles.append(("recortar", camion, tuple(bajan)))
        posibles.append(("cerrar", camion, tuple(id_pedido for id_pedido, _ in en_camion)))

    return rng.choice(posibles) if posibles else None

def apply_truck_move(plan: dict, movimiento: tuple) -> dict:
    # Plan nuevo con un movimiento de camión aplicado; el plan que llega NO se modifica
    tipo, (_, dia), ids = movimiento
    vecino = dict(plan)
    for id_pedido in ids:
        if tipo in ("abrir", "llenar"):
            vecino[id_pedido] = dia                     # sale completo ese día
        else:
            salida = vecino.pop(id_pedido)              # vuelve a ir directo
            if isinstance(salida, dict) and len(salida) > 1:
                vecino[id_pedido] = {d: f for d, f in salida.items() if d != dia}   # partido: conserva sus otros días
    return vecino

def apply_move(plan: dict, movimiento: tuple) -> dict:
    # Plan nuevo con el movimiento aplicado; el plan que llega NO se modifica
    # (la tabú compara muchos vecinos del mismo plan, así que cada uno necesita su propia copia)
    tipo, id_pedido, dia_nuevo = movimiento
    if tipo in TIPOS_CAMION:
        return apply_truck_move(plan, movimiento)

    # 1. Copia del plan
    vecino = dict(plan)

    # 2. "bajar": sale del plan y vuelve a ir directo; "subir" o "cambiar": sale completo ese día
    if tipo == "bajar":
        vecino.pop(id_pedido)
    else:
        vecino[id_pedido] = dia_nuevo

    # 3. El vecino
    return vecino


def penalized_cost(z1: float, z2: float, epsilon: float, alfa: float) -> float:
    # Costo castigado con el que la tabú compara vecinos: Z1 + α × (lo que Z2 se pasa de ε)
    # Si el plan cumple (z2 <= epsilon) no hay multa y el costo castigado es Z1.
    # alfa: pesos por cada día equivalente promedio por pedido de exceso

    # 1. Exceso: lo que Z2 se pasa de ε (0 si cumple)
    exceso = max(0.0, z2 - epsilon)

    # 2. Costo + multa por el exceso
    return z1 + alfa * exceso

def adjust_penalty_weight(alfa: float, cumple: bool, delta: float, alfa_min: float, alfa_max: float) -> float:
    # Ajuste del peso del castigo después de cada iteración (Cordeau, Gendreau y Laporte, 1997):
    #   si el plan actual se pasa del límite, α × (1 + δ): pasarse sale más caro y la búsqueda vuelve a lo factible
    #   si cumple, α ÷ (1 + δ): pasarse sale más barato y la búsqueda se asoma al otro lado del límite
    # Así la búsqueda oscila alrededor del borde Z2 = ε, donde está el óptimo.
    # alfa_min y alfa_max evitan que α llegue a 0 (ignorar el límite) o a infinito (no volver a cruzarlo).

    # 1. Bajar si cumple, subir si se pasa
    if cumple:
        alfa = alfa / (1 + delta)
    else:
        alfa = alfa * (1 + delta)

    # 2. Dentro de los límites
    return min(max(alfa, alfa_min), alfa_max)


def sample_moves(plan: dict, dias_por_pedido: dict, rng: random.Random, tamano: int,
                 camiones: dict | None = None, prob_camion: float = 0.0) -> list:
    # Muestra de movimientos DISTINTOS a partir del plan (la lista de candidatos de cada iteración).
    # Si el plan tiene menos movimientos posibles que 'tamano', devuelve los que encuentre en 10 × tamano intentos.
    # prob_camion: probabilidad de que cada intento sea un movimiento de camión (camiones = truck_slots);
    #   con 0 no se sortea nada extra, así que las corridas anteriores dan lo mismo
    movimientos = []
    vistos =set()

    # 1. Pedir movimientos hasta tener 'tamano' distintos o agotar los intentos
    for _ in range(10*tamano):
        if prob_camion > 0 and rng.random() < prob_camion:
            movimiento = random_truck_move(plan, camiones, rng)
        else:
            movimiento = random_move(plan, dias_por_pedido, rng)
        if movimiento is not None and movimiento not in vistos:
            vistos.add(movimiento)
            movimientos.append(movimiento)
        if len(movimientos) == tamano:
            break

    #2. En el orden en que salieron (con la misma semilla, la misma muestra)
    return movimientos

def is_tabu(movimiento: tuple, tabu_hasta: dict, iteraction: int) -> bool:
    # ¿El movimiento devuelve un pedido a un lugar del que salió hace poco?
    # tabu_hasta: {(id_pedido, origen): iteración hasta la que está prohibido volver ahí}
    #   origen es el día del que salió, o None si salió de directo
    # El destino de "bajar" es None (directo), igual que el origen de un pedido que estaba directo:
    # así una sola regla sirve para los tres tipos de movimiento.
    # Un camión movido hace poco no se puede volver a mover: clave ("camion", region, dia)
    tipo, id_pedido, destino = movimiento
    if tipo in TIPOS_CAMION:
        region, dia = id_pedido
        return tabu_hasta.get(("camion", region, dia), 0) > iteraction

    # Prohibido si volver a ese destino tiene un plazo que todavía no se cumple
    return tabu_hasta.get((id_pedido, destino), 0) > iteraction

def prepare_tabu(instancia: dict, filtrar_retraso: bool = True) -> dict:
    # Lo que la tabú calcula una sola vez por instancia; el memético lo reutiliza en todas sus tabúes cortas
    return {"dias_por_pedido": departure_days(instancia), "camiones": truck_slots(instancia),
            "regiones": prepare_regions(instancia, filtrar_retraso), "filtrar_retraso": filtrar_retraso}

def move_region(movimiento: tuple, regiones: dict) -> str:
    # Región que cambia con el movimiento: la del camión, o la del pedido que se mueve
    tipo, objetivo, _ = movimiento
    if tipo in TIPOS_CAMION:
        region, _ = objetivo
        return region
    return regiones["region_de"][objetivo]

def tabu_search(instancia: dict, epsilon: float, plan_inicial: dict | None = None, semilla: int = 0,
                tamano_muestra: int = 100, tenure: int = 10, delta: float = 0.1,
                max_segundos: float = 600, max_sin_mejora: int = 100,
                filtrar_retraso: bool = True, mostrar: bool = True, prob_camion: float = 0.0,
                preparado: dict | None = None) -> dict:
    # Mínimo costo (Z1) con Z2 <= ε, un punto de la frontera.
    # prob_camion: parte de la muestra que son movimientos de camión (0 = solo pedidos sueltos, como antes)
    # preparado: lo de prepare_tabu, para no recalcularlo en cada llamada (el memético llama la tabú muchas veces)
    # plan_inicial: {} o None = todo directo; o el plan del MILP de un punto vecino (plan_from_milp)
    # Devuelve {"plan", "z1", "z2", "factible", "iteraciones", "evaluaciones", "segundos", "semilla", "historial"}
    if epsilon <= 0:
        raise ValueError(f"epsilon debe ser positivo (es el límite del castigo promedio por pedido): {epsilon}")

    # ---- Preparación
    # 1. Reloj, sorteos repetibles y lo que no cambia en la búsqueda (días de cada pedido, camiones y regiones)
    inicio = time.perf_counter()
    rng = random.Random(semilla)
    preparado = preparado or prepare_tabu(instancia, filtrar_retraso)
    if preparado["filtrar_retraso"] != filtrar_retraso:
        raise ValueError("lo preparado se calculó con otro filtrar_retraso: el relleno daría costos equivocados")
    dias_por_pedido, camiones, regiones = preparado["dias_por_pedido"], preparado["camiones"], preparado["regiones"]

    # 2. Plan actual (se revisa una vez que sea válido: si no, ValueError) y su evaluación región por región
    plan_actual = dict(plan_inicial or {})
    plan_departures(plan_actual, instancia)
    actual = evaluate_by_region(plan_actual, regiones)
    evaluaciones = 1

    # 3. Peso del castigo: pasarse 1 % de ε cuesta como 1 % del costo; límites para que no llegue a 0 ni a infinito
    alfa = actual["z1"] / epsilon
    alfa_min, alfa_max = alfa / 1000, alfa * 1000

    # 4. Mejor plan que cumple el límite (None mientras no haya ninguno)
    mejor = None
    if actual["z2"] <= epsilon + TOLERANCIA_EPSILON:
        mejor = {"plan": plan_actual, "z1": actual["z1"], "z2": actual["z2"]}

    # 5. Memoria tabú y contadores
    tabu_hasta = {}
    historial = []
    iteracion = 0
    sin_mejora = 0

    # ---- Ciclo
    while True:
        iteracion += 1

        # 6. Muestra de vecinos, cada uno con su costo castigado
        candidatos = []
        for movimiento in sample_moves(plan_actual, dias_por_pedido, rng, tamano_muestra, camiones, prob_camion):
            vecino = apply_move(plan_actual, movimiento)
            resultado = reevaluate_region(vecino, move_region(movimiento, regiones), actual, regiones)   # solo la región que cambió
            f = penalized_cost(resultado["z1"], resultado["z2"], epsilon, alfa)
            candidatos.append((f, movimiento, vecino, resultado))
        evaluaciones += len(candidatos)

        # 7. Admisibles: no son tabú, o son tabú pero cumplen la aspiración (cumplen el límite y son récord)
        admisibles = []
        for candidato in candidatos:
            _, movimiento, _, resultado = candidato
            cumple = resultado["z2"] <= epsilon + TOLERANCIA_EPSILON
            es_record = cumple and (mejor is None or resultado["z1"] < mejor["z1"])
            if not is_tabu(movimiento, tabu_hasta, iteracion) or es_record:
                admisibles.append(candidato)

        # 8. El de menor costo castigado, AUNQUE sea peor que el actual (si todos son tabú, el menor de todos)
        _, movimiento, vecino, resultado = min(admisibles or candidatos, key=lambda candidato: candidato[0])

        # 9. Prohibir el regreso del pedido a su origen (un traslado partido no se puede volver a partir: no se registra)
        #    o, si se movió un camión, volver a moverlo
        tipo, id_pedido, _ = movimiento
        if tipo in TIPOS_CAMION:
            region, dia = id_pedido
            tabu_hasta["camion", region, dia] = iteracion + tenure
        else:
            origen = plan_actual.get(id_pedido)
            if not isinstance(origen, dict):
                tabu_hasta[id_pedido, origen] = iteracion + tenure

        # 10. Pasar al vecino
        plan_actual, actual = vecino, resultado

        # 11. ¿Nuevo mejor? Solo si cumple el límite
        cumple = actual["z2"] <= epsilon + TOLERANCIA_EPSILON
        if cumple and (mejor is None or actual["z1"] < mejor["z1"]):
            mejor = {"plan": plan_actual, "z1": actual["z1"], "z2": actual["z2"]}
            sin_mejora = 0
        else:
            sin_mejora += 1

        # 12. Ajustar el castigo
        alfa = adjust_penalty_weight(alfa, cumple, delta, alfa_min, alfa_max)

        # 13. Historial (para la curva de convergencia)
        segundos = time.perf_counter() - inicio
        historial.append({"iteracion": iteracion, "z1": actual["z1"], "z2": actual["z2"], "cumple": cumple,
                          "alfa": alfa, "z1_mejor": mejor["z1"] if mejor else None, "segundos": segundos})
        if mostrar and iteracion % 10 == 0:
            texto_mejor = f"{mejor['z1']:,.0f}" if mejor else "sin plan que cumpla"
            print(f"   iteración {iteracion} · Z1 {actual['z1']:,.0f} · Z2 {actual['z2']:.4f} · "
                  f"mejor {texto_mejor} · {segundos:.0f} s")

        # 14. Parar con lo primero que ocurra
        if segundos >= max_segundos or sin_mejora >= max_sin_mejora:
            break

    # ---- Resultado
    if mejor is None:
        print(f"ADVERTENCIA: la búsqueda tabú no encontró un plan con Z2 <= {epsilon:.4f}")
        mejor = {"plan": None, "z1": None, "z2": None}
    return {**mejor, "factible": mejor["plan"] is not None, "iteraciones": iteracion, "evaluaciones": evaluaciones,
            "segundos": time.perf_counter() - inicio, "semilla": semilla, "historial": pd.DataFrame(historial)}