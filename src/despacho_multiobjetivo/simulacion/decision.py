"""Decisión diaria con futuros (fase 4 del Monte Carlo): piezas comunes de las políticas A y C.

Cada pedido se describe con un diccionario, más rápido que una fila de pandas cuando hay muchos futuros:
  id, peso, flete_directo, flete_tramo (último tramo si va en camión), llegada, ultimo_dia,
  castigo_camion {día: castigo si sale ese día en camión} (solo los días con opción de camión) y
  castigo_directo {día: castigo si sale directo ese día} (sube si esperó: directo tardío)
El ahorro de un pedido es flete_directo − flete_tramo. Las regiones son independientes: todo se calcula por región.
"""
import math
from datetime import timedelta

from despacho_multiobjetivo.reglas_negocio import calendario as cal


def decision_window(conocidos: list, dia) -> list:
    # Días de los futuros: desde el siguiente día de despacho hasta el último día en que puede salir algún conocido
    # (un pedido que llegue después ya no puede compartir camión con ellos)
    if not conocidos:
        return []
    hasta = max(pedido["ultimo_dia"] for pedido in conocidos)
    dias, fecha = [], dia + timedelta(days=1)
    while fecha <= hasta:
        if cal.business_day_factor(fecha) > 0:   # sábado sí (medio día), domingo y festivos no
            dias.append(fecha)
        fecha += timedelta(days=1)
    return dias


def dispatch_when_pays(pedidos: list, dias: list, tabla: tuple) -> dict:
    # R1, la regla de los días siguientes en cada futuro: cada día, entre lo que ya llegó y no ha salido, se busca la
    # carga que más gana (ahorro de los que suben − costo de sus camiones), sumando pedidos de mayor a menor ahorro
    # por kilo; si gana algo, el camión sale ese día. Lo que llega a su último día sin camión sale directo.
    # tabla: la de truck_cost_table de la región (costo, capacidad, camion_metido); solo se usa el costo
    # Devuelve {"costo", "castigo", "salidas": {id: (día, "camion" | "directo")}}
    costo_camiones = tabla[0]
    salidas, costo, castigo = {}, 0.0, 0.0
    pendientes = list(pedidos)

    for dia in dias:
        # 1. Lo que ya llegó, no ha salido, puede ir en camión hoy y ahorra algo, de mayor a menor ahorro por kilo
        candidatos = sorted((p for p in pendientes
                             if p["llegada"] <= dia and dia in p["castigo_camion"] and p["flete_directo"] > p["flete_tramo"]),
                            key=lambda p: -(p["flete_directo"] - p["flete_tramo"]) / p["peso"])

        # 2. La carga más conveniente: la de mayor ganancia al ir sumando pedidos en ese orden
        mejor_ganancia, mejor_n, mejor_kg, kg, ahorro = 0.0, 0, 0, 0.0, 0.0
        for n, pedido in enumerate(candidatos, start=1):
            kg += pedido["peso"]
            ahorro += pedido["flete_directo"] - pedido["flete_tramo"]
            w = math.ceil(kg - 1e-6)   # misma tolerancia que buy_trucks
            if w >= len(costo_camiones):
                break                  # más carga de la que cubre la tabla
            if ahorro - costo_camiones[w] > mejor_ganancia:
                mejor_ganancia, mejor_n, mejor_kg = ahorro - costo_camiones[w], n, w

        # 3. Si alguna carga paga su camión, sale hoy
        if mejor_n:
            suben = candidatos[:mejor_n]
            costo += float(costo_camiones[mejor_kg]) + sum(p["flete_tramo"] for p in suben)
            castigo += sum(p["castigo_camion"][dia] for p in suben)
            salidas.update({p["id"]: (dia, "camion") for p in suben})
            ids = {p["id"] for p in suben}
            pendientes = [p for p in pendientes if p["id"] not in ids]

        # 4. Lo que vence hoy sin camión sale directo (tarde, si esperó)
        for pedido in [p for p in pendientes if p["llegada"] <= dia and p["ultimo_dia"] <= dia]:
            costo += pedido["flete_directo"]
            castigo += pedido["castigo_directo"][dia]
            salidas[pedido["id"]] = (dia, "directo")
        pendientes = [p for p in pendientes if not (p["llegada"] <= dia and p["ultimo_dia"] <= dia)]

    # 5. Borde: lo que vence después de la ventana queda directo el día en que llegó, sin cobrarle espera
    #    (se vuelve a decidir cuando llegue de verdad)
    for pedido in pendientes:
        costo += pedido["flete_directo"]
        castigo += pedido["castigo_directo"][pedido["llegada"]]
        salidas[pedido["id"]] = (pedido["llegada"], "directo")
    return {"costo": costo, "castigo": castigo, "salidas": salidas}


def _lote(decision: dict, conocidos: list, futuros: list, dia, dias: list, tabla: tuple,
          solo_conocidos: bool = False) -> tuple:
    # Una decisión de hoy para los conocidos de una región, probada en cada futuro (como _lote del taller):
    # hoy se ejecuta tal cual y R1 decide los días siguientes con lo que espera más lo que llega en ese futuro.
    # decision: {id: "directo" | "camion" | "esperar"}; futuros: una lista de pedidos por futuro
    # Devuelve (costos, retrasos), uno por futuro; retraso = castigo total / número de pedidos (el Z2 de la ventana)
    # solo_conocidos: el retraso se mide solo en los conocidos, que son los que decide la decisión de hoy
    costo_camiones = tabla[0]

    # 1. Lo de hoy, igual en todos los futuros
    costo_hoy, castigo_hoy, kg_camion, esperan = 0.0, 0.0, 0.0, []
    for pedido in conocidos:
        modo = decision[pedido["id"]]
        if modo == "esperar" and pedido["ultimo_dia"] > dia:
            esperan.append(pedido)
        elif modo == "camion":
            if dia not in pedido["castigo_camion"]:
                raise ValueError(f"{pedido['id']} no puede salir en camión el {dia}")
            costo_hoy += pedido["flete_tramo"]
            castigo_hoy += pedido["castigo_camion"][dia]
            kg_camion += pedido["peso"]
        else:   # directo, o esperar en su último día (sale directo, como en el simulador)
            costo_hoy += pedido["flete_directo"]
            castigo_hoy += pedido["castigo_directo"][dia]
    if kg_camion > 0:
        costo_hoy += float(costo_camiones[math.ceil(kg_camion - 1e-6)])

    # 2. Cada futuro: R1 con lo que espera más lo que llega
    costos, retrasos = [], []
    for futuro in futuros:
        resto = dispatch_when_pays(esperan + list(futuro), dias, tabla)
        costos.append(costo_hoy + resto["costo"])
        if solo_conocidos:
            castigo_esperan = 0.0
            for pedido in esperan:
                salida, modo = resto["salidas"][pedido["id"]]
                castigo_esperan += (pedido["castigo_camion"] if modo == "camion" else pedido["castigo_directo"])[salida]
            retrasos.append((castigo_hoy + castigo_esperan) / len(conocidos))
        else:
            retrasos.append((castigo_hoy + resto["castigo"]) / (len(conocidos) + len(futuro)))
    return costos, retrasos


def obj_promedio(costos: list, retrasos: list, epsilon: float, penalizacion: float) -> float:
    # Costo promedio entre los futuros, más un castigo si el retraso promedio se pasa de ε (como obj_promedio del
    # taller): el límite es sobre el promedio, no sobre cada futuro
    costo_promedio = sum(costos) / len(costos)
    retraso_promedio = sum(retrasos) / len(retrasos)
    return costo_promedio + max(0.0, retraso_promedio - epsilon) * penalizacion
