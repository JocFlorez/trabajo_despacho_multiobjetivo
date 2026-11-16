"""Simulador de horizonte rodante para el Monte Carlo: cada día decide sobre lo conocido y ejecuta solo lo de hoy."""

import math

import pandas as pd

from despacho_multiobjetivo.reglas_negocio import servicio
from despacho_multiobjetivo.reglas_negocio.opciones_salida import operation_start, possible_departure_days

DECISIONES = ("directo", "camion", "esperar")
COLUMNAS_REGISTRO = ["id_pedido", "dia_llegada", "dia_salida", "modo", "fraccion"]

def operation_days(pedidos: pd.DataFrame, espera_maxima: float) -> list:
    # Días de despacho del horizonte: todos los días en que algún pedido puede salir, desde el primero que llega
    # hasta la última salida posible del último (el horizonte pasa del fin de mes si hace falta)
    capturas = pd.to_datetime(pedidos["fecha_captura"]).dropna().dt.date.unique()
    return sorted({dia for captura in capturas for dia in possible_departure_days(captura, espera_maxima)})

def decide_all_direct(conocidos: pd.DataFrame, dia) -> dict:
    # Política de prueba: todo lo conocido sale directo hoy (debe reproducir la operación actual)
    return dict.fromkeys(conocidos["id_pedido"], "directo")

def executed_plan(registro: pd.DataFrame) -> dict:
    # De lo ejecutado al plan del evaluador: {id_pedido: dia_salida} de lo que salió en camión; lo demás va directo
    plan = {}
    for fila in registro[registro["modo"] == "camion"].itertuples():
        if fila.fraccion >= 1 - 1e-9:
            plan[fila.id_pedido] = fila.dia_salida
        else:
            plan.setdefault(fila.id_pedido, {})[fila.dia_salida] = fila.fraccion
    return plan

def known_orders(pedidos: pd.DataFrame, pendientes: pd.DataFrame, dia) -> pd.DataFrame:
    # Lo que se conoce el día 'dia': lo que sigue esperando más lo que llega hoy (dia_llegada == dia).
    # No modifica lo que recibe.
    # 1. Lo que llega hoy
    llegan_hoy = pedidos[pedidos["dia_llegada"] == dia]

    # 2. Juntarlo con lo pendiente, con las filas numeradas desde 0
    if pendientes.empty:
        return llegan_hoy.reset_index(drop=True) #pandas avisa si se junta una tabla vacía
    return pd.concat([pendientes, llegan_hoy], ignore_index=True)

def apply_decision(conocidos: pd.DataFrame, decision: dict, dia) -> tuple:
    # Ejecuta solo lo de hoy. decision: {id_pedido: "directo" | "camion" | "esperar"}
    # Devuelve (salen, pendientes): salen con COLUMNAS_REGISTRO; pendientes con las columnas de conocidos

    # 1. Revisar la decisión: una por pedido conocido, ninguna sobre pedidos que no se conocen, valores válidos
    ids = set(conocidos["id_pedido"])
    sin_decision = ids - set(decision)
    if sin_decision:
        raise ValueError(f"{len(sin_decision)} pedidos conocidos sin decisión: {sorted(sin_decision)}")
    desconocidos = set(decision) - ids
    if desconocidos:
        raise ValueError(f"Decisiones para {len(desconocidos)} pedidos que no se conocen: {sorted(desconocidos)}")
    invalidos = {modo for modo in decision.values() if modo not in DECISIONES}
    if invalidos:
        raise ValueError(f"decisión no válida: {sorted(invalidos)} (se esperaba una de {DECISIONES})")

    #2. Lo que la política deja esperando en su último día sale directo, con aviso
    modo = conocidos["id_pedido"].map(decision)
    vencidos = (modo == "esperar") & (conocidos["ultimo_dia"] <= dia)
    if vencidos.any():
        print(f"ADVERTENCIA: {vencidos.sum()} pedidos que vencen hoy salen directo")
        modo[vencidos] = "directo"

    #3. Lo que sale hoy: directo o en camión
    sale = conocidos[modo != "esperar"].copy()
    salen = pd.DataFrame({
        "id_pedido": sale["id_pedido"],
        "dia_llegada": sale["dia_llegada"],
        "dia_salida": pd.Series([dia] * len(sale), index=sale.index),
        "modo": modo[sale.index],
        "fraccion": sale["fraccion"]
    }).reset_index(drop=True)

    #4. Lo que queda pendiente: los que esperan y no vencen hoy
    pendientes = conocidos[modo == "esperar"].copy().reset_index(drop=True)
    return salen, pendientes

def simular_escenario(pedidos: pd.DataFrame, politica, espera_maxima: float, instancia: dict | None = None,
                      filtrar_retraso: bool = True) -> pd.DataFrame:
    # Recorre los días de operación: conoce, decide con politica(conocidos, dia) y ejecuta solo lo de hoy.
    # Devuelve el registro: una fila por pedido con COLUMNAS_REGISTRO. No modifica pedidos.

    # 1. Pedidos sin fecha de captura: fuera, con aviso (van directo, como en el modelo)
    sin_fecha = pedidos["fecha_captura"].isna()
    if sin_fecha.any():
        print(f"ADVERTENCIA: {sin_fecha.sum()} pedidos sin fecha de captura no entran al simulador")

    # 2. Copia con el día en que cada pedido entra a operación y el último día en que puede salir
    #    (se calcula una vez por fecha de captura, no por pedido)
    capturas = pd.to_datetime(pedidos.loc[~sin_fecha, "fecha_captura"]).dt.date
    llegada = {captura: operation_start(captura) for captura in capturas.unique()}
    ultimo = {captura: possible_departure_days(captura, espera_maxima)[-1] for captura in capturas.unique()}
    simulados = pedidos[~sin_fecha].assign(dia_llegada=capturas.map(llegada), ultimo_dia=capturas.map(ultimo),
                                         fraccion=1.0)

    #3. Cada día: conocer, decidir y ejecutar solo lo de hoy
    pendientes = simulados.iloc[:0]
    salidas = []
    for dia in operation_days(simulados, espera_maxima):
        conocidos = known_orders(simulados, pendientes, dia)
        if conocidos.empty:
            continue
        salen, pendientes = apply_decision(conocidos, politica(conocidos, dia), dia)
        if instancia is not None:
            salen, pendientes = fill_trucks_today(salen, pendientes, dia, instancia, filtrar_retraso)
        if not salen.empty:
            salidas.append(salen)

    #4. Nada puede quedar esperando: el último día de cada pedido está dentro del horizonte
    if not pendientes.empty:
        raise ValueError(f"{len(pendientes)} pedidos quedaron pendientes al final del horizonte: "
                         f"{pendientes['id_pedido'].tolist()[:5]}")

    #5. El registro, una fila por pedido
    if not salidas:
        return pd.DataFrame(columns=COLUMNAS_REGISTRO)
    return pd.concat(salidas, ignore_index=True)


def fill_trucks_today(salen: pd.DataFrame, pendientes: pd.DataFrame, dia, instancia: dict,
                      filtrar_retraso: bool = True) -> tuple:
    # Relleno (fase 3, opción B2): el espacio libre de los camiones que salen hoy se llena con fracciones de
    # traslados presentes hoy, los que salen directo hoy y los que siguen esperando, empezando por el que más ahorra
    # por kilo (como fill_trucks del evaluador). Nunca se compra un camión solo para rellenar.
    # La parte que sube sale hoy en camión; el resto sale directo hoy (si iba directo) o sigue esperando.
    # filtrar_retraso: solo traslados que en camión no llegan peor que directo (el mismo filtro del modelo)
    # Devuelve (salen, pendientes) nuevos; no modifica lo que recibe.
    pedidos, opciones = instancia["pedidos"], instancia["opciones"]

    # 1. Carga de los camiones que salen hoy, por región, y el espacio que les sobra
    carga = {}
    for fila in salen[salen["modo"] == "camion"].itertuples():
        region, peso = opciones.loc[(fila.id_pedido, dia), ["region_logistica", "peso_kg"]]
        carga[region] = carga.get(region, 0.0) + fila.fraccion * peso
    libre = {}
    for region, kg in carga.items():
        _, capacidad, _ = instancia["tablas"][region]
        libre[region] = float(capacidad[math.ceil(kg - 1e-6)]) - kg   # misma tolerancia que buy_trucks
    if not any(espacio > 0 for espacio in libre.values()):
        return salen.copy(), pendientes.copy()

    # 2. Candidatos: traslados presentes hoy con opción de salir hoy hacia una región con espacio
    candidatos = []
    for origen, tabla in (("directo", salen[salen["modo"] == "directo"]), ("esperando", pendientes)):
        for fila in tabla.itertuples():
            if not pedidos.loc[fila.id_pedido, "se_puede_partir"] or (fila.id_pedido, dia) not in opciones.index:
                continue
            opcion = opciones.loc[(fila.id_pedido, dia)]
            ahorro = pedidos.loc[fila.id_pedido, "flete_directo"] - opcion["flete_triangulacion"]
            llega_peor = opcion["castigo"] > pedidos.loc[fila.id_pedido, "castigo_directo"]
            if libre.get(opcion["region_logistica"], 0) <= 0 or ahorro <= 0 or (filtrar_retraso and llega_peor):
                continue
            candidatos.append({"id_pedido": fila.id_pedido, "origen": origen, "fraccion": fila.fraccion,
                               "region": opcion["region_logistica"], "peso": opcion["peso_kg"],
                               "ahorro_kg": ahorro / opcion["peso_kg"]})

    # 3. Llenar, de mayor a menor ahorro por kilo; en los empates, en el orden en que llegaron
    subidas = {}
    for candidato in sorted(candidatos, key=lambda c: -c["ahorro_kg"]):
        espacio = libre[candidato["region"]]
        if espacio <= 0:
            continue
        fraccion = min(espacio / candidato["peso"], candidato["fraccion"])
        subidas[candidato["id_pedido"]] = fraccion
        libre[candidato["region"]] = espacio - fraccion * candidato["peso"]
    if not subidas:
        return salen.copy(), pendientes.copy()

    # 4. Lo que sube sale hoy en camión; a lo directo y a lo pendiente se les descuenta lo que subió
    def descontar(tabla):
        tabla = tabla.copy()
        sube = tabla["id_pedido"].map(subidas).fillna(0.0)
        if "modo" in tabla:
            sube = sube.where(tabla["modo"] == "directo", 0.0)
        tabla["fraccion"] = tabla["fraccion"] - sube
        return tabla[tabla["fraccion"] > 1e-9]

    presentes = pd.concat([salen[salen["modo"] == "directo"], pendientes], ignore_index=True)
    llegada = dict(zip(presentes["id_pedido"], presentes["dia_llegada"]))
    suben = pd.DataFrame({"id_pedido": list(subidas), "dia_llegada": [llegada[i] for i in subidas],
                          "dia_salida": dia, "modo": "camion", "fraccion": list(subidas.values())})
    salen_nuevo = pd.concat([descontar(salen), suben], ignore_index=True)[COLUMNAS_REGISTRO]
    return salen_nuevo, descontar(pendientes).reset_index(drop=True)


def costo_escenario(registro: pd.DataFrame, pedidos: pd.DataFrame, instancia: dict) -> dict:
    # Z1 (costo) y Z2 (días de retraso ponderados por pedido) de lo que ejecutó el simulador.
    # Parte de "todo directo el primer día" y corrige con lo que dice el registro:
    #   - en camión: último tramo en vez de directo, el castigo de esa opción y la carga de su camión
    #   - directo: mismo flete; si esperó, el castigo del directo tardío
    # Lo que no pasó por el simulador (pedidos sin fecha) queda directo el primer día, como en el modelo.
    # El costo directo es el de la carga de datos: lo pagado (modo histórico) o lo cotizado (modo operación).
    info, opciones = instancia["pedidos"], instancia["opciones"]

    # 1. Base: todo directo el primer día
    z1 = float(info["flete_directo"].sum())
    castigo = float(info["castigo_directo"].sum())

    # 2. Lo que salió directo: el flete no cambia; el castigo sí, si esperó
    directos = registro[registro["modo"] == "directo"]
    if not directos.empty:
        datos = directos.merge(pedidos.drop(columns=["dia_llegada", "dia_salida"], errors="ignore"), on="id_pedido")
        tardio = servicio.expected_delay_late_direct(datos).to_numpy()
        castigo += float(((tardio - info.loc[datos["id_pedido"], "castigo_directo"].to_numpy())
                          * datos["fraccion"].to_numpy()).sum())

    # 3. Lo que salió en camión: último tramo y castigo de la opción, y la carga de cada camión
    carga = {}
    for fila in registro[registro["modo"] == "camion"].itertuples():
        opcion = opciones.loc[(fila.id_pedido, fila.dia_salida)]
        z1 += fila.fraccion * (opcion["flete_triangulacion"] - info.loc[fila.id_pedido, "flete_directo"])
        castigo += fila.fraccion * (opcion["castigo"] - info.loc[fila.id_pedido, "castigo_directo"])
        camion = (opcion["region_logistica"], fila.dia_salida)
        carga[camion] = carga.get(camion, 0.0) + fila.fraccion * opcion["peso_kg"]

    # 4. Los camiones: los de menor costo para la carga de cada región y día
    for (region, _), kg in carga.items():
        costo, _, _ = instancia["tablas"][region]
        z1 += float(costo[math.ceil(kg - 1e-6)])

    return {"z1": z1, "z2": castigo / instancia["n_pedidos"]}
