"""Evaluador común de las metaheurísticas: costo y retraso de un plan sin pasar por el solver.

prepare_instance precalcula lo que no cambia entre planes; truck_cost_table, truck_mix y buy_trucks deciden
qué camiones se compran; fill_trucks rellena su espacio libre con traslados. La evaluación del plan y la
conversión desde la solución del MILP se agregan a continuación. prepare_regions y evaluate_by_region evalúan el plan
región por región, para que la tabú recalcule solo la región que cambia cada movimiento.
"""
import math

import numpy as np
import pandas as pd

from despacho_multiobjetivo.datos.data_processing import TIPOS_SIN_PARTIR
from despacho_multiobjetivo.modelos.componentes import enrich_options
from despacho_multiobjetivo.reglas_negocio import servicio


def prepare_instance(pedidos: pd.DataFrame, opciones: pd.DataFrame, capacidades: dict, tarifas_ftl: dict,
                     nodos_regionales: dict, sede: str) -> dict:
    # Lo que no cambia entre un plan y otro, calculado una sola vez al inicio de la metaheurística.
    # pedidos y opciones: los mismos que recibe model_multiobjetivo.build_model.
    # Devuelve un diccionario con:
    #   pedidos: por id_pedido → flete directo, castigo por retraso si va directo, peso, región y si se puede partir
    #   opciones: por (id_pedido, dia_salida) → región, peso, flete del último tramo y castigo si sale ese día
    #   tablas: {región: truck_cost_table hacia su sede regional}, hasta la mayor carga posible de esa región en un día
    #   capacidades y n_pedidos (Z2 es el promedio por pedido)
    info_pedidos = pd.DataFrame({
        "flete_directo": pedidos["flete_sin_consolidar"].values,
        "castigo_directo": servicio.expected_delay_direct(pedidos).values,
        "peso_kg": pedidos["peso_kg"].values,
        "region_logistica": pedidos["region_logistica"].values,
        "se_puede_partir": ~pedidos["tipo"].isin(TIPOS_SIN_PARTIR).values,   # traslados: pueden ir en fracciones
    }, index=pd.Index(pedidos["id_pedido"], name="id_pedido"))

    castigo_opciones = servicio.expected_delay_options(opciones, pedidos)
    info_opciones = (enrich_options(opciones, pedidos)
                     .set_index(["id_pedido", "dia_salida"])[["region_logistica", "peso_kg", "flete_triangulacion"]]
                     .assign(castigo=castigo_opciones))

    # La tabla de cada región llega hasta la mayor carga que podría salir un día: todos sus pedidos con opción ese día
    carga_maxima = (info_opciones.reset_index()
                    .groupby(["region_logistica", "dia_salida"])["peso_kg"].sum()
                    .groupby(level="region_logistica").max())
    tablas = {}
    for region, peso_maximo in carga_maxima.items():
        hub = nodos_regionales.get(region)
        tabla = truck_cost_table(capacidades, tarifas_ftl, sede, hub, peso_maximo) if hub is not None else None
        if tabla is not None:   # sin sede regional o sin tarifa: esa región no consolida
            tablas[region] = tabla

    return {"pedidos": info_pedidos, "opciones": info_opciones, "tablas": tablas,
            "capacidades": capacidades, "n_pedidos": len(pedidos)}


def truck_cost_table(capacidades: dict, tarifas_ftl: dict, sede: str, hub: str,
                     peso_maximo: float) -> tuple[np.ndarray, np.ndarray, list] | None:
    # costo[w]: costo mínimo de los camiones que llevan w kg de la sede al hub
    # capacidad[w]: kg que caben en esa combinación de camiones (sirve para el relleno de traslados)
    # camion_metido[w]: el camión que se metió en la combinación ganadora para w kg (truck_mix la reconstruye con él)
    # Programación dinámica: para cada carga se prueba meter cada tipo de camión v; lo que no cabe en él es una
    # carga más pequeña que ya está resuelta: costo[w] = mín sobre v de (tarifa_v + costo[máx(0, w − capacidad_v)]).
    # Se consulta con math.ceil(peso).

    # 1. Vehículos con tarifa de esta sede a este hub
    vehiculos = [(v, capacidades[v], tarifas_ftl[v, sede, hub]) for v in capacidades if (v, sede, hub) in tarifas_ftl]
    if not vehiculos:
        return None   # a este hub no va camión

    # 2. Arreglos de 0 a W kg
    W = math.ceil(peso_maximo)
    costo = np.full(W + 1, np.inf)
    costo[0] = 0
    capacidad = np.zeros(W + 1)
    camion_metido = [None] * (W + 1)

    # 3. Para cada carga, meter cada tipo de camión: su tarifa + lo más barato para lo que no cupo
    for w in range(1, W + 1):
        for v, cap_v, tarifa_v in vehiculos:
            resto = max(0, w - cap_v)
            candidato = tarifa_v + costo[resto]
            capacidad_candidato = cap_v + capacidad[resto]
            # Con el mismo costo se prefiere la combinación con más espacio: deja más hueco para el relleno
            if candidato < costo[w] or (candidato == costo[w] and capacidad_candidato > capacidad[w]):
                costo[w] = candidato
                capacidad[w] = capacidad_candidato
                camion_metido[w] = v

    return costo, capacidad, camion_metido


def truck_mix(camion_metido: list, capacidades: dict, peso: float) -> dict:
    # Qué camiones forman la combinación más barata para 'peso' kg: se toma el camión que se metió para esa carga,
    # se descuenta su capacidad y se repite con lo que no cupo, hasta llegar a 0.
    # Ejemplo: 37.000 kg → Mula (quedan 3.000) → Turbo (quedan 0) → {"Mula": 1, "Turbo": 1}
    camiones = {}
    w = math.ceil(peso)
    while w > 0:
        v = camion_metido[w]
        camiones[v] = camiones.get(v, 0) + 1
        w = max(0, w - capacidades[v])
    return camiones


def buy_trucks(carga: dict, tablas: dict, capacidades: dict) -> dict:
    # Camiones que se compran en cada (región, día) para la carga que dejó el plan, con la tabla de su sede regional.
    # carga: {(region, dia): kg que salen en camión}
    # tablas: {region: resultado de truck_cost_table hacia la sede regional de esa región}
    # Devuelve {(region, dia): {"costo", "capacidad", "espacio_libre", "camiones"}}; espacio_libre lo usa fill_trucks
    compras = {}
    for (region, dia), kg in carga.items():
        if kg <= 0:
            continue   # ese día no sale camión hacia esa región

        tabla = tablas.get(region)
        if tabla is None:
            raise ValueError(f"No hay camión hacia {region}: el plan no puede consolidar pedidos de esa región")
        costo, capacidad, camion_metido = tabla

        # La tabla va de kilo en kilo. La tolerancia evita que el error de redondeo al sumar fracciones
        # (8.500,0000000001 kg) lleve la carga al kilo siguiente y obligue a comprar un camión más grande
        w = math.ceil(kg - 1e-6)
        if w >= len(costo):
            raise ValueError(f"La carga hacia {region} el {dia} ({kg:,.0f} kg) supera el peso máximo de la tabla "
                             f"({len(costo) - 1:,} kg)")

        compras[region, dia] = {
            "costo": float(costo[w]),
            "capacidad": float(capacidad[w]),
            "espacio_libre": float(capacidad[w] - kg),
            "camiones": truck_mix(camion_metido, capacidades, w),
        }
    return compras

def fill_trucks(espacio_libre: dict, candidatos: pd.DataFrame) -> dict:
    # Relleno voraz: llena el espacio libre de los camiones ya comprados con fracciones de traslados que iban
    # directo, empezando por el que más ahorra por kilo (mochila continua: ordenar por ahorro/kg es óptimo).
    # espacio_libre: {(region, dia): kg libres en los camiones de esa región y ese día}
    # candidatos: columnas id_pedido, dia_salida, region_logistica, peso_kg y ahorro
    #   (ahorro = flete directo − flete del último tramo; solo traslados que van directo en el plan)
    # Devuelve {(id_pedido, dia_salida): fracción del traslado que sube a ese camión}

    #1. Ahorro por kilo; los que no ahorran no se suben
    candidatos = candidatos.assign(ahorro_kg=candidatos["ahorro"] / candidatos["peso_kg"])
    candidatos = candidatos[candidatos["ahorro_kg"] > 0]

    #2. De mayor a menor ahorro por kilo
    candidatos = candidatos.sort_values("ahorro_kg", ascending=False, kind="stable")   # empates: en el orden en que llegan

    #3. Copia del espacio (no se modifica el que llega) y o que ya se subió de cada traslado
    espacio = dict(espacio_libre)
    subido = {}
    fracciones = {}

    #4. Llenar en ese orden
    for fila in candidatos.itertuples():
        camion = (fila.region_logistica, fila.dia_salida)
        libre = espacio.get(camion, 0)
        falta = 1-subido.get(fila.id_pedido, 0)
        if libre <=0 or falta <=0:
            continue
        fraccion = min(libre / fila.peso_kg, falta)
        fracciones[fila.id_pedido, fila.dia_salida] = fraccion
        espacio[camion] = libre - fraccion * fila.peso_kg
        subido[fila.id_pedido] = subido.get(fila.id_pedido, 0) + fraccion

    #5. fracciones de cada traslado en cada camion
    return fracciones


def plan_departures(plan: dict, instancia: dict, tolerancia: float = 1e-6) -> pd.DataFrame:
    # Convierte el plan en una tabla de salidas (id_pedido, dia_salida, fraccion) y revisa que sea válido.
    # plan: {id_pedido: dia_salida} para un pedido que sale completo, o {id_pedido: {dia_salida: fraccion, ...}}
    #   para un traslado partido (por ejemplo, el que el MILP reparte entre dos días)
    pedidos, opciones = instancia["pedidos"], instancia["opciones"]
    filas = []
    for id_pedido, salida in plan.items():
        partes = salida if isinstance(salida, dict) else {salida: 1.0}
        total = sum(partes.values())
        if any(fraccion <= 0 for fraccion in partes.values()) or total > 1 + tolerancia:
            raise ValueError(f"Las fracciones de {id_pedido} deben ser positivas y sumar a lo sumo 1: {partes}")
        if total < 1 - tolerancia and not pedidos.loc[id_pedido, "se_puede_partir"]:
            raise ValueError(f"{id_pedido} no es un traslado: debe salir completo ({partes})")
        filas += [(id_pedido, dia, fraccion) for dia, fraccion in partes.items()]

    salidas = pd.DataFrame(filas, columns=["id_pedido", "dia_salida", "fraccion"])
    llaves = list(zip(salidas["id_pedido"], salidas["dia_salida"]))
    invalidas = [llave for llave in llaves if llave not in opciones.index]
    if invalidas:
        raise ValueError(f"Estas salidas del plan no son opciones válidas: {invalidas[:5]}")
    return salidas.join(opciones[["region_logistica", "peso_kg", "flete_triangulacion", "castigo"]],
                        on=["id_pedido", "dia_salida"])


def departure_days(instancia: dict) -> dict:
    # Días en que cada pedido puede subir a un camión: {id_pedido: [dia_salida, ...]}
    # Solo cuentan las opciones de regiones con camión (las que están en instancia["tablas"]);
    # un pedido sin ninguna de esas opciones no aparece: siempre va directo y no se mueve.

    # 1. Opciones con camión: las de regiones que tienen tabla de costos de camión
    opciones = instancia["opciones"]
    con_camion = opciones[opciones["region_logistica"].isin(list(instancia["tablas"]))]

    #2. Agrupar los dias por pedido
    dias = con_camion.reset_index().groupby("id_pedido")["dia_salida"].apply(list)

    #3. {id_pedido: [dias]}
    return dias.to_dict()


def evaluate_plan(plan: dict, instancia: dict, filtrar_retraso: bool = True) -> dict:
    # Costo (Z1) y días de retraso ponderados por pedido (Z2) de un plan, sin solver.
    # plan: {id_pedido: dia_salida} de los pedidos que salen completos en camión y, para un traslado partido,
    #   {id_pedido: {dia_salida: fraccion, ...}}; lo que no está en el plan va directo.
    # filtrar_retraso: si es True, solo se rellenan traslados cuyo castigo en camión no supera el de ir directo
    #   (el relleno nunca empeora Z2); si es False, se rellena con todo traslado que ahorre costo.
    # Devuelve {"z1", "z2", "costo_camiones", "compras", "relleno"}
    pedidos, opciones = instancia["pedidos"], instancia["opciones"]

    # 1. Salidas del plan (pedido, día, fracción), revisadas
    del_plan = plan_departures(plan, instancia)

    # 2. Carga de cada camión: peso × fracción de lo que sale hacia cada (región, día)
    carga = ((del_plan["peso_kg"] * del_plan["fraccion"])
             .groupby([del_plan["region_logistica"], del_plan["dia_salida"]]).sum().to_dict())

    # 3. Camiones que se compran y espacio que les sobra
    compras = buy_trucks(carga, instancia["tablas"], instancia["capacidades"])
    espacio_libre = {camion: compra["espacio_libre"] for camion, compra in compras.items()}

    # 4. Candidatos al relleno: traslados que van directo (no están en el plan), con lo que ahorran si suben
    candidatos = opciones.reset_index().join(pedidos[["flete_directo", "castigo_directo", "se_puede_partir"]], on="id_pedido")
    candidatos = candidatos[candidatos["se_puede_partir"] & ~candidatos["id_pedido"].isin(plan)]
    if filtrar_retraso:
        candidatos = candidatos[candidatos["castigo"] <= candidatos["castigo_directo"]]
    candidatos = candidatos.assign(ahorro=candidatos["flete_directo"] - candidatos["flete_triangulacion"])

    # 5. Relleno del espacio libre
    relleno = fill_trucks(espacio_libre, candidatos[["id_pedido", "dia_salida", "region_logistica", "peso_kg", "ahorro"]])

    # 6. Parte de cada pedido que va directo: 1 − lo que sale en camión (por el plan o por el relleno)
    fraccion_directa = pd.Series(1.0, index=pedidos.index)
    fraccion_directa = fraccion_directa.sub(del_plan.groupby("id_pedido")["fraccion"].sum(), fill_value=0)
    for (id_pedido, _), fraccion in relleno.items():
        fraccion_directa[id_pedido] -= fraccion

    # 7. Z1: directos + últimos tramos (plan y relleno) + camiones
    costo_camiones = sum(compra["costo"] for compra in compras.values())
    z1 = ((pedidos["flete_directo"] * fraccion_directa).sum()
          + (del_plan["flete_triangulacion"] * del_plan["fraccion"]).sum()
          + sum(opciones.loc[salida, "flete_triangulacion"] * fraccion for salida, fraccion in relleno.items())
          + costo_camiones)

    # 8. Z2: días de retraso ponderados por pedido
    z2 = ((pedidos["castigo_directo"] * fraccion_directa).sum()
          + (del_plan["castigo"] * del_plan["fraccion"]).sum()
          + sum(opciones.loc[salida, "castigo"] * fraccion for salida, fraccion in relleno.items())) / instancia["n_pedidos"]

    return {"z1": z1, "z2": z2, "costo_camiones": costo_camiones, "compras": compras, "relleno": relleno}


def plan_from_milp(x: dict, tolerancia: float = 1e-6) -> dict:
    # Convierte la solución del MILP en un plan para el evaluador (y para sembrar las metaheurísticas).
    # Un pedido que sale completo (x = 1, con tolerancia porque el solver puede devolver 0,9999999) queda como
    # {id_pedido: dia}; un traslado que el MILP partió (0 < x < 1) queda con sus fracciones {id_pedido: {dia: x}},
    # así el evaluador compra los mismos camiones que el MILP
    plan = {}
    for (id_pedido, dia), variable in x.items():
        valor = variable.varValue or 0
        if valor > 1 - tolerancia:
            plan[id_pedido] = dia
        elif valor > tolerancia:
            plan.setdefault(id_pedido, {})[dia] = valor
    return plan


def prepare_regions(instancia: dict, filtrar_retraso: bool = True) -> dict:
    # Datos para evaluar el plan región por región (la tabú recalcula solo la región que cambia un movimiento).
    # Cada región con camión es independiente: sus pedidos solo suben a sus camiones y el relleno solo usa sus traslados.
    # Devuelve {"regiones": {region: datos}, "region_de": {id_pedido: region}, "base_costo", "base_castigo", "n_pedidos"}
    #   datos: pedidos de la región; costo y castigo si todos van directo; por opción (peso, cuánto cambian costo y
    #          castigo si sale en camión); traslados candidatos al relleno en el mismo orden que fill_trucks; tabla de camiones
    #   base_costo y base_castigo: lo de los pedidos de regiones sin camión, que siempre van directo
    pedidos, opciones = instancia["pedidos"], instancia["opciones"]
    con_camion = pedidos["region_logistica"].isin(list(instancia["tablas"]))

    # 1. Por opción: cuánto cambian costo y castigo si sale en camión en vez de directo
    info = opciones.reset_index().join(pedidos[["flete_directo", "castigo_directo", "se_puede_partir"]], on="id_pedido")
    info["d_costo"] = info["flete_triangulacion"] - info["flete_directo"]
    info["d_castigo"] = info["castigo"] - info["castigo_directo"]

    # 2. Candidatos al relleno, en el orden de fill_trucks: traslados que ahorran, de mayor a menor ahorro por kilo
    relleno = info[info["se_puede_partir"] & (info["d_costo"] < 0)]
    if filtrar_retraso:
        relleno = relleno[relleno["castigo"] <= relleno["castigo_directo"]]
    relleno = (relleno.assign(ahorro_kg=-relleno["d_costo"] / relleno["peso_kg"])
               .sort_values("ahorro_kg", ascending=False, kind="stable"))

    # 3. Un registro por región con camión
    regiones = {}
    for region, (costo, capacidad, _) in instancia["tablas"].items():
        de_region = pedidos[pedidos["region_logistica"] == region]
        opc = info[info["region_logistica"] == region]
        rel = relleno[relleno["region_logistica"] == region]
        regiones[region] = {
            "pedidos": list(de_region.index),
            "base_costo": float(de_region["flete_directo"].sum()),
            "base_castigo": float(de_region["castigo_directo"].sum()),
            "opciones": {(i, d): (float(p), float(c), float(k)) for i, d, p, c, k
                         in zip(opc["id_pedido"], opc["dia_salida"], opc["peso_kg"], opc["d_costo"], opc["d_castigo"])},
            "relleno": list(zip(rel["id_pedido"], rel["dia_salida"], rel["peso_kg"].astype(float),
                                (-rel["d_costo"]).astype(float), rel["d_castigo"].astype(float))),
            "costo": costo, "capacidad": capacidad}

    sin_camion = pedidos[~con_camion]
    return {"regiones": regiones, "region_de": pedidos.loc[con_camion, "region_logistica"].to_dict(),
            "base_costo": float(sin_camion["flete_directo"].sum()),
            "base_castigo": float(sin_camion["castigo_directo"].sum()), "n_pedidos": instancia["n_pedidos"]}


def evaluate_region(plan: dict, datos: dict) -> tuple[float, float]:
    # Costo y castigo (sumados, sin dividir por pedidos) de UNA región según el plan: lo mismo que evaluate_plan
    # calcula para esos pedidos, sin pandas
    costo, castigo = datos["base_costo"], datos["base_castigo"]

    # 1. Lo que sale en camión según el plan: último tramo y castigo en vez de directo, y carga de cada día
    carga, en_plan = {}, set()
    for id_pedido in datos["pedidos"]:
        salida = plan.get(id_pedido)
        if salida is None:
            continue
        en_plan.add(id_pedido)
        for dia, fraccion in (salida.items() if isinstance(salida, dict) else [(salida, 1.0)]):
            peso, d_costo, d_castigo = datos["opciones"][id_pedido, dia]
            costo += fraccion * d_costo
            castigo += fraccion * d_castigo
            carga[dia] = carga.get(dia, 0.0) + fraccion * peso

    # 2. Camiones de cada día (misma tolerancia que buy_trucks) y el espacio que les sobra
    libre = {}
    for dia, kg in carga.items():
        if kg > 0:
            w = math.ceil(kg - 1e-6)
            costo += float(datos["costo"][w])
            libre[dia] = float(datos["capacidad"][w]) - kg

    # 3. Relleno voraz con traslados que van directo, igual que fill_trucks
    subido = {}
    for id_pedido, dia, peso, ahorro, d_castigo in datos["relleno"]:
        espacio = libre.get(dia, 0.0)
        falta = 1 - subido.get(id_pedido, 0.0)
        if id_pedido in en_plan or espacio <= 0 or falta <= 0:
            continue
        fraccion = min(espacio / peso, falta)
        costo -= fraccion * ahorro
        castigo += fraccion * d_castigo
        libre[dia] = espacio - fraccion * peso
        subido[id_pedido] = subido.get(id_pedido, 0.0) + fraccion
    return costo, castigo


def region_totals(por_region: dict, regiones: dict) -> dict:
    # Z1 y Z2 a partir del costo y el castigo de cada región, más lo que siempre va directo
    z1 = regiones["base_costo"] + sum(costo for costo, _ in por_region.values())
    z2 = (regiones["base_castigo"] + sum(castigo for _, castigo in por_region.values())) / regiones["n_pedidos"]
    return {"z1": z1, "z2": z2, "por_region": por_region}


def evaluate_by_region(plan: dict, regiones: dict) -> dict:
    # Z1 y Z2 del plan sumando región por región; guarda cada región para luego recalcular solo una
    return region_totals({region: evaluate_region(plan, datos) for region, datos in regiones["regiones"].items()},
                         regiones)


def reevaluate_region(plan: dict, region: str, actual: dict, regiones: dict) -> dict:
    # Z1 y Z2 de un vecino que solo cambió 'region' respecto al plan evaluado en 'actual'; 'actual' no se modifica
    por_region = dict(actual["por_region"])
    por_region[region] = evaluate_region(plan, regiones["regiones"][region])
    return region_totals(por_region, regiones)
