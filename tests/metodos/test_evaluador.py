"""Evaluador de las metaheurísticas: instancia, camiones que se compran y relleno de traslados."""
from datetime import date

import pandas as pd
import pulp
import pytest

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import (buy_trucks, departure_days, evaluate_by_region,
                                                             evaluate_plan, fill_trucks, plan_from_milp,
                                                             prepare_instance, prepare_regions,
                                                             reevaluate_region, truck_cost_table, truck_mix)
from despacho_multiobjetivo.modelos import model_multiobjetivo
from despacho_multiobjetivo.reglas_negocio import servicio
from tests.metodos.instancia_pequena import LUN, MAR, MIE
from tests.metodos.instancia_pequena import instancia as _instancia_pequena

SEDE = "0111"
CAPACIDADES = {"Turbo": 4500, "Sencillo": 8500, "Patineta": 16000, "Poderosa": 22000, "Mula": 34000}
TARIFAS = {  # tarifas inventadas
    ("Turbo", SEDE, "MEDELLIN"): 2_100_000,
    ("Sencillo", SEDE, "MEDELLIN"): 2_300_000,
    ("Patineta", SEDE, "MEDELLIN"): 3_800_000,
    ("Poderosa", SEDE, "MEDELLIN"): 4_500_000,
    ("Mula", SEDE, "MEDELLIN"): 5_000_000,
}


@pytest.fixture(scope="module")
def tabla():
    return truck_cost_table(CAPACIDADES, TARIFAS, SEDE, "MEDELLIN", 40_000)


@pytest.mark.parametrize("peso, costo, capacidad", [
    (0, 0, 0),                    # no sale camión
    (3_000, 2_100_000, 4_500),    # un Turbo
    (34_001, 7_100_000, 38_500),  # Mula + Turbo
    (40_000, 7_300_000, 42_500),  # Mula + Sencillo (Mula + 2 Turbos costaría 9.200.000)
])
def test_costo_y_capacidad_de_la_combinacion_mas_barata(tabla, peso, costo, capacidad):
    costos, capacidades, _ = tabla
    assert costos[peso] == costo
    assert capacidades[peso] == capacidad


def test_el_costo_nunca_baja_cuando_sube_el_peso(tabla):
    costos, _, _ = tabla
    assert all(costos[w] <= costos[w + 1] for w in range(len(costos) - 1))


@pytest.mark.parametrize("peso, camiones", [
    (0, {}),                              # no sale camión
    (3_000, {"Turbo": 1}),
    (6_000, {"Sencillo": 1}),             # un Sencillo ($2,2 M) es más barato que dos Turbos ($4,0 M)
    (34_001, {"Mula": 1, "Turbo": 1}),
    (40_000, {"Mula": 1, "Sencillo": 1}),
])
def test_composicion_de_la_combinacion_mas_barata(tabla, peso, camiones):
    _, _, camion_metido = tabla
    assert truck_mix(camion_metido, CAPACIDADES, peso) == camiones


def test_la_composicion_cuesta_y_carga_lo_que_dice_la_tabla(tabla):
    # Para cualquier peso, sumar tarifas y capacidades de los camiones de truck_mix da lo mismo que la tabla
    costos, capacidades, camion_metido = tabla
    for peso in range(0, 40_001, 997):
        camiones = truck_mix(camion_metido, CAPACIDADES, peso)
        assert sum(TARIFAS[v, SEDE, "MEDELLIN"] * n for v, n in camiones.items()) == costos[peso]
        assert sum(CAPACIDADES[v] * n for v, n in camiones.items()) == capacidades[peso]
        assert capacidades[peso] >= peso   # la carga siempre cabe


def test_hub_sin_tarifa_no_tiene_camion():
    assert truck_cost_table(CAPACIDADES, TARIFAS, SEDE, "BOGOTA", 1_000) is None


# ---------------------------------------------------------------- camiones que se compran para el plan

def test_compra_los_camiones_de_cada_region_y_dia(tabla):
    carga = {("ANTIOQUIA_CHOCO", "lun"): 37_000, ("ANTIOQUIA_CHOCO", "mar"): 3_000}
    compras = buy_trucks(carga, {"ANTIOQUIA_CHOCO": tabla}, CAPACIDADES)
    assert compras[("ANTIOQUIA_CHOCO", "lun")] == {"costo": 7_100_000, "capacidad": 38_500,
                                                   "espacio_libre": 1_500, "camiones": {"Mula": 1, "Turbo": 1}}
    assert compras[("ANTIOQUIA_CHOCO", "mar")]["camiones"] == {"Turbo": 1}


def test_el_error_de_redondeo_no_compra_un_camion_mas_grande(tabla):
    # 4.500 kg sumados con fracciones pueden quedar en 4.500,0000000001: sigue cabiendo en un Turbo
    compras = buy_trucks({("ANTIOQUIA_CHOCO", "lun"): 4_500 + 1e-10}, {"ANTIOQUIA_CHOCO": tabla}, CAPACIDADES)
    assert compras[("ANTIOQUIA_CHOCO", "lun")]["camiones"] == {"Turbo": 1}


def test_sin_carga_no_se_compra_camion(tabla):
    assert buy_trucks({("ANTIOQUIA_CHOCO", "lun"): 0}, {"ANTIOQUIA_CHOCO": tabla}, CAPACIDADES) == {}


def test_region_sin_camion_es_un_error(tabla):
    with pytest.raises(ValueError, match="No hay camión hacia CENTRO"):
        buy_trucks({("CENTRO", "lun"): 1_000}, {"ANTIOQUIA_CHOCO": tabla}, CAPACIDADES)


def test_carga_mayor_que_la_tabla_es_un_error(tabla):
    # La tabla de la prueba llega hasta 40.000 kg
    with pytest.raises(ValueError, match="supera el peso máximo"):
        buy_trucks({("ANTIOQUIA_CHOCO", "lun"): 45_000}, {"ANTIOQUIA_CHOCO": tabla}, CAPACIDADES)


# ---------------------------------------------------------------- relleno de camiones con traslados

LUNES, MARTES = "2026-01-05", "2026-01-06"


def _candidatos(filas):
    """filas: (id_pedido, dia_salida, region, peso_kg, ahorro)."""
    return pd.DataFrame(filas, columns=["id_pedido", "dia_salida", "region_logistica", "peso_kg", "ahorro"])


def test_llena_primero_el_que_mas_ahorra_por_kilo():
    # 1.500 kg libres: T1 ($300/kg) sube completo y T2 ($200/kg) sube la cuarta parte; T3 no ahorra
    espacio = {("ANT", LUNES): 1500}
    fracciones = fill_trucks(espacio, _candidatos([("T2", LUNES, "ANT", 2000, 400_000),
                                                   ("T1", LUNES, "ANT", 1000, 300_000),
                                                   ("T3", LUNES, "ANT", 500, -10_000)]))
    assert fracciones == pytest.approx({("T1", LUNES): 1.0, ("T2", LUNES): 0.25})


def test_no_modifica_el_espacio_que_recibe():
    espacio = {("ANT", LUNES): 1500}
    fill_trucks(espacio, _candidatos([("T1", LUNES, "ANT", 1000, 300_000)]))
    assert espacio == {("ANT", LUNES): 1500}


def test_un_traslado_repartido_en_dos_dias_no_pasa_de_uno():
    espacio = {("ANT", LUNES): 600, ("ANT", MARTES): 900}
    fracciones = fill_trucks(espacio, _candidatos([("T1", LUNES, "ANT", 1000, 300_000),
                                                   ("T1", MARTES, "ANT", 1000, 300_000)]))
    assert sum(fracciones.values()) == pytest.approx(1.0)
    assert fracciones[("T1", LUNES)] == pytest.approx(0.6)


def test_sin_camion_ese_dia_no_sube_nada():
    # No hay camión hacia Oriente el lunes: no se compra uno para rellenar
    fracciones = fill_trucks({("ANT", LUNES): 1500}, _candidatos([("T1", LUNES, "ORIENTE", 1000, 300_000)]))
    assert fracciones == {}


def test_sin_candidatos_devuelve_vacio():
    assert fill_trucks({("ANT", LUNES): 1500}, _candidatos([])) == {}


# ---------------------------------------------------------------- instancia precalculada

LUN, MAR = date(2026, 1, 5), date(2026, 1, 6)


NODOS = {"ANTIOQUIA_CHOCO": "MEDELLIN", "ORIENTE": None}


def _instancia(opciones=None):
    pedidos = pd.DataFrame({
        "id_pedido": ["C_0", "C_1", "T_0", "O_0"],
        "tipo": ["CLIENTES", "CLIENTES", "TRASLADOS", "CLIENTES"],
        "region_logistica": ["ANTIOQUIA_CHOCO", "ANTIOQUIA_CHOCO", "ANTIOQUIA_CHOCO", "ORIENTE"],
        "peso_kg": [3000, 1000, 2000, 500],
        "flete_sin_consolidar": [1_500_000, 600_000, 900_000, 300_000],
        "flete_triangulacion": [100_000, 50_000, 80_000, 40_000],
    }).assign(transportadora_directo="ENVIA", tipo_transportadora="PAQUETEO", sede=SEDE, llega_directo=False)
    if opciones is None:
        opciones = pd.DataFrame([("C_0", LUN, 0), ("C_0", MAR, 1), ("C_1", MAR, 0),
                                 ("T_0", LUN, 0), ("T_0", MAR, 0), ("O_0", LUN, 0)],
                                columns=["id_pedido", "dia_salida", "retraso"])
    return pedidos, opciones, prepare_instance(pedidos, opciones, CAPACIDADES, TARIFAS, NODOS, SEDE)


def test_instancia_trae_los_datos_de_cada_pedido():
    pedidos, _, instancia = _instancia()
    info = instancia["pedidos"]
    assert info.loc["C_0", "flete_directo"] == 1_500_000
    assert not info.loc["C_0", "se_puede_partir"] and info.loc["T_0", "se_puede_partir"]   # clientes enteros, traslados no
    assert info["castigo_directo"].tolist() == pytest.approx(servicio.expected_delay_direct(pedidos).tolist())
    assert instancia["n_pedidos"] == 4


def test_instancia_trae_el_castigo_de_cada_opcion():
    pedidos, opciones, instancia = _instancia()
    info = instancia["opciones"]
    assert info.loc[("C_0", MAR), "flete_triangulacion"] == 100_000
    assert info["castigo"].tolist() == pytest.approx(servicio.expected_delay_options(opciones, pedidos).tolist())


def test_tablas_solo_para_regiones_con_camion_y_hasta_la_mayor_carga_de_un_dia():
    _, _, instancia = _instancia()
    assert set(instancia["tablas"]) == {"ANTIOQUIA_CHOCO"}       # Oriente no tiene sede regional
    costo, _, _ = instancia["tablas"]["ANTIOQUIA_CHOCO"]
    # Antioquia: lunes C_0 + T_0 = 5.000 kg; martes C_0 + C_1 + T_0 = 6.000 kg → la tabla llega a 6.000
    assert len(costo) == 6_001


# ---------------------------------------------------------------- evaluación de un plan

def test_el_evaluador_da_lo_mismo_que_el_milp():
    # El MILP resuelve el caso y su solución, pasada por el evaluador, debe dar el mismo Z1 y el mismo Z2
    pedidos, opciones, instancia = _instancia()
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, CAPACIDADES, TARIFAS, NODOS, SEDE)
    problema.setObjective(z1)
    problema.solve(pulp.HiGHS(msg=0))
    resultado = evaluate_plan(plan_from_milp(x), instancia)
    assert resultado["z1"] == pytest.approx(pulp.value(z1))
    assert resultado["z2"] == pytest.approx(pulp.value(z2))


def test_todo_directo():
    pedidos, _, instancia = _instancia()
    resultado = evaluate_plan({}, instancia)
    assert resultado["z1"] == pytest.approx(pedidos["flete_sin_consolidar"].sum())
    assert resultado["z2"] == pytest.approx(instancia["pedidos"]["castigo_directo"].mean())
    assert resultado["compras"] == {} and resultado["relleno"] == {}


def test_plan_con_clientes_y_relleno_del_traslado():
    # C_0 y C_1 el martes: 4.000 kg → 1 Turbo ($2.000.000) con 500 kg libres → sube 1/4 de T_0 (2.000 kg)
    _, _, instancia = _instancia()
    resultado = evaluate_plan({"C_0": MAR, "C_1": MAR}, instancia)
    assert resultado["relleno"] == pytest.approx({("T_0", MAR): 0.25})
    assert resultado["compras"][("ANTIOQUIA_CHOCO", MAR)]["camiones"] == {"Turbo": 1}
    assert resultado["z1"] == pytest.approx(
        0.75 * 900_000 + 300_000                  # lo que va directo: 3/4 de T_0 y O_0
        + 100_000 + 50_000 + 0.25 * 80_000        # últimos tramos de C_0, C_1 y el pedazo de T_0
        + 2_100_000)                               # el Turbo


def test_un_traslado_en_el_plan_va_completo():
    # T_0 completo el lunes: 2.000 kg → 1 Turbo; no hay más traslados para rellenar
    _, _, instancia = _instancia()
    resultado = evaluate_plan({"T_0": LUN}, instancia)
    assert resultado["relleno"] == {}
    assert resultado["z1"] == pytest.approx(1_500_000 + 600_000 + 300_000 + 80_000 + 2_100_000)


def test_salida_que_no_es_opcion_es_un_error():
    _, _, instancia = _instancia()
    with pytest.raises(ValueError, match="no son opciones válidas"):
        evaluate_plan({"C_1": LUN}, instancia)   # C_1 solo puede salir el martes


def test_filtrar_retraso_no_rellena_si_el_camion_llega_mas_tarde():
    # Ahora T_0 sale el martes con 1 día de retraso planeado: en camión su castigo supera el de ir directo
    opciones = pd.DataFrame([("C_0", MAR, 0), ("C_1", MAR, 0), ("T_0", MAR, 1)],
                            columns=["id_pedido", "dia_salida", "retraso"])
    _, _, instancia = _instancia(opciones)
    con_filtro = evaluate_plan({"C_0": MAR, "C_1": MAR}, instancia, filtrar_retraso=True)
    sin_filtro = evaluate_plan({"C_0": MAR, "C_1": MAR}, instancia, filtrar_retraso=False)
    assert con_filtro["relleno"] == {}                                     # no empeora Z2
    assert sin_filtro["relleno"] == pytest.approx({("T_0", MAR): 0.25})   # ahorra costo
    assert sin_filtro["z1"] < con_filtro["z1"] and sin_filtro["z2"] > con_filtro["z2"]


# ---------------------------------------------------------------- plan a partir de la solución del MILP

def _variable(nombre, valor):
    variable = pulp.LpVariable(nombre)
    variable.varValue = valor
    return variable


def test_plan_desde_milp_completos_y_traslados_partidos():
    x = {("C_0", MAR): _variable("a", 1.0), ("C_0", LUN): _variable("b", 0.0),
         ("C_1", MAR): _variable("c", 0.9999999),   # el solver a veces no devuelve exactamente 1
         ("T_0", LUN): _variable("d", 0.30),         # traslado partido entre dos días: queda con sus fracciones
         ("T_0", MAR): _variable("f", 0.25),
         ("O_0", LUN): _variable("e", None)}         # variable sin valor: cuenta como 0
    assert plan_from_milp(x) == {"C_0": MAR, "C_1": MAR, "T_0": {LUN: 0.30, MAR: 0.25}}


def test_el_traslado_que_el_milp_parte_queda_en_el_plan_con_su_fraccion():
    # Con un solo Turbo disponible el MILP sube C_0 y C_1 el martes y llena los 500 kg libres con 1/4 de T_0
    pedidos, opciones, _ = _instancia()
    solo_turbo = {"Turbo": 4500}
    tarifas = {("Turbo", SEDE, "MEDELLIN"): 1_000_000}
    instancia = prepare_instance(pedidos, opciones, solo_turbo, tarifas, NODOS, SEDE)
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, solo_turbo, tarifas, NODOS, SEDE)
    problema.setObjective(z1)
    problema.solve(pulp.HiGHS(msg=0))

    plan = plan_from_milp(x)
    assert plan["T_0"] == pytest.approx({MAR: 0.25})
    resultado = evaluate_plan(plan, instancia)
    assert (resultado["z1"], resultado["z2"]) == pytest.approx((pulp.value(z1), pulp.value(z2)))


# ---------------------------------------------------------------- planes con traslados partidos

def test_los_camiones_se_compran_para_la_carga_completa_incluidas_las_fracciones():
    # 80 % de T_0 (1.600 kg) el lunes con C_0 (3.000 kg) = 4.600 kg: ya no cabe en un Turbo → Sencillo
    _, _, instancia = _instancia()
    resultado = evaluate_plan({"C_0": LUN, "T_0": {LUN: 0.8}}, instancia)
    assert resultado["compras"][("ANTIOQUIA_CHOCO", LUN)]["camiones"] == {"Sencillo": 1}
    assert resultado["z1"] == pytest.approx(
        600_000 + 300_000 + 0.2 * 900_000          # directo: C_1, O_0 y el 20 % de T_0
        + 100_000 + 0.8 * 80_000                    # últimos tramos: C_0 y el 80 % de T_0
        + 2_300_000)                                # el Sencillo


def test_un_traslado_puede_repartirse_entre_dias():
    _, _, instancia = _instancia()
    resultado = evaluate_plan({"T_0": {LUN: 0.5, MAR: 0.5}}, instancia)
    assert set(resultado["compras"]) == {("ANTIOQUIA_CHOCO", LUN), ("ANTIOQUIA_CHOCO", MAR)}   # 1.000 kg cada día


@pytest.mark.parametrize("plan, mensaje", [
    ({"C_0": {LUN: 0.5}}, "no es un traslado"),                     # un cliente no se parte
    ({"T_0": {LUN: 0.7, MAR: 0.6}}, "sumar a lo sumo 1"),           # más del 100 % del traslado
    ({"T_0": {LUN: 0}}, "positivas"),
])
def test_fracciones_invalidas_son_un_error(plan, mensaje):
    _, _, instancia = _instancia()
    with pytest.raises(ValueError, match=mensaje):
        evaluate_plan(plan, instancia)


# ---------------------------------------------------------------- días en que puede salir cada pedido

def test_dias_de_cada_pedido_con_camion():
    dias = departure_days(_instancia_pequena())
    assert {pedido: sorted(lista) for pedido, lista in dias.items()} == {
        "C_0": [LUN, MAR], "C_1": [MAR], "T_0": [LUN, MIE]}


def test_pedido_de_region_sin_camion_no_se_puede_mover():
    # O_0 tiene opción el lunes, pero hacia Oriente no hay camión: siempre va directo
    assert "O_0" not in departure_days(_instancia_pequena())


def test_los_dias_sirven_como_llave_de_las_opciones():
    # Los días deben quedar del mismo tipo que el índice de opciones: si no, el plan no coincide con ellas
    instancia = _instancia_pequena()
    for pedido, dias in departure_days(instancia).items():
        assert all((pedido, dia) in instancia["opciones"].index for dia in dias)


# ---------------------------------------------------------------- evaluación por región (para la tabú)
# Cada región con camión es independiente: sus pedidos solo suben a sus camiones y el relleno solo usa sus traslados.
# Por eso Z1 y Z2 se pueden sumar región por región, y un movimiento solo obliga a recalcular la región que toca.

PLANES = [{}, {"C_0": MAR, "C_1": MAR}, {"T_0": LUN}, {"C_0": LUN, "T_0": {MAR: 0.5}}, {"C_0": LUN, "C_1": MAR}]


def _instancia_dos_regiones():
    # Igual a la de arriba, pero Oriente también tiene camión (hacia Medellín): O_0 puede salir el lunes en camión
    pedidos, opciones, _ = _instancia()
    nodos = {"ANTIOQUIA_CHOCO": "MEDELLIN", "ORIENTE": "MEDELLIN"}
    return prepare_instance(pedidos, opciones, CAPACIDADES, TARIFAS, nodos, SEDE)


def test_regiones_precalculadas_y_lo_que_siempre_va_directo():
    _, _, instancia = _instancia()
    regiones = prepare_regions(instancia)
    assert set(regiones["regiones"]) == {"ANTIOQUIA_CHOCO"}
    assert set(regiones["regiones"]["ANTIOQUIA_CHOCO"]["pedidos"]) == {"C_0", "C_1", "T_0"}
    assert regiones["region_de"] == {"C_0": "ANTIOQUIA_CHOCO", "C_1": "ANTIOQUIA_CHOCO", "T_0": "ANTIOQUIA_CHOCO"}
    # O_0 es de Oriente, sin camión: siempre va directo y queda en la base
    assert regiones["base_costo"] == pytest.approx(300_000)
    assert regiones["base_castigo"] == pytest.approx(instancia["pedidos"].loc["O_0", "castigo_directo"])


@pytest.mark.parametrize("filtrar", [True, False])
@pytest.mark.parametrize("plan", PLANES)
def test_evaluar_por_region_da_lo_mismo_que_evaluate_plan(plan, filtrar):
    _, _, instancia = _instancia()
    por_region = evaluate_by_region(plan, prepare_regions(instancia, filtrar))
    completo = evaluate_plan(plan, instancia, filtrar)
    assert por_region["z1"] == pytest.approx(completo["z1"], rel=1e-12)
    assert por_region["z2"] == pytest.approx(completo["z2"], rel=1e-12)


@pytest.mark.parametrize("filtrar", [True, False])
def test_el_filtro_de_retraso_del_relleno_se_respeta(filtrar):
    # T_0 sale el martes con 1 día de retraso planeado: con filtro no rellena, sin filtro sí
    opciones = pd.DataFrame([("C_0", MAR, 0), ("C_1", MAR, 0), ("T_0", MAR, 1)],
                            columns=["id_pedido", "dia_salida", "retraso"])
    _, _, instancia = _instancia(opciones)
    plan = {"C_0": MAR, "C_1": MAR}
    por_region = evaluate_by_region(plan, prepare_regions(instancia, filtrar))
    completo = evaluate_plan(plan, instancia, filtrar)
    assert (por_region["z1"], por_region["z2"]) == pytest.approx((completo["z1"], completo["z2"]), rel=1e-12)


def test_con_dos_regiones_cambiar_una_no_toca_la_otra():
    instancia = _instancia_dos_regiones()
    regiones = prepare_regions(instancia)
    antes = evaluate_by_region({"C_0": LUN, "O_0": LUN}, regiones)
    despues = evaluate_by_region({"C_1": MAR, "O_0": LUN}, regiones)
    assert set(antes["por_region"]) == {"ANTIOQUIA_CHOCO", "ORIENTE"}
    assert despues["por_region"]["ORIENTE"] == antes["por_region"]["ORIENTE"]
    assert despues["por_region"]["ANTIOQUIA_CHOCO"] != antes["por_region"]["ANTIOQUIA_CHOCO"]


def test_recalcular_solo_la_region_que_cambio_da_lo_mismo_que_evaluar_todo():
    instancia = _instancia_dos_regiones()
    regiones = prepare_regions(instancia)
    actual = evaluate_by_region({"C_0": LUN, "O_0": LUN}, regiones)
    vecino = {"C_1": MAR, "T_0": MAR, "O_0": LUN}   # solo cambió Antioquia
    recalculado = reevaluate_region(vecino, "ANTIOQUIA_CHOCO", actual, regiones)
    completo = evaluate_plan(vecino, instancia)
    assert recalculado["z1"] == pytest.approx(completo["z1"], rel=1e-12)
    assert recalculado["z2"] == pytest.approx(completo["z2"], rel=1e-12)
    assert actual["por_region"]["ANTIOQUIA_CHOCO"] != recalculado["por_region"]["ANTIOQUIA_CHOCO"]   # no lo modifica


def test_con_empate_en_ahorro_por_kilo_llena_en_el_orden_en_que_llegan():
    # El orden del relleno no puede depender del ordenamiento: la evaluación por región lo repite igual
    espacio = {("ANT", LUNES): 1000}
    fracciones = fill_trucks(espacio, _candidatos([("T1", LUNES, "ANT", 1000, 300_000),
                                                   ("T2", LUNES, "ANT", 1000, 300_000)]))
    assert fracciones == pytest.approx({("T1", LUNES): 1.0})
