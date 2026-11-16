"""Búsqueda tabú: movimientos sobre el plan, castigo adaptativo y búsqueda."""
import random

import pytest

from despacho_multiobjetivo.metodos import frontera_pareto
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import (departure_days, evaluate_by_region,
                                                                    evaluate_plan, plan_from_milp, prepare_regions,
                                                                    reevaluate_region)
from despacho_multiobjetivo.metodos.metaheuristicas.tabu import (adjust_penalty_weight, apply_move,
                                                                is_tabu, move_region, penalized_cost, prepare_tabu,
                                                                random_move,
                                                                random_truck_move, sample_moves, tabu_search,
                                                                truck_slots)
from despacho_multiobjetivo.modelos import model_multiobjetivo
from tests.metodos.instancia_pequena import CAPACIDADES, LUN, MAR, MIE, NODOS, SEDE, TARIFAS
from tests.metodos.instancia_pequena import datos as _datos, instancia as _instancia


# ---------------------------------------------------------------- movimiento al azar

DIAS = {"C_0": [LUN, MAR], "C_1": [MAR], "T_0": [LUN, MIE]}


def _movimientos(plan, n=300, semilla=0):
    rng = random.Random(semilla)
    return [random_move(plan, DIAS, rng) for _ in range(n)]


def test_desde_todo_directo_solo_se_puede_subir():
    for tipo, pedido, dia in _movimientos({}):
        assert tipo == "subir" and dia in DIAS[pedido]


def test_un_pedido_en_camion_baja_o_cambia_a_otro_de_sus_dias():
    plan = {"C_0": LUN, "C_1": MAR, "T_0": MIE}
    for tipo, pedido, dia in _movimientos(plan):
        assert tipo in ("bajar", "cambiar")
        if tipo == "bajar":
            assert dia is None
        else:
            assert dia in DIAS[pedido] and dia != plan[pedido]   # nunca "cambia" al mismo día


def test_pedido_con_un_solo_dia_ya_en_camion_solo_puede_bajar():
    # C_1 solo puede salir el martes y ya está ahí
    assert {(tipo, dia) for tipo, pedido, dia in _movimientos({"C_1": MAR}) if pedido == "C_1"} == {("bajar", None)}


def test_traslado_partido_puede_quedar_completo_en_cualquiera_de_sus_dias():
    # Arranque desde el MILP: T_0 repartido entre lunes y miércoles
    destinos = {dia for tipo, pedido, dia in _movimientos({"T_0": {LUN: 0.3, MIE: 0.5}}) if pedido == "T_0"}
    assert destinos == {None, LUN, MIE}


def test_salen_todos_los_tipos_de_movimiento():
    tipos = {tipo for tipo, _, _ in _movimientos({"C_0": LUN})}
    assert tipos == {"subir", "bajar", "cambiar"}


def test_misma_semilla_mismos_movimientos():
    plan = {"C_0": LUN}
    assert _movimientos(plan, semilla=7) == _movimientos(plan, semilla=7)
    assert _movimientos(plan, semilla=7) != _movimientos(plan, semilla=8)


# ---------------------------------------------------------------- aplicar un movimiento

PLAN = {"C_0": LUN, "T_0": {LUN: 0.3, MIE: 0.5}}


def test_bajar_quita_el_pedido_del_plan():
    assert apply_move(PLAN, ("bajar", "C_0", None)) == {"T_0": {LUN: 0.3, MIE: 0.5}}


def test_subir_agrega_el_pedido_completo_ese_dia():
    assert apply_move(PLAN, ("subir", "C_1", MAR)) == {"C_0": LUN, "T_0": {LUN: 0.3, MIE: 0.5}, "C_1": MAR}


def test_cambiar_reemplaza_el_dia():
    assert apply_move(PLAN, ("cambiar", "C_0", MAR)) == {"C_0": MAR, "T_0": {LUN: 0.3, MIE: 0.5}}


def test_un_traslado_partido_que_se_mueve_queda_completo():
    assert apply_move(PLAN, ("cambiar", "T_0", LUN))["T_0"] == LUN


def test_el_plan_original_no_cambia():
    # La tabú saca muchos vecinos del mismo plan: ninguno puede modificarlo
    plan = {"C_0": LUN, "T_0": {LUN: 0.3, MIE: 0.5}}
    for movimiento in [("bajar", "C_0", None), ("subir", "C_1", MAR), ("cambiar", "T_0", LUN)]:
        apply_move(plan, movimiento)
    assert plan == {"C_0": LUN, "T_0": {LUN: 0.3, MIE: 0.5}}


def test_los_vecinos_siempre_son_planes_validos_para_el_evaluador():
    # Recorrido de 200 movimientos al azar: cada plan que sale lo acepta evaluate_plan (si no, lanza un error)
    instancia = _instancia()
    dias_por_pedido = departure_days(instancia)
    rng = random.Random(0)
    plan = {"T_0": {LUN: 0.3, MIE: 0.5}}   # arranca con un traslado partido, como desde el MILP
    for _ in range(200):
        plan = apply_move(plan, random_move(plan, dias_por_pedido, rng))
        evaluate_plan(plan, instancia)


# ---------------------------------------------------------------- castigo por pasarse del límite de Z2

def test_plan_que_cumple_no_paga_multa():
    assert penalized_cost(231e6, 1.005, 1.014, 227e6) == 231e6


def test_cumplir_con_holgura_no_da_premio():
    # Bajar Z2 por debajo de ε no descuenta costo: el límite es un tope, no algo que se busque bajar
    assert penalized_cost(231e6, 0.5, 1.014, 227e6) == penalized_cost(231e6, 1.014, 1.014, 227e6) == 231e6


def test_plan_que_se_pasa_paga_en_proporcion_al_exceso():
    # Se pasa 0,006 días equivalentes por pedido a 227 M por día: multa de 1,362 M
    assert penalized_cost(228e6, 1.020, 1.014, 227e6) == pytest.approx(229.362e6)


def test_con_la_multa_un_plan_que_se_pasa_poco_puede_ganarle_a_uno_que_cumple():
    # El caso que justifica el castigo: A se pasa un poco pero es más barato que B aun con la multa
    a = penalized_cost(228e6, 1.020, 1.014, 227e6)
    b = penalized_cost(231e6, 1.005, 1.014, 227e6)
    assert a < b


# ---------------------------------------------------------------- ajuste del peso del castigo

def test_si_se_pasa_el_peso_sube():
    assert adjust_penalty_weight(100.0, cumple=False, delta=0.1, alfa_min=1, alfa_max=1e6) == pytest.approx(110.0)


def test_si_cumple_el_peso_baja():
    assert adjust_penalty_weight(110.0, cumple=True, delta=0.1, alfa_min=1, alfa_max=1e6) == pytest.approx(100.0)


def test_una_subida_y_una_bajada_se_cancelan():
    alfa = adjust_penalty_weight(227e6, False, 0.1, 1, 1e12)
    assert adjust_penalty_weight(alfa, True, 0.1, 1, 1e12) == pytest.approx(227e6)


def test_el_peso_no_baja_del_minimo_ni_sube_del_maximo():
    assert adjust_penalty_weight(100.0, True, 0.1, alfa_min=95, alfa_max=1e6) == 95
    assert adjust_penalty_weight(100.0, False, 0.1, alfa_min=1, alfa_max=105) == 105


def test_secuencia_del_ejemplo():
    # no, no, sí, sí, no → 249,7 · 274,7 · 249,7 · 227,0 · 249,7 (millones)
    alfa, historia = 227e6, []
    for cumple in [False, False, True, True, False]:
        alfa = adjust_penalty_weight(alfa, cumple, 0.1, 227e3, 227e9)
        historia.append(alfa / 1e6)
    assert historia == pytest.approx([249.7, 274.67, 249.7, 227.0, 249.7])


# ---------------------------------------------------------------- muestra de movimientos

def test_la_muestra_trae_movimientos_distintos():
    muestra = sample_moves({}, DIAS, random.Random(0), 3)
    assert len(muestra) == 3 and len(set(muestra)) == 3


def test_si_hay_menos_movimientos_posibles_devuelve_todos_sin_quedarse_en_un_ciclo():
    # Desde todo directo solo hay 5 movimientos: subir C_0 (lun, mar), C_1 (mar) y T_0 (lun, mié)
    muestra = sample_moves({}, DIAS, random.Random(0), 100)
    assert set(muestra) == {("subir", "C_0", LUN), ("subir", "C_0", MAR), ("subir", "C_1", MAR),
                            ("subir", "T_0", LUN), ("subir", "T_0", MIE)}


def test_misma_semilla_misma_muestra_en_el_mismo_orden():
    plan = {"C_0": LUN}
    assert sample_moves(plan, DIAS, random.Random(3), 4) == sample_moves(plan, DIAS, random.Random(3), 4)


# ---------------------------------------------------------------- lista tabú

TABU = {("C_0", MAR): 15, ("C_1", None): 12}   # C_0 salió del martes y C_1 de directo


def test_volver_al_dia_de_origen_es_tabu_mientras_dura_el_plazo():
    assert is_tabu(("cambiar", "C_0", MAR), TABU, 8)
    assert is_tabu(("subir", "C_0", MAR), TABU, 8)    # aunque haya bajado entre tanto


def test_ir_a_otro_dia_no_es_tabu():
    assert not is_tabu(("cambiar", "C_0", LUN), TABU, 8)
    assert not is_tabu(("bajar", "C_0", None), TABU, 8)


def test_volver_a_directo_es_tabu_si_salio_de_directo():
    assert is_tabu(("bajar", "C_1", None), TABU, 8)


def test_el_plazo_se_cumple():
    assert not is_tabu(("cambiar", "C_0", MAR), TABU, 15)


def test_pedido_sin_registro_no_es_tabu():
    assert not is_tabu(("subir", "T_0", LUN), TABU, 1)


# ---------------------------------------------------------------- búsqueda completa

def _milp(epsilon):
    # Óptimo exacto del punto ε en la instancia pequeña: la referencia contra la que se mide la tabú
    pedidos, opciones = _datos()
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, CAPACIDADES, TARIFAS, NODOS, SEDE)
    resultado = frontera_pareto.epsilon_point(problema, z1, z2, epsilon, mostrar=False)
    return resultado, plan_from_milp(x)


def _buscar(epsilon, **opciones):
    parametros = {"tamano_muestra": 10, "max_sin_mejora": 30, "mostrar": False} | opciones
    return tabu_search(_instancia(), epsilon, **parametros)


@pytest.mark.parametrize("factor", [1.0, 1.5])   # con el retraso de hoy y con un límite holgado
def test_llega_al_optimo_del_milp_en_la_instancia_pequena(factor):
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean() * factor
    milp, _ = _milp(epsilon)
    resultado = _buscar(epsilon)
    assert resultado["factible"] and resultado["z2"] <= epsilon + 1e-6
    assert resultado["z1"] == pytest.approx(milp["costo"])


def test_arrancando_en_el_optimo_no_lo_empeora():
    # El mejor solo cambia si aparece uno que cumple y es más barato: la búsqueda puede alejarse, el resultado no
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean()
    milp, plan = _milp(epsilon)
    resultado = _buscar(epsilon, plan_inicial=plan)
    assert resultado["z1"] == pytest.approx(milp["costo"])


def test_misma_semilla_mismo_recorrido():
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean()
    a, b = _buscar(epsilon, semilla=4), _buscar(epsilon, semilla=4)
    assert a["plan"] == b["plan"]
    assert a["historial"]["z1"].tolist() == b["historial"]["z1"].tolist()


def test_historial_una_fila_por_iteracion_y_el_mejor_nunca_empeora():
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean()
    resultado = _buscar(epsilon)
    historial = resultado["historial"]
    assert len(historial) == resultado["iteraciones"]
    assert list(historial.columns) == ["iteracion", "z1", "z2", "cumple", "alfa", "z1_mejor", "segundos"]
    mejores = historial["z1_mejor"].dropna()
    assert (mejores.diff().dropna() <= 0).all()
    assert resultado["evaluaciones"] > resultado["iteraciones"]


def test_para_por_iteraciones_sin_mejorar():
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean()
    resultado = _buscar(epsilon, max_sin_mejora=5)
    # Las últimas 5 iteraciones no mejoraron el mejor
    assert resultado["historial"]["z1_mejor"].iloc[-6:].nunique() == 1


def test_para_por_tiempo():
    resultado = _buscar(_instancia()["pedidos"]["castigo_directo"].mean(), max_segundos=0)
    assert resultado["iteraciones"] == 1


def test_limite_imposible_avisa_y_no_devuelve_plan(capsys):
    # El menor Z2 posible en la instancia es ≈ 0,70 (cada pedido en su opción de menor castigo): con ε = 0,1
    # ningún plan cumple
    resultado = _buscar(0.1, max_sin_mejora=5)
    assert not resultado["factible"] and resultado["plan"] is None and resultado["z1"] is None
    assert "ADVERTENCIA" in capsys.readouterr().out


@pytest.mark.parametrize("epsilon", [0.0, -1.0])
def test_epsilon_no_positivo_es_un_error(epsilon):
    with pytest.raises(ValueError, match="epsilon debe ser positivo"):
        _buscar(epsilon)


# ---------------------------------------------------------------- movimientos de camión (abrir, llenar, recortar, cerrar)
# Un movimiento de camión cambia de una vez varios pedidos de una región en un día: (tipo, (region, dia), ids)
#   abrir:    camión vacío; suben los directos que más ahorran por kilo hasta la capacidad de un vehículo al azar
#   llenar:   camión que ya sale; suben los directos que más ahorran por kilo y caben en su espacio libre
#   recortar: bajan los que menos ahorran por kilo hasta que la carga quepa en camiones más baratos
#   cerrar:   bajan todos los que salen ese día hacia esa región
# En la instancia pequeña: Turbo 4.500 kg por 2,0 M y Sencillo 8.500 kg por 2,2 M. Ahorro por kilo
# (flete directo − último tramo): C_1 550, C_0 467 y T_0 410.

ANT = "ANTIOQUIA_CHOCO"
CAMIONES = truck_slots(_instancia())


def _camion(plan, n=300, semilla=0, camiones=CAMIONES):
    rng = random.Random(semilla)
    return [random_truck_move(plan, camiones, rng) for _ in range(n)]


def test_camiones_posibles_con_sus_pedidos_de_mayor_a_menor_ahorro_por_kilo():
    # O_0 es de Oriente (sin sede regional): no tiene camión
    assert set(CAMIONES) == {(ANT, LUN), (ANT, MAR), (ANT, MIE)}
    assert CAMIONES[ANT, MAR]["pedidos"] == [("C_1", 1000, 550_000), ("C_0", 3000, 1_400_000)]
    assert [pedido for pedido, _, _ in CAMIONES[ANT, LUN]["pedidos"]] == ["C_0", "T_0"]
    assert CAMIONES[ANT, LUN]["capacidades"] == [4500, 8500]


def test_abrir_con_turbo_sube_lo_que_cabe_y_el_resto_sigue_directo():
    # Lunes: C_0 (3.000) y T_0 (2.000) suman 5.000 kg. Con Turbo solo cabe C_0; con Sencillo caben los dos
    lunes = {movimiento for movimiento in _camion({}) if movimiento and movimiento[1] == (ANT, LUN)}
    assert lunes == {("abrir", (ANT, LUN), ("C_0",)), ("abrir", (ANT, LUN), ("C_0", "T_0"))}


def test_abrir_no_sube_pedidos_que_no_ahorran():
    lunes = dict(CAMIONES[ANT, LUN], pedidos=[("C_0", 3000, 1_400_000), ("T_0", 2000, -10_000)])
    movimientos = set(_camion({}, camiones={(ANT, LUN): lunes}))
    assert movimientos == {("abrir", (ANT, LUN), ("C_0",))}


def test_llenar_sube_lo_que_cabe_en_el_espacio_libre():
    # Martes: C_1 (1.000 kg) va en una Turbo y le sobran 3.500 kg, donde cabe C_0 (3.000)
    assert ("llenar", (ANT, MAR), ("C_0",)) in set(_camion({"C_1": MAR}))


def test_si_no_cabe_nadie_no_se_llena():
    # Lunes: T_0 (2.000 kg) en una Turbo deja 2.500 libres y C_0 pesa 3.000; tampoco hay camión más barato
    lunes = {movimiento for movimiento in _camion({"T_0": LUN}) if movimiento and movimiento[1] == (ANT, LUN)}
    assert lunes == {("cerrar", (ANT, LUN), ("T_0",))}


def test_recortar_baja_lo_que_menos_ahorra_para_pasar_de_sencillo_a_turbo():
    # Lunes con C_0 y T_0: 5.000 kg necesitan un Sencillo; bajando T_0 quedan 3.000 kg, que caben en una Turbo
    lunes = {movimiento for movimiento in _camion({"C_0": LUN, "T_0": LUN}) if movimiento and movimiento[1] == (ANT, LUN)}
    assert lunes == {("recortar", (ANT, LUN), ("T_0",)), ("cerrar", (ANT, LUN), ("C_0", "T_0"))}


def test_un_camion_en_uso_sin_espacio_util_solo_se_cierra():
    movimientos = set(_camion({"C_0": LUN, "C_1": MAR}))
    assert movimientos == {("cerrar", (ANT, LUN), ("C_0",)),      # T_0 (2.000) no cabe en los 1.500 libres
                           ("cerrar", (ANT, MAR), ("C_1",)),      # C_0 ya sale el lunes: no se toca
                           ("abrir", (ANT, MIE), ("T_0",))}


def test_el_traslado_partido_cuenta_su_fraccion_en_la_carga():
    # T_0 con 30 % el lunes son 600 kg: en esa Turbo quedan 3.900 libres y C_0 (3.000) cabe
    movimientos = set(_camion({"T_0": {LUN: 0.3, MIE: 0.5}}))
    assert ("llenar", (ANT, LUN), ("C_0",)) in movimientos
    assert ("cerrar", (ANT, LUN), ("T_0",)) in movimientos


def test_sin_nada_que_mover_no_hay_movimiento():
    # En el camión del martes solo está C_0, que ya sale el lunes: no hay nadie que subir ni que bajar
    martes = dict(CAMIONES[ANT, MAR], pedidos=[("C_0", 3000, 1_400_000)])
    assert _camion({"C_0": LUN}, n=5, camiones={(ANT, MAR): martes}) == [None] * 5


def test_abrir_y_llenar_suben_completos_ese_dia_sin_tocar_a_los_demas():
    plan = {"C_1": MAR}
    assert apply_move(plan, ("abrir", (ANT, LUN), ("C_0", "T_0"))) == {"C_1": MAR, "C_0": LUN, "T_0": LUN}
    assert apply_move(plan, ("llenar", (ANT, MAR), ("C_0",))) == {"C_1": MAR, "C_0": MAR}
    assert plan == {"C_1": MAR}


def test_recortar_y_cerrar_mandan_a_directo_y_al_partido_solo_le_quitan_ese_dia():
    plan = {"C_0": LUN, "C_1": MAR, "T_0": {LUN: 0.3, MIE: 0.5}}
    assert apply_move(plan, ("cerrar", (ANT, LUN), ("C_0", "T_0"))) == {"C_1": MAR, "T_0": {MIE: 0.5}}
    assert apply_move(plan, ("recortar", (ANT, LUN), ("T_0",))) == {"C_0": LUN, "C_1": MAR, "T_0": {MIE: 0.5}}
    assert apply_move({"T_0": {LUN: 0.3}}, ("cerrar", (ANT, LUN), ("T_0",))) == {}
    assert plan == {"C_0": LUN, "C_1": MAR, "T_0": {LUN: 0.3, MIE: 0.5}}


def test_un_camion_movido_hace_poco_es_tabu_para_los_cuatro_movimientos():
    tabu = {("camion", ANT, LUN): 10}
    for tipo in ("abrir", "llenar", "recortar", "cerrar"):
        assert is_tabu((tipo, (ANT, LUN), ("T_0",)), tabu, 8)
        assert not is_tabu((tipo, (ANT, LUN), ("T_0",)), tabu, 10)
    assert not is_tabu(("abrir", (ANT, MIE), ("T_0",)), tabu, 8)


def test_sin_probabilidad_de_camion_la_muestra_es_la_de_antes():
    # Agregar los movimientos de camión no cambia las corridas viejas: con prob_camion = 0 sale la misma muestra
    plan = {"C_0": LUN}
    antes = sample_moves(plan, DIAS, random.Random(3), 4)
    assert sample_moves(plan, DIAS, random.Random(3), 4, camiones=CAMIONES, prob_camion=0) == antes


def test_con_probabilidad_uno_solo_salen_movimientos_de_camion():
    muestra = sample_moves({"C_0": LUN}, DIAS, random.Random(0), 3, camiones=CAMIONES, prob_camion=1)
    assert muestra and all(tipo in ("abrir", "llenar", "recortar", "cerrar") for tipo, _, _ in muestra)


def test_recorrido_mezclado_siempre_da_planes_validos_para_el_evaluador():
    instancia = _instancia()
    dias_por_pedido, camiones = departure_days(instancia), truck_slots(instancia)
    rng = random.Random(0)
    plan = {"T_0": {LUN: 0.3, MIE: 0.5}}
    for _ in range(200):
        muestra = sample_moves(plan, dias_por_pedido, rng, 1, camiones=camiones, prob_camion=0.5)
        if muestra:
            plan = apply_move(plan, muestra[0])
        evaluate_plan(plan, instancia)


def test_con_movimientos_de_camion_llega_al_optimo_del_milp():
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean()
    milp, _ = _milp(epsilon)
    resultado = _buscar(epsilon, prob_camion=0.3)
    assert resultado["factible"] and resultado["z1"] == pytest.approx(milp["costo"])


def test_sin_probabilidad_de_camion_la_busqueda_es_la_de_antes():
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean()
    antes, ahora = _buscar(epsilon, semilla=2), _buscar(epsilon, semilla=2, prob_camion=0)
    assert antes["historial"]["z1"].tolist() == ahora["historial"]["z1"].tolist()


# ---------------------------------------------------------------- evaluación por región dentro de la tabú

def test_region_que_cambia_con_cada_movimiento():
    regiones = prepare_regions(_instancia())
    assert move_region(("subir", "C_0", LUN), regiones) == ANT
    assert move_region(("bajar", "T_0", None), regiones) == ANT
    assert move_region(("abrir", (ANT, LUN), ("C_0",)), regiones) == ANT


def test_recalcular_por_region_en_un_recorrido_da_lo_mismo_que_evaluate_plan():
    # 200 movimientos mezclados (pedidos y camiones): en cada paso, recalcular solo la región que cambió
    # debe dar el mismo Z1 y Z2 que evaluar el plan completo
    instancia = _instancia()
    regiones = prepare_regions(instancia)
    dias_por_pedido, camiones = departure_days(instancia), truck_slots(instancia)
    rng = random.Random(1)
    plan = {"T_0": {LUN: 0.3, MIE: 0.5}}
    actual = evaluate_by_region(plan, regiones)
    for _ in range(200):
        muestra = sample_moves(plan, dias_por_pedido, rng, 1, camiones=camiones, prob_camion=0.5)
        if not muestra:
            continue
        vecino = apply_move(plan, muestra[0])
        actual = reevaluate_region(vecino, move_region(muestra[0], regiones), actual, regiones)
        completo = evaluate_plan(vecino, instancia)
        assert (actual["z1"], actual["z2"]) == pytest.approx((completo["z1"], completo["z2"]), rel=1e-12)
        plan = vecino


def test_lo_que_reporta_la_tabu_coincide_con_evaluate_plan():
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean()
    resultado = _buscar(epsilon, prob_camion=0.3)
    completo = evaluate_plan(resultado["plan"], _instancia())
    assert (resultado["z1"], resultado["z2"]) == pytest.approx((completo["z1"], completo["z2"]), rel=1e-12)


def test_un_plan_inicial_invalido_es_un_error():
    with pytest.raises(ValueError, match="no son opciones válidas"):
        _buscar(1.0, plan_inicial={"C_1": LUN})   # C_1 solo puede salir el martes


# ---------------------------------------------------------------- lo que la tabú prepara una sola vez (lo reutiliza el memético)

def test_con_lo_preparado_la_busqueda_es_la_misma():
    instancia = _instancia()
    epsilon = instancia["pedidos"]["castigo_directo"].mean()
    sin, con = _buscar(epsilon, semilla=2, prob_camion=0.3), _buscar(epsilon, semilla=2, prob_camion=0.3,
                                                                       preparado=prepare_tabu(instancia))
    assert sin["historial"]["z1"].tolist() == con["historial"]["z1"].tolist()


def test_lo_preparado_con_otro_filtro_de_retraso_es_un_error():
    # El relleno cambia con filtrar_retraso: usar lo preparado con el otro valor daría costos equivocados
    preparado = prepare_tabu(_instancia(), filtrar_retraso=False)
    with pytest.raises(ValueError, match="filtrar_retraso"):
        _buscar(1.0, preparado=preparado, filtrar_retraso=True)
