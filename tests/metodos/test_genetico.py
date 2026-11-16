"""Algoritmo genético NSGA-II: cromosoma, problema, operadores y corrida."""
import itertools
import random

import numpy as np
import pytest
from pymoo.core.problem import Problem

from despacho_multiobjetivo.metodos import frontera_pareto
from despacho_multiobjetivo.metodos.metaheuristicas import genetico
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import evaluate_plan, plan_from_milp
from despacho_multiobjetivo.metodos.metaheuristicas.genetico import (ConsolidationProblem, ReassignMutation,
                                                                    RegionCrossover, SeededSampling,
                                                                    TruckRepair, best_for_epsilon,
                                                                    chromosome_layout, decode, encode, run_nsga2)
from despacho_multiobjetivo.modelos import model_multiobjetivo
from tests.metodos.instancia_pequena import CAPACIDADES, LUN, MAR, MIE, NODOS, SEDE, TARIFAS
from tests.metodos.instancia_pequena import datos as _datos
from tests.metodos.instancia_pequena import instancia as _instancia


# ---------------------------------------------------------------- qué significa cada posición del cromosoma

def test_una_posicion_por_pedido_que_se_puede_mover_en_orden_fijo():
    # O_0 es de Oriente (sin camión): no tiene gen. Los pedidos quedan en orden alfabético
    layout = chromosome_layout(_instancia())
    assert layout["pedidos"] == ["C_0", "C_1", "T_0"]


def test_los_dias_de_cada_pedido_van_en_orden_cronologico():
    layout = chromosome_layout(_instancia())
    assert layout["dias"] == [[LUN, MAR], [MAR], [LUN, MIE]]


def test_region_de_cada_posicion():
    layout = chromosome_layout(_instancia())
    assert layout["regiones"] == ["ANTIOQUIA_CHOCO"] * 3


def test_el_mismo_layout_en_cada_llamada():
    assert chromosome_layout(_instancia()) == chromosome_layout(_instancia())


# ---------------------------------------------------------------- cromosoma → plan

def test_todo_en_cero_es_todo_directo():
    assert decode([0, 0, 0], chromosome_layout(_instancia())) == {}


def test_cada_gen_distinto_de_cero_sale_su_k_esimo_dia():
    # Posiciones: C_0 (lun, mar), C_1 (mar), T_0 (lun, mié)
    layout = chromosome_layout(_instancia())
    assert decode([2, 1, 0], layout) == {"C_0": MAR, "C_1": MAR}
    assert decode([1, 0, 2], layout) == {"C_0": LUN, "T_0": MIE}
    assert decode([2, 1, 2], layout) == {"C_0": MAR, "C_1": MAR, "T_0": MIE}


def test_genes_decimales_de_pymoo():
    # pymoo puede entregar 2.0 o 0.9999999 en vez de 2 o 1
    assert decode([2.0, 0.9999999, 0.0], chromosome_layout(_instancia())) == {"C_0": MAR, "C_1": MAR}


def test_el_plan_decodificado_lo_acepta_el_evaluador():
    instancia = _instancia()
    plan = decode([2, 1, 2], chromosome_layout(instancia))
    evaluate_plan(plan, instancia)   # si el plan no fuera válido, lanzaría un error


# ---------------------------------------------------------------- plan → cromosoma

def test_todo_directo_es_todo_en_cero():
    assert encode({}, chromosome_layout(_instancia())).tolist() == [0, 0, 0]


def test_cada_pedido_del_plan_queda_con_la_posicion_de_su_dia():
    # Posiciones: C_0 (lun, mar), C_1 (mar), T_0 (lun, mié)
    assert encode({"C_0": MAR, "T_0": MIE}, chromosome_layout(_instancia())).tolist() == [2, 0, 2]


def test_un_traslado_partido_queda_en_cero_para_que_lo_rellene_el_evaluador():
    # El cromosoma no guarda fracciones. Ponerlo completo en un día compra más camión; en 0, fill_trucks lo
    # acomoda en el espacio libre (con datos reales, la semilla del 115 % pasa de +8,1 % a +2,7 % de costo)
    plan = {"C_0": LUN, "T_0": {LUN: 0.3, MIE: 0.6}}
    assert encode(plan, chromosome_layout(_instancia())).tolist() == [1, 0, 0]


def test_traslado_partido_de_un_pedido_sin_gen_tambien_es_un_error():
    with pytest.raises(ValueError, match="O_0 no está en el cromosoma"):
        encode({"O_0": {LUN: 0.5}}, chromosome_layout(_instancia()))


def test_pedido_que_no_puede_salir_en_camion_es_un_error():
    # O_0 es de Oriente: no tiene posición en el cromosoma
    with pytest.raises(ValueError, match="O_0 no está en el cromosoma"):
        encode({"O_0": LUN}, chromosome_layout(_instancia()))


def test_codificar_y_decodificar_devuelve_lo_mismo():
    # Ida y vuelta con cromosomas al azar: decode(x) → plan → encode(plan) = x
    layout = chromosome_layout(_instancia())
    rng = random.Random(0)
    for _ in range(50):
        x = [rng.randint(0, len(dias)) for dias in layout["dias"]]
        assert encode(decode(x, layout), layout).tolist() == x


# ---------------------------------------------------------------- el problema para pymoo

def _problema(filtrar_retraso=True):
    instancia = _instancia()
    return instancia, ConsolidationProblem(instancia, chromosome_layout(instancia), filtrar_retraso)


def test_una_variable_por_gen_y_dos_objetivos():
    _, problema = _problema()
    assert problema.n_var == 3 and problema.n_obj == 2


def test_rango_de_cada_gen_de_cero_al_numero_de_dias_del_pedido():
    # C_0 tiene 2 días posibles, C_1 uno y T_0 dos
    _, problema = _problema()
    assert problema.xl.tolist() == [0, 0, 0]
    assert problema.xu.tolist() == [2, 1, 2]


def test_las_variables_son_enteras():
    _, problema = _problema()
    assert problema.vtype is int


@pytest.mark.parametrize("x", [[0, 0, 0], [2, 1, 2], [1, 0, 2]])
def test_los_objetivos_son_z1_y_z2_del_evaluador(x):
    instancia, problema = _problema()
    esperado = evaluate_plan(decode(x, chromosome_layout(instancia)), instancia)
    F = problema.evaluate(np.array(x), return_values_of=["F"])
    assert F.tolist() == pytest.approx([esperado["z1"], esperado["z2"]])


def test_evalua_una_poblacion_completa():
    # pymoo evalúa muchos individuos a la vez: una fila por individuo, una columna por objetivo
    _, problema = _problema()
    F = problema.evaluate(np.array([[0, 0, 0], [2, 1, 2]]), return_values_of=["F"])
    assert F.shape == (2, 2)


def test_respeta_el_filtro_de_retraso_del_relleno():
    instancia, problema = _problema(filtrar_retraso=False)
    plan = decode([2, 1, 0], chromosome_layout(instancia))
    esperado = evaluate_plan(plan, instancia, filtrar_retraso=False)
    assert problema.evaluate(np.array([2, 1, 0]), return_values_of=["F"]).tolist() == pytest.approx(
        [esperado["z1"], esperado["z2"]])


# ---------------------------------------------------------------- población inicial

def _muestra(semillas, n, semilla=0, problema=None):
    problema = problema or _problema()[1]
    return SeededSampling(semillas).do(problema, n, random_state=np.random.default_rng(semilla)).get("X")


def test_una_fila_por_individuo_y_una_columna_por_gen():
    assert _muestra([], 10).shape == (10, 3)


def test_las_semillas_van_primero_tal_cual():
    X = _muestra([np.array([0, 0, 0]), np.array([2, 1, 2])], 10)
    assert X[0].tolist() == [0, 0, 0] and X[1].tolist() == [2, 1, 2]


def test_si_hay_mas_semillas_que_individuos_se_toman_las_primeras():
    X = _muestra([np.array([0, 0, 0]), np.array([2, 1, 2]), np.array([1, 1, 1])], 2)
    assert X.tolist() == [[0, 0, 0], [2, 1, 2]]


def test_los_genes_aleatorios_son_enteros_dentro_de_su_rango():
    X = _muestra([], 200)
    assert np.issubdtype(X.dtype, np.integer)
    assert (X >= 0).all() and (X <= np.array([2, 1, 2])).all()


def test_los_individuos_aleatorios_tienen_densidades_distintas():
    # Cada individuo aleatorio sube a camión una proporción distinta de pedidos: unos casi todo directo,
    # otros casi todo en camión. Con 200 genes y 50 individuos se ven los dos extremos
    grande = Problem(n_var=200, n_obj=2, xl=0, xu=4, vtype=int)
    en_camion = (_muestra([], 50, problema=grande) > 0).mean(axis=1)
    assert en_camion.min() < 0.2 and en_camion.max() > 0.8


def test_misma_semilla_misma_poblacion():
    assert (_muestra([], 20, semilla=5) == _muestra([], 20, semilla=5)).all()


# ---------------------------------------------------------------- cruce por región

# 6 genes de 3 regiones; el padre A tiene todos sus genes en 1 y el padre B en 2, para ver de quién viene cada gen
REGIONES = ["ANT", "ANT", "COSTA", "COSTA", "COSTA", "EJE"]
PADRE_A, PADRE_B = [1] * 6, [2] * 6


def _cruzar(padres_a, padres_b, semilla=0):
    # pymoo entrega los padres como una matriz (2 padres × n_cruces × n_genes) y espera los hijos igual
    X = np.array([padres_a, padres_b])
    return RegionCrossover(REGIONES)._do(None, X, random_state=np.random.default_rng(semilla))


def test_dos_hijos_por_cada_pareja_de_padres():
    hijos = _cruzar([PADRE_A] * 4, [PADRE_B] * 4)
    assert hijos.shape == (2, 4, 6)


def test_cada_region_del_hijo_viene_completa_de_un_solo_padre():
    # Lo que evita "partir medio camión": dentro de una región, todos los genes vienen del mismo padre
    hijos = _cruzar([PADRE_A] * 50, [PADRE_B] * 50)
    for hijo in hijos.reshape(-1, 6):
        for region in set(REGIONES):
            genes = {hijo[i] for i, r in enumerate(REGIONES) if r == region}
            assert len(genes) == 1


def test_el_segundo_hijo_recibe_lo_que_no_recibio_el_primero():
    hijos = _cruzar([PADRE_A] * 50, [PADRE_B] * 50)
    assert (hijos[0] + hijos[1] == 3).all()   # donde uno tiene 1 (del padre A), el otro tiene 2 (del padre B)


def test_los_hijos_mezclan_regiones_de_los_dos_padres():
    # En 50 cruces no todos los hijos salen iguales a un padre: hay regiones de A y de B en el mismo hijo
    primeros = _cruzar([PADRE_A] * 50, [PADRE_B] * 50)[0]
    mezclados = [hijo for hijo in primeros if 1 in hijo and 2 in hijo]
    assert len(mezclados) > 0


def test_padres_iguales_dan_hijos_iguales_a_ellos():
    padre = [0, 2, 1, 0, 3, 1]
    hijos = _cruzar([padre] * 3, [padre] * 3)
    assert (hijos == np.array(padre)).all()


def test_el_cruce_con_regiones_reales_del_layout():
    # Con la instancia pequeña las 3 posiciones son de Antioquia: cada hijo es una copia completa de un padre
    layout = chromosome_layout(_instancia())
    X = np.array([[[2, 1, 0]] * 10, [[0, 0, 2]] * 10])
    hijos = RegionCrossover(layout["regiones"])._do(None, X, random_state=np.random.default_rng(1))
    for hijo in hijos.reshape(-1, 3):
        assert hijo.tolist() in ([2, 1, 0], [0, 0, 2])


def test_misma_semilla_mismos_hijos():
    assert (_cruzar([PADRE_A] * 10, [PADRE_B] * 10, semilla=3) == _cruzar([PADRE_A] * 10, [PADRE_B] * 10, semilla=3)).all()


# ---------------------------------------------------------------- mutación

# 200 genes con 4 días posibles cada uno (máximo 4), como los pedidos reales con E = 3
GRANDE = Problem(n_var=200, n_obj=2, xl=0, xu=4, vtype=int)


def _mutar(X, prob_var=None, semilla=0, problema=GRANDE):
    return ReassignMutation(prob_var)._do(problema, np.array(X), random_state=np.random.default_rng(semilla))


def test_la_mutacion_no_cambia_la_forma_ni_modifica_la_entrada():
    X = np.full((10, 200), 2)
    mutados = _mutar(X)
    assert mutados.shape == (10, 200)
    assert (X == 2).all()                       # la población que entra queda intacta


def test_sin_probabilidad_no_muta_nada():
    X = np.full((10, 200), 2)
    assert (_mutar(X, prob_var=0.0) == X).all()


def test_con_probabilidad_uno_todos_los_genes_cambian_de_valor():
    # Un gen que muta siempre pasa a OTRO valor: mutar a lo mismo sería no mutar
    X = np.full((10, 200), 2)
    assert (_mutar(X, prob_var=1.0) != 2).all()


def test_los_genes_mutados_quedan_dentro_de_su_rango():
    rng = np.random.default_rng(1)
    X = rng.integers(0, 5, size=(50, 200))
    mutados = _mutar(X, prob_var=0.5)
    assert np.issubdtype(mutados.dtype, np.integer)
    assert (mutados >= 0).all() and (mutados <= 4).all()


def test_por_defecto_cambia_en_promedio_un_gen_por_individuo():
    # pymoo usa por defecto probabilidad 1 / n_var por gen: con 200 genes, ~1 cambio por individuo
    X = np.full((300, 200), 2)
    cambios = (_mutar(X) != X).sum(axis=1)
    assert 0.6 < cambios.mean() < 1.4


def test_un_gen_puede_mutar_a_cualquiera_de_los_otros_valores():
    # Partiendo de 2, con muchos cambios aparecen los otros cuatro valores posibles: 0 (directo), 1, 3 y 4
    X = np.full((50, 200), 2)
    assert set(np.unique(_mutar(X, prob_var=1.0))) == {0, 1, 3, 4}


def test_misma_semilla_misma_mutacion():
    X = np.full((10, 200), 2)
    assert (_mutar(X, semilla=4) == _mutar(X, semilla=4)).all()


# ---------------------------------------------------------------- reparación: cerrar camiones que no se pagan

# En la instancia pequeña un Turbo cuesta $2.100.000. Ahorro de subir cada pedido (flete directo − último tramo):
# C_0 1.400.000, C_1 550.000, T_0 820.000. Salir en camión mejora el servicio, salvo C_0 el martes (sale 1 día tarde)

def _reparar(filas):
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    problema = ConsolidationProblem(instancia, layout)
    return TruckRepair(instancia, layout)._do(problema, np.array(filas))


def test_cierra_el_camion_que_no_se_paga_si_no_empeora_el_servicio():
    # C_0 solo el martes: el Turbo ($2 M) no se paga con $1,4 M de ahorro, y directo C_0 llega antes → a directo
    assert _reparar([[2, 0, 0]]).tolist() == [[0, 0, 0]]


def test_no_cierra_el_camion_si_cerrarlo_empeora_el_servicio():
    # C_1 solo el martes: el Turbo no se paga, pero en camión C_1 tiene mejor servicio que directo: se deja
    assert _reparar([[0, 1, 0]]).tolist() == [[0, 1, 0]]


def test_no_cierra_el_camion_que_se_paga():
    # C_0 y T_0 el lunes ahorran $2,22 M: pagan el Turbo
    assert _reparar([[1, 0, 1]]).tolist() == [[1, 0, 1]]


def test_cada_camion_se_revisa_por_separado():
    # C_0 y C_1 el martes juntos no pagan el Turbo y, en conjunto, directo no empeora el servicio → se cierran;
    # T_0 el miércoles, aunque no se paga, tiene mejor servicio en camión → se deja
    assert _reparar([[2, 1, 2]]).tolist() == [[0, 0, 2]]


def test_la_reparada_nunca_es_peor_que_la_original():
    # Con los 18 cromosomas posibles: la reparada cuesta lo mismo o menos y tiene igual o menos retraso
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    todos = [list(x) for x in itertools.product(range(3), range(2), range(3))]
    for original, reparada in zip(todos, _reparar(todos)):
        antes = evaluate_plan(decode(original, layout), instancia)
        despues = evaluate_plan(decode(reparada, layout), instancia)
        assert despues["z1"] <= antes["z1"] + 1e-6 and despues["z2"] <= antes["z2"] + 1e-9


def test_la_reparacion_no_modifica_la_entrada():
    filas = np.array([[2, 0, 0]])
    _reparar(filas)
    assert filas.tolist() == [[2, 0, 0]]


# ---------------------------------------------------------------- corrida de NSGA-II

def _correr(semillas=(), semilla=0, generaciones=8):
    # Con número fijo de generaciones (no por tiempo) para que la corrida se pueda repetir
    return run_nsga2(_instancia(), semillas=semillas, pop_size=10, max_generaciones=generaciones, semilla=semilla)


def _milp(factor):
    pedidos, opciones = _datos()
    epsilon = _instancia()["pedidos"]["castigo_directo"].mean() * factor
    problema, x, u, y, z1, z2 = model_multiobjetivo.build_model(opciones, pedidos, CAPACIDADES, TARIFAS, NODOS, SEDE)
    return epsilon, frontera_pareto.epsilon_point(problema, z1, z2, epsilon, mostrar=False), plan_from_milp(x)


def test_el_frente_no_tiene_planes_dominados():
    frente = _correr()["frente"]
    for _, a in frente.iterrows():
        dominado = ((frente["z1"] <= a["z1"]) & (frente["z2"] <= a["z2"])
                    & ((frente["z1"] < a["z1"]) | (frente["z2"] < a["z2"]))).any()
        assert not dominado


def test_cada_plan_del_frente_coincide_con_su_evaluacion():
    instancia = _instancia()
    for _, fila in _correr()["frente"].iterrows():
        resultado = evaluate_plan(fila["plan"], instancia)
        assert (resultado["z1"], resultado["z2"]) == pytest.approx((fila["z1"], fila["z2"]))


@pytest.mark.parametrize("factor", [1.0, 1.5])   # con el retraso de hoy y con un límite holgado
def test_llega_al_optimo_del_milp_en_la_instancia_pequena(factor):
    epsilon, milp, _ = _milp(factor)
    mejor = best_for_epsilon(_correr(), epsilon)
    assert mejor["z2"] <= epsilon + 1e-6
    assert mejor["z1"] == pytest.approx(milp["costo"])


def test_sembrado_con_el_milp_no_queda_peor_que_la_semilla():
    epsilon, milp, plan = _milp(1.0)
    mejor = best_for_epsilon(_correr(semillas=[plan], generaciones=2), epsilon)
    assert mejor["z1"] <= milp["costo"] + 1e-6


def test_sin_plan_que_cumpla_el_limite_devuelve_none():
    # El menor Z2 posible en la instancia es ≈ 0,70: con ε = 0,1 ningún plan del frente cumple
    assert best_for_epsilon(_correr(), 0.1) is None


def test_historial_una_fila_por_generacion():
    resultado = _correr(generaciones=5)
    historial = resultado["historial"]
    assert len(historial) == resultado["generaciones"] == 5
    assert list(historial.columns) == ["generacion", "segundos", "tamano_frente", "z1_minimo", "z2_minimo"]


def test_misma_semilla_mismo_frente():
    a, b = _correr(semilla=3), _correr(semilla=3)
    assert a["frente"][["z1", "z2"]].values.tolist() == b["frente"][["z1", "z2"]].values.tolist()


# ---------------------------------------------------------------- operadores genéricos de pymoo (comparación)

def test_con_operadores_genericos_el_frente_tambien_es_valido():
    # Cruce SBX y mutación polinomial con redondeo (lo que trae pymoo para enteros), sin reparación
    instancia = _instancia()
    resultado = run_nsga2(instancia, pop_size=10, max_generaciones=5, semilla=0, operadores="genericos")
    for _, fila in resultado["frente"].iterrows():
        evaluado = evaluate_plan(fila["plan"], instancia)
        assert (evaluado["z1"], evaluado["z2"]) == pytest.approx((fila["z1"], fila["z2"]))


def test_operadores_desconocidos_es_un_error():
    with pytest.raises(ValueError, match="operadores"):
        run_nsga2(_instancia(), pop_size=10, max_generaciones=2, operadores="otros")


# ---------------------------------------------------------------- evaluación rápida (por región) dentro del problema

@pytest.mark.parametrize("filtrar", [True, False])
def test_todos_los_cromosomas_dan_lo_mismo_que_evaluate_plan(filtrar):
    # La evaluación por región es la misma que usa la tabú: debe dar el Z1 y el Z2 del evaluador completo
    instancia, problema = _problema(filtrar_retraso=filtrar)
    layout = chromosome_layout(instancia)
    for x in itertools.product(range(3), range(2), range(3)):
        esperado = evaluate_plan(decode(x, layout), instancia, filtrar_retraso=filtrar)
        F = problema.evaluate(np.array(x), return_values_of=["F"])
        assert F.tolist() == pytest.approx([esperado["z1"], esperado["z2"]], rel=1e-12)


def test_las_regiones_se_preparan_una_sola_vez_por_corrida(monkeypatch):
    preparar = genetico.prepare_regions
    veces = []
    monkeypatch.setattr(genetico, "prepare_regions", lambda *args, **kwargs: veces.append(1) or preparar(*args, **kwargs))
    run_nsga2(_instancia(), pop_size=10, max_generaciones=3)
    assert len(veces) == 1
