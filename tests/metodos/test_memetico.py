"""Versión memética del genético: mejora de individuos con una tabú corta."""
import itertools

import numpy as np
import pytest
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination

from despacho_multiobjetivo.metodos.metaheuristicas import memetico
from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import evaluate_plan
from despacho_multiobjetivo.metodos.metaheuristicas.genetico import (ConsolidationProblem, ReassignMutation,
                                                                    RegionCrossover, SeededSampling, TruckRepair,
                                                                    chromosome_layout, decode)
from despacho_multiobjetivo.metodos.metaheuristicas.memetico import (MemeticNSGA2, improve_plan, run_memetic,
                                                                    select_for_improvement)
from tests.metodos.instancia_pequena import instancia as _instancia


# ---------------------------------------------------------------- memético: mejorar un individuo con la tabú

def _objetivos(x, instancia):
    resultado = evaluate_plan(decode(x, chromosome_layout(instancia)), instancia)
    return resultado["z1"], resultado["z2"]


def _todos_los_cromosomas():
    return [np.array(x) for x in itertools.product(range(3), range(2), range(3))]


def test_la_tabu_busca_con_epsilon_igual_al_retraso_del_individuo(monkeypatch):
    # ε = Z2 del propio individuo y arranca desde su plan: baja el costo sin empeorar el servicio
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    x = np.array([1, 1, 2])
    plan = decode(x, layout)
    llamadas = []

    def tabu_falsa(instancia, epsilon, **opciones):
        llamadas.append({"epsilon": epsilon, **opciones})
        return {"plan": opciones["plan_inicial"]}

    monkeypatch.setattr(memetico, "tabu_search", tabu_falsa)
    improve_plan(x, layout, instancia, tamano_muestra=7, max_sin_mejora=3, semilla=5)

    assert llamadas[0]["epsilon"] == pytest.approx(evaluate_plan(plan, instancia)["z2"])
    assert llamadas[0]["plan_inicial"] == plan
    assert (llamadas[0]["tamano_muestra"], llamadas[0]["max_sin_mejora"], llamadas[0]["semilla"]) == (7, 3, 5)
    assert llamadas[0]["mostrar"] is False


def test_mejorar_por_defecto_tolera_tres_iteraciones_sin_mejora(monkeypatch):
    instancia = _instancia()
    llamadas = []

    def tabu_falsa(instancia, epsilon, **opciones):
        llamadas.append(opciones)
        return {"plan": opciones["plan_inicial"]}

    monkeypatch.setattr(memetico, "tabu_search", tabu_falsa)
    improve_plan(np.array([1, 1, 2]), chromosome_layout(instancia), instancia)
    assert llamadas[0]["max_sin_mejora"] == 3


def test_mejorar_devuelve_un_cromosoma_valido():
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    mejorado = improve_plan(np.array([1, 1, 2]), layout, instancia)
    assert mejorado.shape == (3,)
    assert all(0 <= gen <= len(dias) for gen, dias in zip(mejorado, layout["dias"]))


def test_mejorar_no_cambia_el_cromosoma_que_recibe():
    instancia = _instancia()
    x = np.array([1, 1, 2])
    improve_plan(x, chromosome_layout(instancia), instancia)
    assert x.tolist() == [1, 1, 2]


def test_mejorar_nunca_empeora_ninguno_de_los_dos_objetivos():
    # Con los 18 cromosomas posibles (1e-6 es la holgura con la que la tabú decide si cumple ε)
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    for x in _todos_los_cromosomas():
        z1, z2 = _objetivos(x, instancia)
        nuevo_z1, nuevo_z2 = _objetivos(improve_plan(x, layout, instancia), instancia)
        assert nuevo_z1 <= z1 + 1e-6
        assert nuevo_z2 <= z2 + 1e-6 + 1e-9


def test_mejorar_baja_el_costo_de_algun_cromosoma():
    # Si nunca bajara el costo, la función no estaría haciendo nada
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    assert any(_objetivos(improve_plan(x, layout, instancia), instancia)[0] < _objetivos(x, instancia)[0] - 1e-6
               for x in _todos_los_cromosomas())


def test_misma_semilla_misma_mejora():
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    x = np.array([2, 1, 0])
    assert (improve_plan(x, layout, instancia, semilla=3).tolist()
            == improve_plan(x, layout, instancia, semilla=3).tolist())


# ---------------------------------------------------------------- memético: a quién mejorar

def test_escoge_primero_a_los_que_tienen_mas_espacio_alrededor():
    rangos = np.array([0, 0, 0, 0])
    apinamiento = np.array([0.1, 0.9, 0.3, 0.5])
    assert select_for_improvement(rangos, apinamiento, 2).tolist() == [1, 3]


def test_solo_escoge_del_frente():
    # El de rango 1 tiene distancia infinita, pero está dominado: no se mejora
    rangos = np.array([0, 1, 0])
    apinamiento = np.array([0.2, np.inf, 0.1])
    assert select_for_improvement(rangos, apinamiento, 2).tolist() == [0, 2]


def test_los_extremos_del_frente_entran_primero():
    rangos = np.array([0, 0, 0, 0])
    apinamiento = np.array([0.5, np.inf, 0.9, np.inf])
    assert select_for_improvement(rangos, apinamiento, 3).tolist() == [1, 3, 2]


def test_si_el_frente_es_pequeno_devuelve_solo_el_frente():
    # No se completa la cantidad con individuos dominados
    rangos = np.array([1, 0, 1, 0])
    apinamiento = np.array([0.9, 0.1, 0.8, 0.2])
    assert select_for_improvement(rangos, apinamiento, 10).tolist() == [3, 1]


def test_en_un_empate_conserva_el_orden_de_la_poblacion():
    # Sin sorteos: con la misma población se escogen siempre los mismos
    rangos = np.zeros(4, dtype=int)
    apinamiento = np.array([0.2, 0.5, 0.5, 0.5])
    assert select_for_improvement(rangos, apinamiento, 2).tolist() == [1, 2]


def test_funciona_con_el_rango_y_el_apinamiento_que_calcula_pymoo():
    # F = (costo, retraso): los dos primeros y el cuarto forman el frente; el tercero está dominado
    F = np.array([[10.0, 1.0], [5.0, 5.0], [9.0, 9.0], [1.0, 9.0]])
    poblacion = Population.new("F", F)
    poblacion.set("G", np.zeros((len(F), 0)))
    poblacion = RankAndCrowding().do(Problem(n_var=1, n_obj=2), poblacion, n_survive=len(F))
    indices = select_for_improvement(poblacion.get("rank"), poblacion.get("crowding"), 3)
    assert sorted(poblacion[indices].get("F")[:, 0].tolist()) == [1.0, 5.0, 10.0]


# ---------------------------------------------------------------- memético: el algoritmo completo

def _algoritmo(instancia, layout, **opciones):
    return MemeticNSGA2(instancia, layout, pop_size=10, sampling=SeededSampling([]),
                        crossover=RegionCrossover(layout["regiones"]), mutation=ReassignMutation(),
                        repair=TruckRepair(instancia, layout), eliminate_duplicates=True, **opciones)


def _correr_memetico(generaciones=4, semilla=0, **opciones):
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    algoritmo = _algoritmo(instancia, layout, **opciones)
    resultado = minimize(ConsolidationProblem(instancia, layout), algoritmo, MaximumGenerationTermination(generaciones),
                         seed=semilla)
    return instancia, layout, resultado.algorithm, resultado   # minimize trabaja sobre una copia del algoritmo


def _espiar_mejora(monkeypatch):
    # Reemplaza improve_plan por una que no cambia nada y anota con qué la llamaron
    llamadas = []

    def mejora_falsa(x, layout, instancia, **opciones):
        llamadas.append({"x": np.array(x), **opciones})
        return np.array(x)

    monkeypatch.setattr(memetico, "improve_plan", mejora_falsa)
    return llamadas


def test_mejora_a_lo_sumo_la_cantidad_pedida_en_cada_generacion(monkeypatch):
    # La primera generación solo evalúa la población inicial: la mejora empieza desde la segunda
    llamadas = _espiar_mejora(monkeypatch)
    _correr_memetico(generaciones=4, individuos=2)
    assert 0 < len(llamadas) <= 2 * 3


def test_le_pasa_a_la_tabu_sus_parametros(monkeypatch):
    llamadas = _espiar_mejora(monkeypatch)
    _correr_memetico(generaciones=2, individuos=1, tamano_muestra=7, max_sin_mejora=3)
    assert (llamadas[0]["tamano_muestra"], llamadas[0]["max_sin_mejora"]) == (7, 3)


def test_por_defecto_la_tabu_tolera_tres_iteraciones_sin_mejora(monkeypatch):
    # Con 1 pararía en el primer paso sin récord y la memoria tabú casi no actuaría
    llamadas = _espiar_mejora(monkeypatch)
    _correr_memetico(generaciones=2, individuos=1)
    assert llamadas[0]["max_sin_mejora"] == 3


def test_cada_llamada_a_la_tabu_tiene_su_propia_semilla(monkeypatch):
    # Si no, la misma población repetiría exactamente la misma búsqueda
    llamadas = _espiar_mejora(monkeypatch)
    _correr_memetico(generaciones=5, individuos=2)
    assert len({llamada["semilla"] for llamada in llamadas}) > 1


def test_sin_tiempo_no_mejora_a_nadie(monkeypatch):
    llamadas = _espiar_mejora(monkeypatch)
    _correr_memetico(generaciones=3, max_segundos=0)
    assert llamadas == []


def test_el_mejorado_reemplaza_al_original_y_queda_evaluado(monkeypatch):
    instancia, layout, algoritmo, _ = _correr_memetico(generaciones=2, individuos=1)
    actuales = {tuple(x) for x in algoritmo.pop.get("X").astype(int).tolist()}
    nuevo = next(x for x in itertools.product(range(3), range(2), range(3)) if x not in actuales)
    tamano = len(algoritmo.pop)
    escogido = select_for_improvement(algoritmo.pop.get("rank"), algoritmo.pop.get("crowding"), 1)[0]
    original = tuple(algoritmo.pop[escogido].X.astype(int).tolist())
    monkeypatch.setattr(memetico, "improve_plan", lambda x, layout, instancia, **opciones: np.array(nuevo))

    algoritmo._improve_population()

    finales = {tuple(x): f for x, f in zip(algoritmo.pop.get("X").astype(int).tolist(), algoritmo.pop.get("F").tolist())}
    assert len(algoritmo.pop) == tamano
    assert original not in finales
    assert finales[nuevo] == pytest.approx(list(_objetivos(np.array(nuevo), instancia)))


def test_no_mete_duplicados_a_la_poblacion(monkeypatch):
    # Si la mejora devuelve un cromosoma que ya está en la población, se conserva el original
    instancia, layout, algoritmo, _ = _correr_memetico(generaciones=2, individuos=2)
    existente = np.array(algoritmo.pop[0].X).astype(int)
    antes = sorted(algoritmo.pop.get("X").astype(int).tolist())
    monkeypatch.setattr(memetico, "improve_plan", lambda x, layout, instancia, **opciones: existente.copy())
    algoritmo._improve_population()
    assert sorted(algoritmo.pop.get("X").astype(int).tolist()) == antes


def test_el_frente_final_coincide_con_el_evaluador():
    instancia, layout, _, resultado = _correr_memetico(generaciones=5, individuos=3)
    for x, f in zip(np.atleast_2d(resultado.X), np.atleast_2d(resultado.F)):
        assert list(f) == pytest.approx(list(_objetivos(x, instancia)))


def test_misma_semilla_mismo_frente_memetico():
    a = _correr_memetico(generaciones=5, semilla=2, individuos=3)[3]
    b = _correr_memetico(generaciones=5, semilla=2, individuos=3)[3]
    assert np.atleast_2d(a.F).tolist() == np.atleast_2d(b.F).tolist()


# ---------------------------------------------------------------- memético: la corrida

def test_la_corrida_devuelve_lo_mismo_que_el_genetico():
    # Así se pueden comparar con las mismas funciones (best_for_epsilon y los scripts)
    resultado = run_memetic(_instancia(), pop_size=10, max_generaciones=3, individuos=2)
    assert set(resultado) == {"frente", "generaciones", "segundos", "semilla", "historial"}
    assert list(resultado["frente"].columns) == ["z1", "z2", "plan"]
    assert resultado["generaciones"] == 3


def test_el_frente_de_la_corrida_coincide_con_el_evaluador():
    instancia = _instancia()
    resultado = run_memetic(instancia, pop_size=10, max_generaciones=5, individuos=3)
    for _, fila in resultado["frente"].iterrows():
        evaluado = evaluate_plan(fila["plan"], instancia)
        assert (evaluado["z1"], evaluado["z2"]) == pytest.approx((fila["z1"], fila["z2"]))


def test_la_corrida_le_pasa_los_parametros_a_la_mejora(monkeypatch):
    llamadas = _espiar_mejora(monkeypatch)
    run_memetic(_instancia(), pop_size=10, max_generaciones=2, individuos=1, tamano_muestra=7, max_sin_mejora=2)
    assert len(llamadas) == 1
    assert (llamadas[0]["tamano_muestra"], llamadas[0]["max_sin_mejora"]) == (7, 2)


def test_la_corrida_usa_las_semillas_de_la_poblacion_inicial():
    # Con el plan más barato de la instancia como semilla, el frente no puede quedar más caro que él
    instancia = _instancia()
    plan = decode(np.array([1, 1, 1]), chromosome_layout(instancia))
    costo = evaluate_plan(plan, instancia)["z1"]
    resultado = run_memetic(instancia, semillas=[plan], pop_size=10, max_generaciones=2, individuos=1)
    assert resultado["frente"]["z1"].min() <= costo + 1e-6


def test_misma_semilla_misma_corrida_memetica():
    a = run_memetic(_instancia(), pop_size=10, max_generaciones=4, individuos=2, semilla=1)
    b = run_memetic(_instancia(), pop_size=10, max_generaciones=4, individuos=2, semilla=1)
    assert a["frente"][["z1", "z2"]].values.tolist() == b["frente"][["z1", "z2"]].values.tolist()


def test_la_corrida_para_por_tiempo():
    resultado = run_memetic(_instancia(), pop_size=10, max_segundos=1, individuos=1)
    assert resultado["segundos"] < 20


# ---------------------------------------------------------------- memético con la tabú mejorada (camiones y regiones)

def _tabu_espia(monkeypatch):
    llamadas = []

    def tabu_falsa(instancia, epsilon, **opciones):
        llamadas.append({"epsilon": epsilon, **opciones})
        return {"plan": opciones["plan_inicial"]}

    monkeypatch.setattr(memetico, "tabu_search", tabu_falsa)
    return llamadas


def test_mejorar_le_pasa_a_la_tabu_los_camiones_y_lo_preparado(monkeypatch):
    llamadas = _tabu_espia(monkeypatch)
    instancia = _instancia()
    improve_plan(np.array([1, 1, 2]), chromosome_layout(instancia), instancia, prob_camion=0.3)
    assert llamadas[0]["prob_camion"] == 0.3
    assert llamadas[0]["preparado"] is not None


def test_por_defecto_la_mejora_no_usa_movimientos_de_camion(monkeypatch):
    # Así las corridas anteriores del memético siguen dando lo mismo
    llamadas = _tabu_espia(monkeypatch)
    instancia = _instancia()
    improve_plan(np.array([1, 1, 2]), chromosome_layout(instancia), instancia)
    assert llamadas[0]["prob_camion"] == 0


def test_con_camiones_mejorar_nunca_empeora_ninguno_de_los_dos_objetivos():
    instancia = _instancia()
    layout = chromosome_layout(instancia)
    for x in _todos_los_cromosomas():
        z1, z2 = _objetivos(x, instancia)
        nuevo_z1, nuevo_z2 = _objetivos(improve_plan(x, layout, instancia, prob_camion=0.5), instancia)
        assert nuevo_z1 <= z1 + 1e-6 and nuevo_z2 <= z2 + 1e-9


def test_el_algoritmo_prepara_la_tabu_una_sola_vez(monkeypatch):
    # Días por pedido, camiones y regiones no cambian entre una tabú corta y otra: se calculan al crear el algoritmo
    preparar = memetico.prepare_tabu
    veces = []
    monkeypatch.setattr(memetico, "prepare_tabu", lambda *args, **kwargs: veces.append(1) or preparar(*args, **kwargs))
    _correr_memetico(generaciones=4, individuos=2)
    assert len(veces) == 1


def test_le_pasa_a_la_mejora_los_camiones_y_lo_preparado(monkeypatch):
    llamadas = _espiar_mejora(monkeypatch)
    _, _, algoritmo, _ = _correr_memetico(generaciones=2, individuos=1, prob_camion=0.3)
    assert llamadas[0]["prob_camion"] == 0.3
    assert llamadas[0]["preparado"] is not None


def test_la_corrida_le_pasa_la_proporcion_de_camiones(monkeypatch):
    llamadas = _espiar_mejora(monkeypatch)
    run_memetic(_instancia(), pop_size=10, max_generaciones=2, individuos=1, prob_camion=0.5)
    assert llamadas[0]["prob_camion"] == 0.5
