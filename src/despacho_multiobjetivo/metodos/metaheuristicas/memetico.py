"""Versión memética del genético NSGA-II: algunos individuos se mejoran con una tabú corta en cada generación.

Reutiliza el cromosoma y los operadores de genetico.py y la búsqueda de tabu.py. La tabú busca con ε igual al Z2 del
propio individuo, así baja su costo sin empeorar su servicio (Ishibuchi, Yoshida y Murata, 2003; Lust y Teghem, 2008).
"""

import time

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.max_time import TimeBasedTermination

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import evaluate_by_region
from despacho_multiobjetivo.metodos.metaheuristicas.genetico import (ConsolidationProblem, GenerationHistory,
                                                                    ReassignMutation, RegionCrossover,
                                                                    SeededSampling, TruckRepair,
                                                                    chromosome_layout, decode, encode)
from despacho_multiobjetivo.metodos.metaheuristicas.tabu import prepare_tabu, tabu_search


def improve_plan(x, layout: dict, instancia: dict, tamano_muestra: int = 20, max_sin_mejora: int = 3,
                 semilla: int = 0, filtrar_retraso: bool = True, max_segundos: float = 10,
                 prob_camion: float = 0.0, preparado: dict | None = None) -> np.ndarray:
    # Mejora un individuo con una tabú corta: baja su costo sin empeorar su servicio.
    # Devuelve un cromosoma nuevo; el que recibe no se modifica.
    # prob_camion: parte de la muestra de cada tabú que son movimientos de camión
    # preparado: lo de prepare_tabu (si es None se calcula aquí)
    preparado = preparado or prepare_tabu(instancia, filtrar_retraso)

    # 1. Del cromosoma al plan
    plan = decode(x, layout)

    # 2. ε = el Z2 de este mismo individuo, calculado igual que en la tabú (el plan de partida siempre cumple,
    #    así que el resultado no puede ser peor)
    epsilon = evaluate_by_region(plan, preparado["regiones"])["z2"]

    # 3. Tabú corta desde ese plan, sin imprimir nada
    resultado = tabu_search(instancia, epsilon, plan_inicial=plan, semilla=semilla, tamano_muestra=tamano_muestra,
                            max_sin_mejora=max_sin_mejora, max_segundos=max_segundos,
                            filtrar_retraso=filtrar_retraso, mostrar=False,
                            prob_camion=prob_camion, preparado=preparado)

    # 4. Del plan mejorado al cromosoma
    return encode(resultado["plan"], layout)


def select_for_improvement(rangos: np.ndarray, apinamiento: np.ndarray, cantidad: int) -> np.ndarray:
    # A quién mejorar en esta generación: los individuos del frente con más espacio alrededor.
    # rangos y apinamiento: los que NSGA-II ya calculó (población.get("rank") y población.get("crowding")).
    # Devuelve sus posiciones en la población; si el frente tiene menos de 'cantidad', solo los del frente.

    # 1. Posiciones de los individuos del frente (rango 0: nadie los domina)
    frente = np.flatnonzero(rangos == 0)

    # 2. De más a menos espacio; "stable" conserva el orden de la población en los empates, sin sorteos
    orden = np.argsort(-apinamiento[frente], kind="stable")

    # 3. Los primeros 'cantidad'
    return frente[orden[:cantidad]]


class MemeticNSGA2(NSGA2):
    # NSGA-II que, después de la selección de cada generación, mejora con una tabú corta a los individuos del frente
    # con más espacio alrededor. El mejorado reemplaza al original (Ishibuchi, Yoshida y Murata, 2003).
    # individuos: cuántos se mejoran por generación; tamano_muestra y max_sin_mejora: los de cada tabú corta
    # max_segundos: deja de mejorar al cumplirse el tiempo, para no pasarse del presupuesto de la corrida
    # prob_camion: parte de la muestra de cada tabú corta que son movimientos de camión
    def __init__(self, instancia: dict, layout: dict, individuos: int = 10, tamano_muestra: int = 20,
                 max_sin_mejora: int = 3, filtrar_retraso: bool = True, max_segundos: float | None = None,
                 prob_camion: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.instancia = instancia
        self.layout = layout
        self.individuos = individuos
        self.tamano_muestra = tamano_muestra
        self.max_sin_mejora = max_sin_mejora
        self.filtrar_retraso = filtrar_retraso
        self.max_segundos = max_segundos
        self.prob_camion = prob_camion
        self.preparado = prepare_tabu(instancia, filtrar_retraso)   # una sola vez para todas las tabúes cortas

    def _advance(self, infills=None, **kwargs):
        # Lo normal de NSGA-II (juntar padres e hijos y seleccionar) y luego la mejora
        super()._advance(infills=infills, **kwargs)
        self._improve_population()

    def _improve_population(self) -> None:
        # 1. A quién mejorar
        posiciones = select_for_improvement(self.pop.get("rank"), self.pop.get("crowding"), self.individuos)
        X = self.pop.get("X").astype(int)

        # 2. Mejorar a cada uno; si queda igual a otro de la población, se conserva el original
        cambiaron = []
        for posicion in posiciones:
            if self.max_segundos is not None and time.time() - self.start_time >= self.max_segundos:
                break
            semilla = int(self.random_state.integers(2 ** 31))
            mejorado = improve_plan(X[posicion], self.layout, self.instancia, tamano_muestra=self.tamano_muestra,
                                    max_sin_mejora=self.max_sin_mejora, semilla=semilla,
                                    filtrar_retraso=self.filtrar_retraso,
                                    prob_camion=self.prob_camion, preparado=self.preparado)
            if not (X == mejorado).all(axis=1).any():
                X[posicion] = mejorado
                cambiaron.append(posicion)

        # 3. Evaluar a los que cambiaron, ponerlos en la población y recalcular rangos y apiñamiento
        if cambiaron:
            nuevos = Population.new("X", X[cambiaron])
            self.evaluator.eval(self.problem, nuevos, algorithm=self)
            self.pop[cambiaron] = nuevos
            self.pop = self.survival.do(self.problem, self.pop, n_survive=self.pop_size, algorithm=self,
                                        random_state=self.random_state)


def run_memetic(instancia: dict, semillas: list = (), pop_size: int = 100, max_segundos: float = 600,
                max_generaciones: int | None = None, semilla: int = 0, filtrar_retraso: bool = True,
                mostrar: bool = False, individuos: int = 10, tamano_muestra: int = 20,
                max_sin_mejora: int = 3, prob_camion: float = 0.0) -> dict:
    # Corrida del memético con los mismos operadores y la misma salida de run_nsga2, para compararlos directamente
    layout = chromosome_layout(instancia)
    problema = ConsolidationProblem(instancia, layout, filtrar_retraso)
    algoritmo = MemeticNSGA2(instancia, layout, individuos=individuos, tamano_muestra=tamano_muestra,
                             max_sin_mejora=max_sin_mejora, filtrar_retraso=filtrar_retraso, prob_camion=prob_camion,
                             max_segundos=None if max_generaciones else max_segundos, pop_size=pop_size,
                             sampling=SeededSampling([encode(plan, layout) for plan in semillas]),
                             crossover=RegionCrossover(layout["regiones"]), mutation=ReassignMutation(),
                             repair=TruckRepair(instancia, layout), eliminate_duplicates=True)
    parada = MaximumGenerationTermination(max_generaciones) if max_generaciones else TimeBasedTermination(max_segundos)
    historial = GenerationHistory()

    inicio = time.perf_counter()
    resultado = minimize(problema, algoritmo, parada, seed=semilla, verbose=mostrar, callback=historial)

    X, F = np.atleast_2d(resultado.X), np.atleast_2d(resultado.F)
    frente = pd.DataFrame({"z1": F[:, 0], "z2": F[:, 1], "plan": [decode(x, layout) for x in X]})
    frente = frente.sort_values("z2").reset_index(drop=True)
    return {"frente": frente, "generaciones": len(historial.filas), "segundos": time.perf_counter() - inicio,
            "semilla": semilla, "historial": pd.DataFrame(historial.filas)}
