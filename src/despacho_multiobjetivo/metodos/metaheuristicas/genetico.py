"""Algoritmo genético NSGA-II (pymoo): aproxima la frontera costo–servicio (Z1, Z2) con una población de planes.

Cada plan es un cromosoma: un vector de enteros, uno por pedido que puede subir a un camión (0 = directo,
k = su k-ésimo día posible). Se evalúa con el mismo evaluador de la tabú. El cruce es por región, para no
desarmar los camiones de cada padre, y la población inicial mezcla planes aleatorios con semillas
(todo directo y soluciones del MILP).
"""

import math
import time

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.max_time import TimeBasedTermination

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import (departure_days, evaluate_by_region,
                                                                    prepare_regions)

def chromosome_layout(instancia: dict) -> dict:
    # Qué significa cada posición del cromosoma, fijo durante toda la corrida.
    # Devuelve {"pedidos": [id_pedido, ...], "dias": [[dia, ...], ...], "regiones": [region, ...]}
    #   la posición i es el pedido pedidos[i]; su gen vale 0 (directo) o k (sale el día dias[i][k - 1])   #Que 
    # 1. Días de cada pedido que se puede mover (los de regiones con camión)
    dias_por_pedido = departure_days(instancia)

    # 2. Orden fijo de los pedidos, y sus días en orden cronológico (gen 1 = el primer día posible)
    pedidos = sorted(dias_por_pedido)
    dias = [sorted(dias_por_pedido[id_pedido]) for id_pedido in pedidos]

    # 3. Región de cada pedido (para el cruce por región)
    regiones = instancia["pedidos"].loc[pedidos, "region_logistica"].tolist()

    return {"pedidos": pedidos, "dias": dias, "regiones": regiones}


def decode(x, layout: dict) -> dict:
    #Cromosoma -> plan para evaluate_plan: {id_pedido: dia} de los genes distintos de 0 (los de 0 van drecto)
    # Ejemplo: x = [0, 2, 1] con dias = [[lun, mar], [lun, mar], [mié]] → {pedido_1: mar, pedido_2: mié}
    pedidos, dias = layout["pedidos"], layout["dias"]

    plan = {}
    for i, gen in enumerate(x):
        gen = int(round(gen)) #pymoo puede entregar los genes como decimales: se pasan a entero
        if gen > 0:
            plan[pedidos[i]] = dias[i][gen-1]
    return plan
        
def encode(plan: dict, layout: dict) -> np.ndarray:
    # Plan → cromosoma: para sembrar la población con soluciones del MILP (plan_from_milp) o con todo directo ({})
    # Un traslado partido ({dia: fraccion}) queda en 0: el cromosoma no guarda fracciones y evaluate_plan lo rellena
    # en el espacio libre de los camiones. Ponerlo completo en un día compraría más camión del que pagó el MILP
    posicion = {id_pedido: i for i, id_pedido in enumerate(layout["pedidos"])}
    x = np.zeros(len(layout["pedidos"]), dtype=int)      # todo en 0 = todo directo

    for id_pedido, salida in plan.items():
        if id_pedido not in posicion:
            raise ValueError(f"{id_pedido} no está en el cromosoma: no puede salir en camión")
        if isinstance(salida, dict):
            continue
        i = posicion[id_pedido]
        x[i] = layout["dias"][i].index(salida) + 1
    return x

class ConsolidationProblem(ElementwiseProblem):
    # El problema de consolidación para pymoo: un gen por pedido que se puede mover (0 = directo, k = su k-ésimo día),
    # y dos objetivos que se minimizan: costo (Z1) y días de retraso ponderados por pedido (Z2), con el mismo
    # evaluador de la tabú

    def __init__(self, instancia: dict, layout:dict, filtrar_retraso: bool = True):
        #1. Datos que necesita _Evaluate, guardados dentro del problema
        self.instancia = instancia
        self.layout = layout
        self.filtrar_retraso = filtrar_retraso
        self.regiones = prepare_regions(instancia, filtrar_retraso)   # evaluación rápida, la misma de la tabú

        #2. Rango de cada gen: de 0 (directo) al número de días posibles de su pedido
        n_var = len(layout["pedidos"])
        maximos = np.array([len(dias) for dias in layout["dias"]])

        #3. Descripción del problema para el molde de pymoo
        super().__init__(n_var=n_var, n_obj=2, xl=np.zeros(n_var), xu=maximos, vtype=int)

    def _evaluate(self, x, out, *args, **kwargs):
        # pymoo la llama con un individuo x (su cromosoma) y espera los objetivos en out["F"]
        plan = decode(x, self.layout)
        resultado = evaluate_by_region(plan, self.regiones)
        out["F"] = [resultado["z1"], resultado["z2"]]


class SeededSampling(Sampling):
    #Poblacion inicial: primero las semillas (todo directo, planes del Milp) tal cual, y el resto aleatorio
    #con densidades distintas: cada individuo sube al camión una proporció distinta de pedidos (de casi nada a casi todo)
    #para que la población cubra desde el lado de mejor servicio hasta el menor costo

    def __init__(self, semillas:list):
        super().__init__()
        self.semillas = semillas

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        #pymoo la llama una vez al inicio; devuelve un matri de n_samples filas (individuos) x n_var columnas (genes)

        #1. Las semillas que quepan, tal cual
        filas = [np.asarray(semilla, dtype=int) for semilla in self.semillas[:n_samples]]

        #2. El resto: individuos aleatorios, cda uno con su propia densidad
        maximos = problem.xu.astype(int)
        while len(filas) < n_samples:
            densidad = random_state.random() #proporción de pedidos que suben al camión
            en_camion = random_state.random(problem.n_var) < densidad # ue genes suben (True) y cuales van directo
            dia = random_state.integers(1, maximos + 1) # un dia al azar para cada gen, de 1 a su máximo
            filas.append(np.where(en_camion, dia, 0)) # el dia si sube, 0 si va directo

        #3. Todo junto una fila por individuo
        return np.array(filas, dtype=int)


class RegionCrossover(Crossover):
    # Cruce por región: el hijo hereda todos los genes de una región de un mismo padre (una moneda por región),
    # para no desarmar los camiones que cada padre armó. Se apoya en que el costo de los camiones de una región
    # no depende de las demás

    def __init__(self, regiones: list):
        super().__init__(n_parents=2, n_offsprings=2)   # 2 padres -> 2 hijos
        regiones = np.array(regiones)
        # Posiciones del cromosoma de cada región, calculadas una sola vez
        self.grupos = [np.where(regiones == region)[0] for region in sorted(set(regiones))]

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        # X: matriz de padres (2 x n_cruces x n_genes); devuelve los hijos con la misma forma
        _, n_cruces, _ = X.shape
        hijos = X.copy()                                # hijo 1 = padre A e hijo 2 = padre B, de entrada
        for k in range(n_cruces):                       # cada pareja de padres
            for posiciones in self.grupos:              # cada región
                if random_state.random() < 0.5:         # sello: se intercambia esta región entre los hijos
                    hijos[0, k, posiciones] = X[1, k, posiciones]
                    hijos[1, k, posiciones] = X[0, k, posiciones]
        return hijos


class ReassignMutation(Mutation):
    # Mutación por reasignación: cada gen, con probabilidad prob_var, pasa a OTRO de sus valores posibles,
    # escogido al azar con la misma probabilidad (directo o cualquiera de sus otros días). Los días son opciones,
    # no números cercanos: de "día 2" se puede pasar a "directo" igual de fácil que a "día 3"

    def __init__(self, prob_var=None):
        # prob_var: probabilidad de mutar cada gen; None = la de pymoo por defecto, 1 / número de genes
        super().__init__(prob=1.0, prob_var=prob_var)

    def _do(self, problem, X, *args, random_state=None, **kwargs):
        # X: matriz de individuos (n_individuos × n_genes); devuelve los individuos mutados con la misma forma
        mutados = np.array(X, dtype=int)                                     # copia: X no se modifica
        maximos = np.broadcast_to(problem.xu.astype(int), mutados.shape)     # máximo de cada gen, en cada fila

        # 1. Qué genes mutan
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        mutan = random_state.random(mutados.shape) < prob_var

        # 2. A qué valor pasan: uno al azar entre los OTROS valores posibles
        otro = random_state.integers(0, maximos)       # de 0 a máximo − 1 (hay un valor menos: el actual)
        otro = otro + (otro >= mutados)                # se salta el valor actual

        # 3. Aplicar solo en los genes que mutan
        mutados[mutan] = otro[mutan]
        return mutados


class TruckRepair(Repair):
    # Reparación (idea de Chu y Beasley, 1997): después del cruce y la mutación, revisa cada camión del hijo
    # (región, día) y lo cierra —sus pedidos pasan a directo— si a la vez:
    #   - no se paga: el camión cuesta más que lo que ahorran sus pedidos frente a ir directo, y
    #   - cerrarlo no empeora el servicio: directo, esos pedidos no suman más castigo por retraso
    # Así la solución reparada nunca es peor que la original en ninguno de los dos objetivos: no se pierde
    # ningún punto de la frontera, solo se limpian camiones casi vacíos que el cruce o la mutación dejaron sueltos

    def __init__(self, instancia: dict, layout: dict):
        super().__init__()
        pedidos, opciones = instancia["pedidos"], instancia["opciones"]
        self.tablas = instancia["tablas"]
        self.regiones = layout["regiones"]
        self.peso = pedidos.loc[layout["pedidos"], "peso_kg"].to_numpy()
        # Para cada gen y cada día posible: (día, ahorro de subirlo, cambio en el castigo si vuelve a directo)
        self.salidas = []
        for id_pedido, dias in zip(layout["pedidos"], layout["dias"]):
            directo = pedidos.loc[id_pedido]
            self.salidas.append([(dia,
                                  directo["flete_directo"] - opciones.loc[(id_pedido, dia), "flete_triangulacion"],
                                  directo["castigo_directo"] - opciones.loc[(id_pedido, dia), "castigo"])
                                 for dia in dias])

    def _do(self, problem, X, **kwargs):
        reparados = np.array(X, dtype=int)                 # copia: X no se modifica
        for fila in reparados:
            # 1. Camiones del individuo: genes, kilos, ahorro y cambio de castigo por (región, día)
            camiones = {}
            for i in np.flatnonzero(fila):
                dia, ahorro, cambio_castigo = self.salidas[i][fila[i] - 1]
                camion = camiones.setdefault((self.regiones[i], dia), {"genes": [], "kg": 0.0, "ahorro": 0.0,
                                                                        "cambio_castigo": 0.0})
                camion["genes"].append(i)
                camion["kg"] += self.peso[i]
                camion["ahorro"] += ahorro
                camion["cambio_castigo"] += cambio_castigo

            # 2. Cerrar los que no se pagan y cuyo cierre no empeora el servicio
            for (region, _), camion in camiones.items():
                costo, _, _ = self.tablas[region]
                kg = math.ceil(camion["kg"] - 1e-6)
                if kg < len(costo) and costo[kg] > camion["ahorro"] and camion["cambio_castigo"] <= 0:
                    fila[camion["genes"]] = 0
        return reparados


class GenerationHistory(Callback):
    # Lo que pasa en cada generación, para la curva de convergencia: tamaño del frente y mejor costo y mejor
    # retraso dentro del frente

    def __init__(self):
        super().__init__()
        self.inicio = time.perf_counter()
        self.filas = []

    def notify(self, algorithm):
        F = algorithm.opt.get("F")                      # objetivos del frente no dominado actual
        self.filas.append({"generacion": algorithm.n_gen, "segundos": time.perf_counter() - self.inicio,
                           "tamano_frente": len(F), "z1_minimo": F[:, 0].min(), "z2_minimo": F[:, 1].min()})


def run_nsga2(instancia: dict, semillas: list = (), pop_size: int = 100, max_segundos: float = 600,
              max_generaciones: int | None = None, semilla: int = 0, filtrar_retraso: bool = True,
              mostrar: bool = False, operadores: str = "propios") -> dict:
    # NSGA-II de pymoo con los operadores propios: población inicial con semillas, cruce por región, mutación por
    # reasignación y reparación de camiones que no se pagan. Aproxima la frontera costo–servicio en una corrida.
    # semillas: planes para la población inicial (todo directo = {}, planes de plan_from_milp, …)
    # Para por tiempo (max_segundos) o, si se da, por número de generaciones (para pruebas que se puedan repetir)
    # operadores: "propios" (cruce por región, mutación por reasignación y reparación de camiones) o "genericos"
    #   (cruce SBX y mutación polinomial con redondeo, lo que trae pymoo para enteros, sin reparación), para medir
    #   cuánto aportan los propios; la población inicial es la misma en los dos casos
    # Devuelve {"frente" (z1, z2, plan, ordenado por z2), "generaciones", "segundos", "semilla", "historial"}
    layout = chromosome_layout(instancia)
    problema = ConsolidationProblem(instancia, layout, filtrar_retraso)
    poblacion_inicial = SeededSampling([encode(plan, layout) for plan in semillas])
    if operadores == "propios":
        algoritmo = NSGA2(pop_size=pop_size, sampling=poblacion_inicial,
                          crossover=RegionCrossover(layout["regiones"]), mutation=ReassignMutation(),
                          repair=TruckRepair(instancia, layout), eliminate_duplicates=True)
    elif operadores == "genericos":
        algoritmo = NSGA2(pop_size=pop_size, sampling=poblacion_inicial,
                          crossover=SBX(vtype=float, repair=RoundingRepair()),
                          mutation=PM(vtype=float, repair=RoundingRepair()), eliminate_duplicates=True)
    else:
        raise ValueError(f"operadores debe ser 'propios' o 'genericos': {operadores}")
    parada = MaximumGenerationTermination(max_generaciones) if max_generaciones else TimeBasedTermination(max_segundos)
    historial = GenerationHistory()

    inicio = time.perf_counter()
    resultado = minimize(problema, algoritmo, parada, seed=semilla, verbose=mostrar, callback=historial)

    X, F = np.atleast_2d(resultado.X), np.atleast_2d(resultado.F)
    frente = pd.DataFrame({"z1": F[:, 0], "z2": F[:, 1], "plan": [decode(x, layout) for x in X]})
    frente = frente.sort_values("z2").reset_index(drop=True)
    # Generaciones: las que registró el historial (pymoo deja su contador una unidad adelante al terminar)
    return {"frente": frente, "generaciones": len(historial.filas), "segundos": time.perf_counter() - inicio,
            "semilla": semilla, "historial": pd.DataFrame(historial.filas)}


def best_for_epsilon(resultado: dict, epsilon: float, tolerancia: float = 1e-6) -> dict | None:
    # El plan más barato del frente que cumple Z2 <= ε (para comparar con el MILP y la tabú en el mismo punto);
    # None si ningún plan del frente cumple
    frente = resultado["frente"]
    cumplen = frente[frente["z2"] <= epsilon + tolerancia]
    if cumplen.empty:
        return None
    mejor = cumplen.loc[cumplen["z1"].idxmin()]
    return {"plan": mejor["plan"], "z1": mejor["z1"], "z2": mejor["z2"]}
