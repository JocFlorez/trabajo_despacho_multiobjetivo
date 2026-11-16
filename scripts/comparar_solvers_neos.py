"""Comparación de solvers comerciales (CPLEX, FICO Xpress) en el servidor NEOS frente a HiGHS local, con el mismo modelo.

No hay licencias comerciales: NEOS ejecuta los solvers en sus servidores. Gurobi no está disponible por programa en
NEOS (solo por su formulario web), así que se comparan CPLEX y Xpress. Enviar el modelo es mandar datos a un
servidor externo, así que se envía ANONIMIZADO Y ESCALADO:
  - formato AMPL (.nl) sin nombres: variables y restricciones numeradas, sin pedidos, clientes, regiones ni camiones
  - el costo que se minimiza se multiplica por un factor aleatorio que no se guarda: el solver ve otros números,
    pero el plan óptimo es el mismo (multiplicar el objetivo por una constante positiva no cambia el óptimo)
NEOS devuelve el plan: su costo y su retraso se calculan aquí con el modelo original.

Casos (los mismos de comparar_pyomo.py):
  min_costo          mínimo costo sin límite de retraso (el más difícil para el solver)
  epsilon_actual     mínimo costo con Z2 <= castigo por retraso de la operación actual

Uso:
  python scripts/comparar_solvers_neos.py              solo escribe los .nl que se enviarían, para revisarlos
  python scripts/comparar_solvers_neos.py --enviar     envía a NEOS (necesita la variable de entorno NEOS_EMAIL)
Resultados en data/processed/neos/: los .nl, el registro de cada solver y comparacion_solvers.csv
"""
import argparse
import os
import random
import time

import pandas as pd
import pyomo.environ as pyo
from pyomo.common.tee import capture_output

from despacho_multiobjetivo.config import (SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PROCESSED,
                                           GAP_REL_FRONTERA, LIMITE_SEGUNDOS_FRONTERA)
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import opciones_salida, servicio
from despacho_multiobjetivo.modelos import version_pyomo as modelo_pyomo
from despacho_multiobjetivo.modelos.version_pyomo import set_objective
from despacho_multiobjetivo.metodos import resolver_pyomo

CARPETA = PROCESSED / "neos"
# Nombre de cada solver en NEOS y sus opciones en la interfaz AMPL: mismo gap y mismo tiempo límite que HiGHS
SOLVERS_NEOS = {
    "cplex": ("cplex", {"time": LIMITE_SEGUNDOS_FRONTERA, "mipgap": GAP_REL_FRONTERA, "mipdisplay": 2, "timing": 1}),
    "xpress": ("fico-xpress", {"maxtime": LIMITE_SEGUNDOS_FRONTERA, "miprelstop": GAP_REL_FRONTERA, "outlev": 1}),
}


def prepare_case(modelo: pyo.ConcreteModel, caso: str, epsilon: float, factor: float) -> None:
    # Deja el modelo listo para el caso: objetivo escalado y, si aplica, el límite de retraso
    if hasattr(modelo, "epsilon"):
        modelo.del_component(modelo.epsilon)
    if caso == "epsilon_actual":
        modelo.epsilon = pyo.Constraint(expr=modelo.z2 <= epsilon * (1 + 1e-6))
    set_objective(modelo, factor * modelo.z1)


def solve_neos(modelo: pyo.ConcreteModel, solver: str) -> dict:
    # Envía a NEOS el modelo ya preparado; devuelve el plan, así que costo y retraso se calculan con el modelo original.
    # El tiempo medido incluye la subida y la cola del servidor: es una cota superior del tiempo de solución.
    # Pyomo borra el registro detallado del solver; se guarda lo que sí devuelve NEOS: lo que se alcanzó a mostrar
    # mientras corría y el resumen final (estado, mensaje del solver, iteraciones y nodos)
    nombre_neos, opciones = SOLVERS_NEOS[solver]
    inicio = time.perf_counter()
    with capture_output() as salida:
        resultado = pyo.SolverManagerFactory("neos").solve(modelo, opt=nombre_neos, options=opciones, tee=True)
    return {"estado": str(resultado.solver.termination_condition), "costo": pyo.value(modelo.z1),
            "retraso": pyo.value(modelo.z2), "segundos_con_cola": time.perf_counter() - inicio,
            "registro": f"{salida.getvalue()}\n{resultado}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CPLEX y Xpress en NEOS frente a HiGHS, con el modelo anonimizado")
    parser.add_argument("--enviar", action="store_true", help="envía a NEOS (sin esto solo escribe los .nl)")
    parser.add_argument("--solvers", nargs="+", default=["cplex", "xpress"], choices=sorted(SOLVERS_NEOS))
    parser.add_argument("--casos", nargs="+", default=["epsilon_actual", "min_costo"], choices=["epsilon_actual", "min_costo"])
    parser.add_argument("--sin-highs", action="store_true", help="no resolver la referencia local con HiGHS")
    args = parser.parse_args()
    if args.enviar and not os.environ.get("NEOS_EMAIL"):
        raise SystemExit("NEOS exige un correo: defina la variable de entorno NEOS_EMAIL antes de enviar")
    CARPETA.mkdir(parents=True, exist_ok=True)

    print("Cargando datos")
    pedidos, nodos_regionales, capacidades, tarifas_ftl, tiempos_ftl = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    opciones = opciones_salida.build_departure_options(pedidos, tiempos_ftl, ESPERA_MAXIMA)
    epsilon = servicio.expected_delay_direct(pedidos).mean()
    modelo = modelo_pyomo.build_model(opciones, pedidos, capacidades, tarifas_ftl, nodos_regionales, SEDE)
    modelo.name = "modelo"   # el nombre del modelo también viaja en el encabezado del .nl
    factor = random.SystemRandom().uniform(0.2, 5.0)   # secreto: no se imprime ni se guarda

    filas = []
    for caso in args.casos:
        # 1. Referencia local con HiGHS (no envía nada)
        if not args.sin_highs:
            print(f"\n{caso} · HiGHS local")
            if caso == "epsilon_actual":
                highs = resolver_pyomo.epsilon_point(modelo, epsilon, mostrar=False)
            else:
                highs = resolver_pyomo.cost_extreme(modelo, mostrar=False)
            print(f"   {highs['estado']} · costo {highs['costo']:,.0f} · Z2 {highs['retraso']:.4f} · {highs['segundos_resolver']:.0f} s")
            filas.append({"caso": caso, "solver": "highs (local)", "estado": highs["estado"], "costo": highs["costo"],
                          "retraso": highs["retraso"], "segundos_con_cola": highs["segundos_resolver"]})

        # 2. Lo que se enviaría: el modelo anonimizado y escalado en formato AMPL, para revisarlo antes
        prepare_case(modelo, caso, epsilon, factor)
        ruta_nl = CARPETA / f"{caso}.nl"
        modelo.write(str(ruta_nl), io_options={"symbolic_solver_labels": False})
        print(f"   Modelo anonimizado en {ruta_nl}")

        # 3. NEOS
        if args.enviar:
            for solver in args.solvers:
                print(f"{caso} · {solver} en NEOS (incluye la espera en la cola del servidor)")
                neos = solve_neos(modelo, solver)
                (CARPETA / f"{caso}_{solver}.log").write_text(neos.pop("registro"), encoding="utf-8")
                print(f"   {neos['estado']} · costo {neos['costo']:,.0f} · Z2 {neos['retraso']:.4f} · "
                      f"{neos['segundos_con_cola']:.0f} s con cola")
                filas.append({"caso": caso, "solver": f"{solver} (NEOS)", **neos})
        if hasattr(modelo, "epsilon"):
            modelo.del_component(modelo.epsilon)

    if filas:
        tabla = pd.DataFrame(filas)
        tabla.to_csv(CARPETA / "comparacion_solvers.csv", index=False)
        print("\n" + tabla.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    if not args.enviar:
        print("\nNo se envió nada. Revise los .nl y, para enviar: python scripts/comparar_solvers_neos.py --enviar")
    else:
        print("\nLos tiempos de NEOS incluyen la subida y la cola del servidor: son una cota superior del tiempo de solución")
