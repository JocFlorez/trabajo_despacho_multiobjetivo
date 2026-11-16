import pulp

from despacho_multiobjetivo.config import GAP_REL_MENSUAL

def solve(problema: pulp.LpProblem) -> pulp.LpProblem:
    # El modelo mensual es la referencia y resuelve en segundos: se exige el óptimo exacto (GAP_REL_MENSUAL = 0)
    problema.solve(pulp.HiGHS(msg=True, gapRel=GAP_REL_MENSUAL))
    estado = pulp.LpStatus[problema.status]
    if estado != "Optimal":
        raise ValueError(f"El modelo no encontró una solución óptima (estado: {estado})")
    return problema
