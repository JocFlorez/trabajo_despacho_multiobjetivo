import pandas as pd
import pulp

BASE_POR_MODO = {"historico": "Pagado hoy", "operacion": "Costo directo cotizado"}

def extract_order_results(pedidos: pd.DataFrame, x: dict) -> pd.DataFrame:
    pedidos = pedidos.copy()
    pedidos["consolidado"] = [x[id_pedido].varValue for id_pedido in pedidos["id_pedido"]]
    return pedidos

def extract_vehicle_results(y: dict) -> pd.DataFrame:
    filas = []
    for (v, r), variable in y.items():
        filas.append({"vehiculo": v, "region": r, "cantidad": variable.varValue})
    return pd.DataFrame(filas)

def build_summary(pedidos_resultado: pd.DataFrame, problema: pulp.LpProblem, y: dict, modo: str) -> pd.DataFrame:
    base = BASE_POR_MODO[modo]
    costo_base = pedidos_resultado["flete_sin_consolidar"].sum()
    costo_total = pulp.value(problema.objective)
    ahorro = costo_base - costo_total
    pct_ahorro = ahorro / costo_base * 100

    regiones_con_capacidad = {region for (_vehiculo, region) in y}
    elegibles = pedidos_resultado[pedidos_resultado["region_logistica"].isin(regiones_con_capacidad)&~pedidos_resultado["sin_ruta_triangulacion"]]

    costo_base_elegible = elegibles["flete_sin_consolidar"].sum()
    pct_cobertura = costo_base_elegible / costo_base * 100
    pct_ahorro_elegible = (ahorro / costo_base_elegible * 100) if costo_base_elegible else 0

    # Ponderado por consolidado: los traslados pueden consolidarse parcialmente (x continuo)
    costo_base_consolidado = (pedidos_resultado["consolidado"] * pedidos_resultado["flete_sin_consolidar"]).sum()
    pct_ahorro_consolidado = (ahorro / costo_base_consolidado * 100) if costo_base_consolidado else 0

    return pd.DataFrame({
        "concepto": [f"{base} ($)", "Costo total óptimo ($)", "Ahorro ($)", f"% Ahorro sobre {base.lower()}",
                     f"{base} elegible ($)", f"% Cobertura (elegible / {base.lower()})", "% Ahorro sobre elegible",
                     f"{base} consolidado ($)", "% Ahorro sobre consolidado"],
        "valor": [costo_base, costo_total, ahorro, pct_ahorro,
                  costo_base_elegible, pct_cobertura, pct_ahorro_elegible,
                  costo_base_consolidado, pct_ahorro_consolidado],
    })

def export_report(resumen: pd.DataFrame, pedidos_resultado: pd.DataFrame, vehiculos_resultado: pd.DataFrame,
                  revision_manual: pd.DataFrame, ruta: str) -> None:
    with pd.ExcelWriter(ruta) as writer:
        resumen.to_excel(writer, sheet_name="Resumen", index=False)
        pedidos_resultado.to_excel(writer, sheet_name="Pedidos", index=False)
        vehiculos_resultado.to_excel(writer, sheet_name="Vehiculos", index=False)
        if not revision_manual.empty:
            revision_manual.to_excel(writer, sheet_name="Revision manual", index=False)