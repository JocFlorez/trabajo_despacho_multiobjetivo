import pandas as pd
import pulp
import pytest

from despacho_multiobjetivo.reportes import report


def _problema_con_costo(costo_optimo):
    """Problema ya 'resuelto' cuyo objetivo vale costo_optimo, sin llamar al solver."""
    problema = pulp.LpProblem("reporte", pulp.LpMinimize)
    v = pulp.LpVariable("v", lowBound=0, upBound=0)
    problema += v + costo_optimo
    v.varValue = 0
    return problema


def _resultado():
    return pd.DataFrame({
        "id_pedido": ["A_0", "B_0", "C_0", "D_0"],
        "flete_sin_consolidar": [1000.0, 500.0, 300.0, 200.0],
        "region_logistica": ["ANTIOQUIA_CHOCO", "ANTIOQUIA_CHOCO", "ANTIOQUIA_CHOCO", "ORIENTE"],
        "sin_ruta_triangulacion": [False, False, True, False],   # C cayó en Big-M
        "consolidado": [1.0, 0.5, 0.0, 0.0],                     # B se consolidó a medias
    })


Y = {("Turbo", "ANTIOQUIA_CHOCO"): None}   # Oriente no tiene capacidad


def _como_dict(resumen):
    return dict(zip(resumen["concepto"], resumen["valor"]))


def test_resumen_historico_calcula_ahorro_y_porcentajes():
    r = _como_dict(report.build_summary(_resultado(), _problema_con_costo(1400), Y, "historico"))
    assert r["Pagado hoy ($)"] == 2000
    assert r["Costo total óptimo ($)"] == 1400
    assert r["Ahorro ($)"] == 600
    assert r["% Ahorro sobre pagado hoy"] == pytest.approx(30)
    # elegibles: A y B (C está en Big-M, D no tiene capacidad)
    assert r["Pagado hoy elegible ($)"] == 1500
    assert r["% Cobertura (elegible / pagado hoy)"] == pytest.approx(75)
    assert r["% Ahorro sobre elegible"] == pytest.approx(40)
    # consolidado ponderado: 1000 * 1 + 500 * 0,5
    assert r["Pagado hoy consolidado ($)"] == 1250
    assert r["% Ahorro sobre consolidado"] == pytest.approx(48)


def test_resumen_operacion_cambia_las_etiquetas_de_la_base():
    resumen = report.build_summary(_resultado(), _problema_con_costo(1400), Y, "operacion")
    conceptos = resumen["concepto"].tolist()
    assert conceptos[0] == "Costo directo cotizado ($)"
    assert "% Ahorro sobre costo directo cotizado" in conceptos
    assert not any("Pagado" in c for c in conceptos)


def test_resumen_sin_nada_consolidado_no_divide_por_cero():
    resultado = _resultado().assign(consolidado=0.0)
    r = _como_dict(report.build_summary(resultado, _problema_con_costo(2000), Y, "historico"))
    assert r["Ahorro ($)"] == 0
    assert r["% Ahorro sobre consolidado"] == 0


def _vacio():
    return pd.DataFrame()


def test_exportar_reporte_sin_revision_manual_no_crea_esa_hoja(tmp_path):
    ruta = tmp_path / "reporte.xlsx"
    resumen = report.build_summary(_resultado(), _problema_con_costo(1400), Y, "historico")
    report.export_report(resumen, _resultado(), pd.DataFrame({"vehiculo": ["Turbo"]}), _vacio(), ruta)
    assert pd.ExcelFile(ruta).sheet_names == ["Resumen", "Pedidos", "Vehiculos"]


def test_exportar_reporte_con_revision_manual_agrega_la_hoja(tmp_path):
    ruta = tmp_path / "reporte.xlsx"
    resumen = report.build_summary(_resultado(), _problema_con_costo(1400), Y, "operacion")
    revision = pd.DataFrame({"id_pedido": ["Z_0"], "ciudad_destino": ["PUNTA BRAVA"]})
    report.export_report(resumen, _resultado(), pd.DataFrame({"vehiculo": ["Turbo"]}), revision, ruta)
    hojas = pd.ExcelFile(ruta).sheet_names
    assert hojas == ["Resumen", "Pedidos", "Vehiculos", "Revision manual"]
    assert pd.read_excel(ruta, "Revision manual")["id_pedido"].tolist() == ["Z_0"]


def test_extraer_resultados_pedidos_agrega_consolidado_sin_modificar_la_entrada():
    x = {"A_0": pulp.LpVariable("xa"), "B_0": pulp.LpVariable("xb")}
    x["A_0"].varValue, x["B_0"].varValue = 1, 0.5
    pedidos = pd.DataFrame({"id_pedido": ["B_0", "A_0"]})
    resultado = report.extract_order_results(pedidos, x)
    assert resultado["consolidado"].tolist() == [0.5, 1]
    assert "consolidado" not in pedidos.columns


def test_extraer_resultados_vehiculos_una_fila_por_vehiculo_y_region():
    y = {("Mula", "COSTA_CARIBE"): pulp.LpVariable("y1"), ("Turbo", "CENTRO_SUR"): pulp.LpVariable("y2")}
    y[("Mula", "COSTA_CARIBE")].varValue, y[("Turbo", "CENTRO_SUR")].varValue = 2, 0
    resultado = report.extract_vehicle_results(y)
    assert resultado.to_dict("records") == [
        {"vehiculo": "Mula", "region": "COSTA_CARIBE", "cantidad": 2},
        {"vehiculo": "Turbo", "region": "CENTRO_SUR", "cantidad": 0},
    ]
