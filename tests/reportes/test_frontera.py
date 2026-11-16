"""Métricas de cada plan de la frontera y gráfica del espacio de objetivos."""
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import pulp
import pytest

from despacho_multiobjetivo.reportes.frontera import (plan_metrics, plot_frontier, plot_frontier_real_terms,
                                                      real_terms_table)

LUNES, MARTES = date(2026, 1, 5), date(2026, 1, 6)


def _variable(nombre, valor):
    variable = pulp.LpVariable(nombre)
    variable.varValue = valor
    return variable


def test_metricas_del_plan_en_lenguaje_natural():
    # 2 pedidos: A va directo; B sale en camión el martes, después de esperar 1 día
    pedidos = pd.DataFrame({"id_pedido": ["A", "B"]})
    opciones = pd.DataFrame({"id_pedido": ["B", "B"], "dia_salida": [LUNES, MARTES], "espera": [0.0, 1.0]})
    u = {"A": _variable("uA", 1), "B": _variable("uB", 0)}
    x = {("B", LUNES): _variable("xB1", 0), ("B", MARTES): _variable("xB2", 1)}
    y = {("Turbo", "R", MARTES): _variable("y", 1)}
    indice = pd.MultiIndex.from_tuples([("B", LUNES), ("B", MARTES)], names=["id_pedido", "dia_salida"])
    metricas = plan_metrics(pedidos, opciones, x, u, y,
                            prob_directo=pd.Series({"A": 0.8, "B": 0.6}),
                            prob_opciones=pd.Series([0.85, 0.0], index=indice),
                            dias_directo=pd.Series({"A": 0.3, "B": 0.5}),
                            dias_opciones=pd.Series([0.2, 1.4], index=indice))
    assert metricas == pytest.approx({
        "nivel_servicio": (0.8 + 0.0) / 2,          # A a tiempo 80 %; B sale tarde por plan
        "retraso_promedio_dias": (0.3 + 1.4) / 2,   # días hábiles de retraso por pedido, sin castigo
        "pedidos_en_camion": 1,
        "espera_promedio": 1.0,
        "camiones": 1,
    })


def test_grafica_de_la_frontera(tmp_path):
    matriz = pd.DataFrame({"costo": [100, 200, 100, 200], "retraso": [2.0, 1.0, 1.0, 2.0]},
                          index=["Optimizar Z1 (costo)", "Optimizar Z2 (retraso)", "Ideal", "Nadir"])
    puntos = pd.DataFrame({"costo": [200, 150, 100], "retraso": [1.0, 1.5, 2.0]})
    ruta = tmp_path / "frontera.png"
    plot_frontier(matriz, puntos, {"costo": 180, "retraso": 1.6}, "Frontera de prueba", str(ruta))
    assert ruta.exists() and ruta.stat().st_size > 0


def test_el_eje_de_la_frontera_usa_el_nombre_de_z2(tmp_path, monkeypatch):
    matriz = pd.DataFrame({"costo": [100, 200, 100, 200], "retraso": [2.0, 1.0, 1.0, 2.0]},
                          index=["Optimizar Z1 (costo)", "Optimizar Z2 (retraso)", "Ideal", "Nadir"])
    puntos = pd.DataFrame({"costo": [200, 100], "retraso": [1.0, 2.0]})
    figuras = []
    monkeypatch.setattr(plt, "close", figuras.append)   # conserva la figura para leerle el eje
    plot_frontier(matriz, puntos, {"costo": 180, "retraso": 1.6}, "Frontera de prueba", str(tmp_path / "f.png"))
    assert figuras[0].axes[0].get_xlabel() == "Días de retraso ponderados por pedido"


# ---------------------------------------------------------------- frontera en días reales y % a tiempo

def _puntos_reales():
    return pd.DataFrame({"costo": [100.0, 300.0, 200.0], "ahorro": [50.0, -150.0, -50.0], "retraso": [2.0, 1.0, 1.5],
                         "nivel_servicio": [0.7, 0.9, 0.8], "retraso_promedio_dias": [0.4, 0.2, 0.3],
                         "camiones": [1, 3, 2]})


def test_tabla_en_dias_reales_ordenada_por_servicio():
    tabla = real_terms_table(_puntos_reales(), peso_total_kg=100.0)
    assert list(tabla["retraso"]) == [1.0, 1.5, 2.0]
    assert list(tabla["a_tiempo_pct"]) == pytest.approx([90.0, 80.0, 70.0])
    assert list(tabla["retraso_promedio_dias"]) == [0.2, 0.3, 0.4]


def test_flete_por_kilo_es_el_costo_entre_el_peso_total():
    tabla = real_terms_table(_puntos_reales(), peso_total_kg=100.0)
    assert list(tabla["flete_por_kg"]) == pytest.approx([3.0, 2.0, 1.0])


def test_grafica_en_dias_reales_tiene_los_dos_ejes_de_servicio(tmp_path, monkeypatch):
    figuras = []
    monkeypatch.setattr(plt, "close", figuras.append)
    ruta = tmp_path / "reales.png"
    plot_frontier_real_terms(_puntos_reales(), {"costo": 180, "retraso_promedio_dias": 0.25, "nivel_servicio": 0.85},
                             "Frontera de prueba", str(ruta))
    assert ruta.exists() and ruta.stat().st_size > 0
    assert [eje.get_xlabel() for eje in figuras[0].axes] == ["Días hábiles reales de retraso por pedido",
                                                             "Pedidos a tiempo (%)"]
