"""Salidas de las metaheurísticas: curva de convergencia."""
import pandas as pd

from despacho_multiobjetivo.reportes.metaheuristicas import plot_convergence


def _historial():
    # 4 iteraciones: la 2 se pasa del límite; el mejor aparece en la 1 y mejora en la 3
    return pd.DataFrame({"iteracion": [1, 2, 3, 4], "z1": [250e6, 228e6, 235e6, 240e6],
                         "z2": [1.0, 1.02, 1.01, 0.99], "cumple": [True, False, True, True],
                         "alfa": [2.2e8, 2.4e8, 2.2e8, 2.0e8], "z1_mejor": [250e6, 250e6, 235e6, 235e6],
                         "segundos": [4.0, 8.0, 12.0, 16.0]})


def test_grafica_de_convergencia(tmp_path):
    ruta = tmp_path / "convergencia.png"
    plot_convergence(_historial(), 1.0, 230e6, "Convergencia de prueba", str(ruta))
    assert ruta.exists() and ruta.stat().st_size > 0


def test_sin_referencia_del_milp_ni_plan_que_cumpla(tmp_path):
    # Corrida que nunca encontró un plan que cumpla (z1_mejor vacío) y sin MILP para comparar
    historial = _historial().assign(cumple=False, z1_mejor=None)
    ruta = tmp_path / "convergencia.png"
    plot_convergence(historial, 1.014, None, "Sin plan que cumpla", str(ruta))
    assert ruta.exists() and ruta.stat().st_size > 0


def test_lee_el_historial_guardado_en_csv(tmp_path):
    # El script guarda el historial en CSV y la gráfica puede hacerse después, leyéndolo de nuevo
    archivo = tmp_path / "historial.csv"
    _historial().to_csv(archivo, index=False)
    ruta = tmp_path / "convergencia.png"
    plot_convergence(pd.read_csv(archivo), 1.0, 230e6, "Desde CSV", str(ruta))
    assert ruta.exists() and ruta.stat().st_size > 0
