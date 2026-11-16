"""La carga de datos encadena los pasos de data_processing en el orden correcto (sin leer archivos reales)."""
import pandas as pd

from despacho_multiobjetivo.datos import cargar
from despacho_multiobjetivo.datos import data_processing as dp


def _simular_data_processing(monkeypatch, pasos):
    def registrar(nombre, resultado):
        def paso(*args):
            pasos.append((nombre, args[1:] if nombre != "get_shipments" else args))
            return resultado(*args)
        return paso

    pedidos = pd.DataFrame({"guia": ["A", "A", "B"]})
    monkeypatch.setattr(dp, "get_shipments", registrar("get_shipments", lambda *a: pedidos))
    monkeypatch.setattr(dp, "load_regional_hubs", registrar("nodos", lambda: {"ANTIOQUIA_CHOCO": "MEDELLIN"}))
    monkeypatch.setattr(dp, "rate_shipments", registrar("rate_shipments", lambda p, n: p.assign(triangulacion=1)))
    monkeypatch.setattr(dp, "rate_ltl_direct", registrar("rate_ltl_direct", lambda p: p.assign(directo=1)))
    monkeypatch.setattr(dp, "prepare_unconsolidated_cost",
                        registrar("costo", lambda p, modo: (p.assign(modo=modo), p.iloc[0:0])))
    monkeypatch.setattr(dp, "load_vehicle_capacities", lambda: {"Turbo": 4500})
    monkeypatch.setattr(dp, "load_ftl_rates", lambda: {("Turbo", "0111", "MEDELLIN"): 1})
    monkeypatch.setattr(dp, "load_ftl_transit_times", lambda: {("0111", "MEDELLIN"): 1})


def test_preparar_pedidos_cotiza_y_devuelve_la_revision_manual(monkeypatch):
    pasos = []
    _simular_data_processing(monkeypatch, pasos)
    pedidos, revision, nodos = cargar.prepare_orders("0111", "2026-01-01", "2026-01-31", "operacion")
    assert [nombre for nombre, _ in pasos] == ["get_shipments", "nodos", "rate_shipments", "rate_ltl_direct", "costo"]
    assert pasos[0][1] == ("0111", "2026-01-01", "2026-01-31")
    assert pedidos["id_pedido"].tolist() == ["A_0", "A_1", "B_0"]   # el identificador único se agrega antes de cotizar
    assert set(pedidos["modo"]) == {"operacion"}
    assert revision.empty and nodos == {"ANTIOQUIA_CHOCO": "MEDELLIN"}


def test_cargar_datos_agrega_los_parametros_de_camiones_y_tiempos(monkeypatch):
    _simular_data_processing(monkeypatch, [])
    pedidos, nodos, capacidades, tarifas, tiempos = cargar.load_data("0111", "2026-01-01", "2026-01-31", "historico")
    assert len(pedidos) == 3 and set(pedidos["modo"]) == {"historico"}
    assert capacidades == {"Turbo": 4500} and tiempos == {("0111", "MEDELLIN"): 1}
