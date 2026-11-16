"""Selección de despachos de ENVIA y marca de a tiempo del script de medición (sin leer el archivo real)."""
from datetime import date

import pandas as pd

from despacho_multiobjetivo.datos import data_processing
from scripts import medir_cumplimiento


def _despachos():
    return pd.DataFrame({
        "transportadora": ["COLVANES", "colvanes ", "SERVIENTREGA", "COLVANES", "COLVANES"],
        # Captura mar 13 con promesa 3 → límite vie 16 (sábado no hábil)
        "fecha_captura": ["2026-01-13", "2026-01-13", "2026-01-13", None, "2026-01-13"],
        "fecha_entrega": ["2026-01-16", "2026-01-19", "2026-01-16", "2026-01-16", "2026-01-10"],
        "dias_promesa_entrega": [3, 3, 3, 3, 3],
    })


def test_solo_envia_con_fechas_completas_y_coherentes(monkeypatch):
    monkeypatch.setattr(data_processing, "process_shipments", _despachos)
    envia = medir_cumplimiento.envia_with_dates()
    # Fila 1 y 2: COLVANES (con mayúsculas o espacios) es ENVIA; fuera Servientrega, la que no tiene captura
    # y la que figura entregada antes de capturada
    assert envia.index.tolist() == [0, 1]


def test_marca_a_tiempo_con_la_promesa_estricta(monkeypatch):
    monkeypatch.setattr(data_processing, "process_shipments", _despachos)
    envia = medir_cumplimiento.mark_on_time(medir_cumplimiento.envia_with_dates())
    assert envia["fecha_limite"].tolist() == [date(2026, 1, 16), date(2026, 1, 16)]
    assert envia["a_tiempo"].tolist() == [True, False]   # el viernes 16 cumple; el lunes 19 llega tarde
