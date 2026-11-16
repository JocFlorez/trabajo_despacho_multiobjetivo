import argparse
import sys

import pytest

from despacho_multiobjetivo.config import PROCESSED
from scripts import main


def test_nombre_del_reporte_lleva_sede_mes_y_modo():
    assert main.default_report_path("0111", "2026-01-01", "2026-01-31", "historico") == \
        str(PROCESSED / "reporte_0111_2026-01_historico.xlsx")


def test_nombre_del_reporte_con_varios_meses():
    assert main.default_report_path("0211", "2026-01-01", "2026-03-31", "operacion") == \
        str(PROCESSED / "reporte_0211_2026-01_a_2026-03_operacion.xlsx")


def test_el_modo_por_defecto_es_operacion(monkeypatch):
    # En uso real se corre en modo operación; las pruebas de la tesis pasan --modo historico
    monkeypatch.setattr(sys, "argv", ["main.py", "--sede", "0111", "--mes-inicio", "2026-01-01", "--mes-fin", "2026-01-31"])
    argumentos = main.parse_arguments()
    assert (argumentos.sede, argumentos.modo, argumentos.ruta_reporte) == ("0111", "operacion", None)


def test_fecha_invalida_muestra_la_fecha_entre_comillas():
    with pytest.raises(argparse.ArgumentTypeError, match="Fecha inválida '2026-13-01'"):
        main.valid_date("2026-13-01")


def test_solo_acepta_las_sedes_del_estudio(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main.py", "--sede", "9999", "--mes-inicio", "2026-01-01", "--mes-fin", "2026-01-31"])
    with pytest.raises(SystemExit):
        main.parse_arguments()
