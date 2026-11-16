"""Tablero de avance de las corridas: qué se lee de los archivos de resultados."""
import os
import time

from scripts.ver_avance import format_minutes, pending_stages, stage_progress, time_jump


def _crear(carpeta, nombre, antiguedad_min=0):
    ruta = carpeta / nombre
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("x")
    marca = time.time() - antiguedad_min * 60
    os.utime(ruta, (marca, marca))
    return ruta


def _etapa(**cambios):
    return {"nombre": "Prueba", "archivos": ["a.csv", "b.csv", "c.csv"], "minutos": 10, **cambios}


# ---------------------------------------------------------------- avance de una etapa

def test_sin_archivos_no_hay_avance(tmp_path):
    avance = stage_progress(_etapa(), tmp_path)
    assert (avance["hechos"], avance["total"]) == (0, 3)
    assert avance["minutos_restantes"] == 30


def test_cuenta_los_archivos_que_ya_existen(tmp_path):
    _crear(tmp_path, "a.csv")
    avance = stage_progress(_etapa(), tmp_path)
    assert avance["hechos"] == 1
    assert avance["minutos_restantes"] == 20


def test_descuenta_lo_que_lleva_la_unidad_en_curso(tmp_path):
    # La última terminó hace 4 minutos y cada una tarda 10: a la que está corriendo le quedan 6
    _crear(tmp_path, "a.csv", antiguedad_min=4)
    avance = stage_progress(_etapa(), tmp_path)
    assert 15 < avance["minutos_restantes"] < 17


def test_etapa_completa_no_tiene_tiempo_restante(tmp_path):
    for nombre in ("a.csv", "b.csv", "c.csv"):
        _crear(tmp_path, nombre)
    avance = stage_progress(_etapa(), tmp_path)
    assert avance["completa"]
    assert avance["minutos_restantes"] == 0


def test_una_unidad_pasada_de_tiempo_se_marca_como_demorada(tmp_path):
    _crear(tmp_path, "a.csv", antiguedad_min=25)
    assert stage_progress(_etapa(), tmp_path)["demorada"]
    _crear(tmp_path, "a.csv", antiguedad_min=5)
    assert not stage_progress(_etapa(), tmp_path)["demorada"]


def test_archivo_viejo_no_cuenta_si_falta_el_paso_previo(tmp_path):
    # La semilla 0 del genético ya tiene un archivo viejo: solo cuenta cuando ese viejo se movió a otra carpeta
    _crear(tmp_path, "a.csv")
    etapa = _etapa(archivos=["a.csv"], requiere="aparte/a.csv")
    assert stage_progress(etapa, tmp_path)["hechos"] == 0
    _crear(tmp_path, "aparte/a.csv")
    assert stage_progress(etapa, tmp_path)["hechos"] == 1


def test_la_primera_unidad_no_se_marca_demorada_sin_referencia(tmp_path):
    assert not stage_progress(_etapa(), tmp_path)["demorada"]


# ---------------------------------------------------------------- suspensiones del computador

def test_detecta_un_salto_de_tiempo_en_el_historial(tmp_path):
    ruta = tmp_path / "h.csv"
    ruta.write_text("iteracion,segundos\n1,4.0\n2,8.1\n3,500.0\n")
    assert time_jump(ruta)


def test_un_historial_normal_no_tiene_salto(tmp_path):
    ruta = tmp_path / "h.csv"
    ruta.write_text("iteracion,segundos\n1,4.0\n2,8.1\n3,12.2\n")
    assert not time_jump(ruta)


def test_archivo_sin_columna_de_segundos_no_es_un_salto(tmp_path):
    ruta = tmp_path / "h.csv"
    ruta.write_text("punto,brecha\n100%,0.03\n")
    assert not time_jump(ruta)


# ---------------------------------------------------------------- lectura en pantalla

def test_formato_de_minutos():
    assert format_minutes(0) == "0 min"
    assert format_minutes(45.4) == "45 min"
    assert format_minutes(125) == "2 h 05 min"


def test_las_etapas_de_esta_noche_estan_en_orden():
    etapas = pending_stages("0111_2026-01_E3")
    assert [len(etapa["archivos"]) for etapa in etapas] == [20, 1, 4]
    assert etapas[0]["archivos"][0] == "tabu_tenure5_0111_2026-01_E3_p86_s0_historial.csv"
    assert etapas[1]["requiere"].startswith("comparacion_interferida/")
