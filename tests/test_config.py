"""Rutas del proyecto y escenario base."""
from despacho_multiobjetivo import config


def test_la_raiz_es_la_carpeta_del_proyecto():
    assert (config.RAIZ / "pyproject.toml").exists()
    assert (config.RAIZ / "src" / "despacho_multiobjetivo").is_dir()


def test_las_carpetas_de_datos_estan_dentro_del_proyecto():
    for carpeta in (config.RAW, config.PROCESSED, config.EXTERNAL):
        assert carpeta.parent == config.RAIZ / "data"


def test_el_prefijo_de_la_frontera_lleva_sede_mes_y_espera():
    assert config.PREFIJO_FRONTERA.endswith(f"frontera_{config.SEDE}_{config.MES_INICIO[:7]}_E{config.ESPERA_MAXIMA}")
