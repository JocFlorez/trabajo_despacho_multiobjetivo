import json

from despacho_multiobjetivo import datos_empresa

LLAVES = {"archivo_despachos", "guia_region", "guia_ciudad", "cumplimiento_envia", "histograma_dias_tarde",
          "valor_declarado_cotizacion"}


def test_sin_archivo_privado_usa_el_ejemplo_y_avisa(tmp_path, capsys):
    datos = datos_empresa.load_company_data(tmp_path / "no_existe.json")
    assert LLAVES <= set(datos)
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_con_archivo_privado_lo_usa(tmp_path):
    privado = tmp_path / "datos_empresa.json"
    privado.write_text(json.dumps({"archivo_despachos": "otro.xlsx"}), encoding="utf-8")
    assert datos_empresa.load_company_data(privado)["archivo_despachos"] == "otro.xlsx"


def test_el_ejemplo_tiene_el_mismo_formato_que_usa_el_modelo():
    with open(datos_empresa.RUTA_EJEMPLO, encoding="utf-8") as archivo:
        ejemplo = json.load(archivo)
    assert LLAVES <= set(ejemplo)
    assert abs(sum(ejemplo["histograma_dias_tarde"].values()) - 1) < 1e-9
    assert all({"sede", "region", "valor"} <= set(fila) for fila in ejemplo["cumplimiento_envia"])
