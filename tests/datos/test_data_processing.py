from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from despacho_multiobjetivo import parametros
from despacho_multiobjetivo.datos import data_processing as dp


# ---------------------------------------------------------------- utilidades

def test_a_float_convierte_decimal_y_numeros():
    assert dp._to_float(Decimal("13600")) == 13600.0
    assert dp._to_float(3) == 3.0


def test_a_float_devuelve_none_sin_ruta_o_vacio():
    assert dp._to_float("NO HAY RUTA") is None
    assert dp._to_float(None) is None


# ---------------------------------------------------------- limpieza de tipos

def test_limpiar_tipos_columnas_rellena_sede_y_lee_fecha_dia_primero():
    df = pd.DataFrame({
        "peso": ["120"], "valor_flete": ["5000"], "unidades": ["3"],
        "sede": ["111"], "sede_flete": ["106"],
        "mes": ["2026-03-02"], "fecha_captura": ["4/03/2026"], "fecha_entrega": ["6/03/2026"],
    })
    limpio = dp.clean_column_types(df)
    assert limpio.loc[0, "sede"] == "0111"
    assert limpio.loc[0, "sede_flete"] == "0106"
    assert limpio.loc[0, "peso"] == 120
    # 4/03/2026 es 4 de marzo, no 3 de abril
    assert limpio.loc[0, "fecha_captura"] == pd.Timestamp("2026-03-04")
    assert limpio.loc[0, "fecha_entrega"] == pd.Timestamp("2026-03-06")


def test_filtrar_cedis_conserva_solo_0111_y_0211():
    df = pd.DataFrame({"sede": ["0111", "0211", "0106", "0212"]})
    assert dp.filter_cedis(df)["sede"].tolist() == ["0111", "0211"]


def test_excluir_am_quita_solo_el_area_metropolitana_de_la_propia_sede():
    df = pd.DataFrame({
        "sede": ["0111", "0111", "0211", "0211"],
        "ciudad_destino": ["SOACHA", "BARRANQUILLA", "SOLEDAD", "BOGOTA"],
    })
    resultado = dp.exclude_metro_area(df)
    assert list(zip(resultado["sede"], resultado["ciudad_destino"])) == [
        ("0111", "BARRANQUILLA"),
        ("0211", "BOGOTA"),
    ]


def test_excluir_entradas_al_cedi_quita_recolecciones_y_devoluciones():
    df = pd.DataFrame({"tipo_despacho": ["CLIENTES", "PROVEEDORES", "TRASLADOS", "DEVOLUCION",
                                         "TRASLADOS AL GASTO", "OTROS"]})
    assert dp.exclude_inbound_to_cedi(df)["tipo_despacho"].tolist() == \
        ["CLIENTES", "TRASLADOS", "TRASLADOS AL GASTO", "OTROS"]


def test_search_department_usa_centro_poblado_si_no_es_municipio():
    divipola = pd.DataFrame({
        "Municipio": ["MEDELLIN", "IBAGUE"],
        "CentroPoblado": ["SANTA ELENA", "BUENOS AIRES"],
        "Departamento": ["ANTIOQUIA", "TOLIMA"],
    })
    df = pd.DataFrame({"ciudad_destino": ["MEDELLIN", "BUENOS AIRES", "NO EXISTE"]})
    resultado = dp.search_department(df, divipola)
    assert resultado["departamento"].tolist()[:2] == ["ANTIOQUIA", "TOLIMA"]
    assert pd.isna(resultado.loc[2, "departamento"])


# ------------------------------------------------- limpieza del histórico

def test_leer_fechas_con_formatos_mezclados():
    # Los cuatro formatos que trae el histórico dan la misma fecha, sin hora
    columna = pd.Series(["2025-05-28 00:00:00", "2025-05-28", "28/05/2025 07:43:01 p.m.", "28/05/2025"])
    fechas = dp.parse_dates(columna)
    assert (fechas == pd.Timestamp("2025-05-28")).all()


def test_leer_fechas_con_barras_es_dia_mes():
    assert dp.parse_dates(pd.Series(["4/03/2026"]))[0] == pd.Timestamp("2026-03-04")


def test_leer_fechas_mes_dia_solo_si_dia_mes_es_imposible():
    # 6/13/2024 no puede ser día 6 del mes 13: es 13 de junio
    assert dp.parse_dates(pd.Series(["6/13/2024"]))[0] == pd.Timestamp("2024-06-13")


def test_leer_fechas_basura_o_vacia_queda_vacia():
    fechas = dp.parse_dates(pd.Series(["99/999999", "10:22:33", None]))
    assert fechas.isna().all()


def test_leer_fechas_que_ya_son_fecha_no_cambian():
    columna = pd.Series(pd.to_datetime(["2026-01-05", "2026-01-06"]))
    assert dp.parse_dates(columna).tolist() == columna.tolist()


def test_limpiar_tipos_columnas_convierte_dias_promesa_a_numero():
    df = pd.DataFrame({
        "peso": ["120"], "valor_flete": ["5000"], "unidades": ["3"], "sede": ["111"], "sede_flete": ["106"],
        "mes": ["2025-05-01"], "fecha_captura": ["2025-05-28"], "fecha_entrega": ["30/05/2025"],
        "dias_promesa_entrega": ["3"],
    })
    limpio = dp.clean_column_types(df)
    assert limpio.loc[0, "dias_promesa_entrega"] == 3
    assert limpio.loc[0, "fecha_entrega"] == pd.Timestamp("2025-05-30")


def test_normalizar_tipo_quita_espacios_y_unifica():
    df = pd.DataFrame({"tipo_despacho": ["TRASLADOS   ", " CLIENTES", "TRASLADO AL GASTO", "CX", "COMEX", "VARIOS"]})
    assert dp.normalize_dispatch_type(df)["tipo_despacho"].tolist() == \
        ["TRASLADOS", "CLIENTES", "TRASLADOS AL GASTO", "OTROS", "OTROS", "OTROS"]


def test_excluir_entradas_al_cedi_quita_devoluciones_en_plural():
    df = pd.DataFrame({"tipo_despacho": ["CLIENTES", "DEVOLUCIONES"]})
    assert dp.exclude_inbound_to_cedi(df)["tipo_despacho"].tolist() == ["CLIENTES"]


def test_quitar_duplicados_deja_una_y_respeta_guias_con_datos_distintos():
    df = pd.DataFrame({"guia": ["A", "A", "B", "B"], "peso": [10.0, 10.0, 5.0, 7.0]})
    resultado = dp.drop_exact_duplicates(df)
    assert list(zip(resultado["guia"], resultado["peso"])) == [("A", 10.0), ("B", 5.0), ("B", 7.0)]


def test_excluir_peso_no_positivo_deja_los_vacios():
    # El peso vacío no se quita aquí: lo quita get_shipments
    df = pd.DataFrame({"peso": [0.0, -5.0, 12.0, np.nan]})
    resultado = dp.exclude_non_positive_weight(df)
    assert resultado["peso"].tolist()[0] == 12.0
    assert len(resultado) == 2 and resultado["peso"].isna().sum() == 1


def test_borrar_fechas_imposibles_conserva_el_despacho():
    df = pd.DataFrame({
        "fecha_captura": pd.to_datetime(["2025-05-10", "2025-05-10"]),
        "fecha_entrega": pd.to_datetime(["2025-05-08", "2025-05-12"]),
        "dias_promesa_entrega": [-1.0, 2.0],
    })
    resultado = dp.blank_impossible_dates(df)
    assert len(resultado) == 2
    assert pd.isna(resultado.loc[0, "fecha_entrega"]) and pd.isna(resultado.loc[0, "dias_promesa_entrega"])
    assert resultado.loc[1, "fecha_entrega"] == pd.Timestamp("2025-05-12")
    assert resultado.loc[1, "dias_promesa_entrega"] == 2


def test_limpiar_despachos_aplica_todos_los_pasos():
    df = pd.DataFrame({
        "guia": ["A", "A", "B", "C"], "ciudad_destino": ["CALI"] * 4, "mes": pd.to_datetime(["2025-05-02"] * 4),
        "documento": [None] * 4, "factura": ["F1"] * 4, "valor_flete": [100.0] * 4,
        "tipo_despacho": ["CLIENTES ", "CLIENTES ", "CX", "CLIENTES"],
        "peso": [10.0, 10.0, 3.0, 0.0],
        "fecha_captura": pd.to_datetime(["2025-05-10"] * 4),
        "fecha_entrega": pd.to_datetime(["2025-05-12"] * 4),
        "dias_promesa_entrega": [2.0, 2.0, 2.0, 2.0],
    })
    resultado = dp.clean_shipments(df)
    assert resultado["tipo_despacho"].tolist() == ["CLIENTES", "OTROS"]


def test_excluir_am_reconoce_la_ciudad_con_espacios_o_minusculas():
    # En ene–jun 2024 la ciudad viene " MOSQUERA" y se colaba al modelo
    df = pd.DataFrame({"sede": ["0111", "0111", "0111"], "ciudad_destino": [" MOSQUERA", "mosquera ", "CALI"]})
    assert dp.exclude_metro_area(df)["ciudad_destino"].tolist() == ["CALI"]


def test_normalizar_ciudad_quita_comillas_y_corrige_nombres():
    ciudades = pd.Series(["'BARRANQUILLA", "b/quilla ", "SANTAMARTA", "MEDELLIN Y SABANETA", "IBAGUÉ", "0"])
    assert dp.normalize_city(ciudades).tolist() == \
        ["BARRANQUILLA", "BARRANQUILLA", "SANTA MARTA", "MEDELLIN", "IBAGUE", "0"]


def test_la_gran_via_queda_en_magdalena():
    divipola = pd.read_excel(dp.EXTERNAL / "divipola.xlsx")
    for columna in ("Municipio", "CentroPoblado", "Departamento"):
        divipola[columna] = dp.normalize_city(divipola[columna])
    df = pd.DataFrame({"ciudad_destino": dp.normalize_city(pd.Series(["LA GRAN VI"]))})
    resultado = dp.search_department(df, divipola)
    assert resultado.loc[0, "ciudad_destino"] == "LA GRAN VIA"
    assert resultado.loc[0, "departamento"] == "MAGDALENA"


def test_corregir_ciudad_segun_origen_san_jose_d():
    # "SAN JOSE D": desde Mosquera es San José del Guaviare; desde Sabaneta, San José del Nus
    df = pd.DataFrame({"sede": ["0111", "0106", "0212", "0211"], "ciudad_destino": ["SAN JOSE D"] * 4})
    assert dp.correct_city_by_origin(df)["ciudad_destino"].tolist() == \
        ["SAN JOSE DEL GUAVIARE", "SAN JOSE DEL NUS", "SAN JOSE DEL NUS", "SAN JOSE D"]


def test_mapear_region_resuelve_guamal_del_meta():
    # GUAMAL existe en Magdalena y en Meta: "GUAMAL - META" va a la región del Meta
    df = pd.DataFrame({"ciudad_destino": ["GUAMAL - META"], "departamento": [None], "guia": ["1"], "sede": ["0111"]})
    assert dp.map_region(df).loc[0, "region_logistica"] == "CENTRO"


def _registros(**columnas):
    # Filas con la misma guía, fecha, peso y destino; lo que cambia viene en columnas
    n = len(next(iter(columnas.values())))
    base = {"guia": ["G1"] * n, "fecha_captura": pd.to_datetime(["2026-05-29"] * n), "peso": [100.0] * n,
            "ciudad_destino": ["CALI"] * n, "mes": pd.to_datetime(["2026-05-02"] * n),
            "tipo_despacho": ["TRASLADOS"] * n, "valor_flete": [1000.0] * n,
            "documento": [None] * n, "factura": ["F1"] * n}
    base.update(columnas)
    return pd.DataFrame(base)


def test_registros_repetidos_con_documentos_distintos_se_dejan_todos():
    # Pedidos distintos en el mismo camión (la guía es la placa)
    df = _registros(documento=["AK-1", "AK-2", "AK-3"])
    assert len(dp.resolve_duplicate_records(df)) == 3


def test_registros_repetidos_con_cobro_partido_se_unen_sumando_el_flete():
    df = _registros(tipo_despacho=["TRASLADOS", "OTROS"], valor_flete=[2_000_000.0, 300_000.0])
    resultado = dp.resolve_duplicate_records(df)
    assert len(resultado) == 1
    assert resultado.iloc[0]["valor_flete"] == 2_300_000
    assert resultado.iloc[0]["tipo_despacho"] == "TRASLADOS"   # el tipo del cobro mayor
    assert resultado.iloc[0]["peso"] == 100                    # el peso una sola vez


def test_registros_repetidos_con_otro_tipo_y_mismo_flete_dejan_el_de_clientes():
    # Patrón A (decisión de Johan del 2026-11-16): se deja la fila de CLIENTES
    df = _registros(tipo_despacho=["TRASLADOS", "CLIENTES"])
    resultado = dp.resolve_duplicate_records(df)
    assert resultado["tipo_despacho"].tolist() == ["CLIENTES"]


def test_registros_repetidos_con_otro_tipo_y_mismo_flete_sin_clientes_dejan_uno():
    df = _registros(tipo_despacho=["TRASLADOS", "TRASLADOS AL GASTO"])
    assert len(dp.resolve_duplicate_records(df)) == 1


def test_registros_repetidos_facturados_dos_veces_dejan_el_que_trae_documento():
    df = _registros(documento=[None, "TR-1"], factura=["FE1", "FE2"],
                    mes=pd.to_datetime(["2024-04-02", "2024-05-02"]), valor_flete=[400_000.0, 380_000.0])
    resultado = dp.resolve_duplicate_records(df)
    assert resultado["documento"].tolist() == ["TR-1"]


def test_registros_repetidos_con_documentos_distintos_y_una_factura_no_se_unen():
    # Un camión con varios pedidos y una sola factura: no es un cobro partido
    df = _registros(documento=["AK-1", "TR-2"], tipo_despacho=["CLIENTES", "TRASLADOS AL GASTO"],
                    valor_flete=[2_000.0, 500.0])
    assert len(dp.resolve_duplicate_records(df)) == 2


def test_registros_sin_guia_o_sin_peso_no_se_agrupan():
    df = _registros(guia=[None, None], peso=[np.nan, np.nan], documento=[None, None], factura=["F1", None])
    assert len(dp.resolve_duplicate_records(df)) == 2


def test_registros_repetidos_sin_ninguna_factura_se_dejan_todos():
    # Flota propia: la guía es la placa, muchos pedidos sin factura; no hay cómo saber si son repetidos
    df = _registros(documento=["AJ-1", None], factura=[None, None], valor_flete=[0.0, 0.0])
    assert len(dp.resolve_duplicate_records(df)) == 2


def test_registros_repetidos_sin_factura_dejan_la_copia_con_factura():
    df = _registros(documento=["NO DISPONIBLE", None], factura=["FE1", None])
    resultado = dp.resolve_duplicate_records(df)
    assert resultado["factura"].tolist() == ["FE1"]


# -------------------------------------------------- correcciones de datos

def test_completar_ciudad_destino_solo_rellena_vacios_de_guias_conocidas(monkeypatch):
    # Guías inventadas: las reales están en data/external/datos_empresa.json (no se publican)
    monkeypatch.setattr(dp, "GUIA_CIUDAD_OVERRIDE", {"G-1": "BARRANQUILLA", "G-2": "SABANETA", "G-3": "CARTAGENA"})
    df = pd.DataFrame({
        "guia": ["G-1", "G-2", "99999", "G-3"],
        "ciudad_destino": [None, None, None, "CALI"],
    })
    resultado = dp.fill_destination_city(df)
    assert resultado.loc[0, "ciudad_destino"] == "BARRANQUILLA"
    assert resultado.loc[1, "ciudad_destino"] == "SABANETA"
    assert pd.isna(resultado.loc[2, "ciudad_destino"])
    assert resultado.loc[3, "ciudad_destino"] == "CALI"  # un valor existente no se sobrescribe


def test_completar_unidades_una_por_tonelada_minimo_una():
    df = pd.DataFrame({"peso": [4190.0, 963.0, 7.0, 150.0, 500.0],
                       "unidades": [np.nan, np.nan, np.nan, np.nan, 12.0]})
    resultado = dp.fill_units(df)
    assert resultado["unidades"].tolist() == [4, 1, 1, 1, 12]


def test_completar_unidades_tambien_corrige_las_unidades_en_cero():
    df = pd.DataFrame({"peso": [3000.0, 200.0, 50.0], "unidades": [0.0, 0.0, 3.0]})
    assert dp.fill_units(df)["unidades"].tolist() == [3, 1, 3]


def test_etiquetar_tipo_transportadora_no_excluye_y_normaliza_mayusculas(monkeypatch, capsys):
    monkeypatch.setattr(dp, "load_carrier_types",
                        lambda: {"FRACOR": "MASIVO", "CORREKAMINOS": "PAQUETEO"})
    df = pd.DataFrame({"transportadora": ["FRACOR", "Correkaminos ", "DESCONOCIDA"]})
    resultado = dp.tag_carrier_type(df)
    assert len(resultado) == 3
    assert resultado["tipo_transportadora"].tolist()[:2] == ["MASIVO", "PAQUETEO"]
    assert pd.isna(resultado.loc[2, "tipo_transportadora"])
    assert "sin clasificar" in capsys.readouterr().out


def test_mapear_region_aplica_homonimos_e_interior_baq():
    df = pd.DataFrame({
        "guia": ["1", "2", "3", "4"],
        "sede": ["0111", "0111", "0111", "0211"],
        "ciudad_destino": ["RIONEGRO", "CALDAS", "BUCARAMANGA", "BUCARAMANGA"],
        "departamento": ["SANTANDER", "BOYACA", "SANTANDER", "SANTANDER"],
    })
    resultado = dp.map_region(df)
    assert resultado["region_logistica"].tolist() == [
        "ANTIOQUIA_CHOCO",   # Rionegro es el de Antioquia
        "ANTIOQUIA_CHOCO",   # Caldas es el de Antioquia
        "SANTANDERES",
        "INTERIOR_BAQ",      # desde Barranquilla, Santanderes va por el hub Bogotá
    ]


# ------------------------------------------------------------ get_shipments

def _despachos_procesados():
    base = {
        "tipo_despacho": "CLIENTES", "ciudad_destino": "CALI", "ciudad_sede": "BOGOTA",
        "region_logistica": "SUR_OCCIDENTE", "fecha_captura": pd.Timestamp("2026-01-05"),
        "dias_promesa_entrega": 2, "unidades": 1, "tipo_ruta": "NACIONAL",
        "fecha_entrega": pd.Timestamp("2026-01-07"), "latitud_destino": 3.4,
        "longitud_destino": -76.5, "sede_flete": "0111", "tipo_transportadora": "PAQUETEO",
        "transportadora": "COLVANES",
    }
    filas = [
        {"guia": "A", "sede": "0111", "mes": pd.Timestamp("2026-01-02"), "peso": 10.0, "valor_flete": 5000.0},
        {"guia": "B", "sede": "0111", "mes": pd.Timestamp("2026-01-02"), "peso": 20.0, "valor_flete": np.nan},
        {"guia": "C", "sede": "0111", "mes": pd.Timestamp("2026-01-02"), "peso": np.nan, "valor_flete": 3000.0},
        {"guia": "D", "sede": "0211", "mes": pd.Timestamp("2026-01-02"), "peso": 30.0, "valor_flete": 4000.0},
        {"guia": "E", "sede": "0111", "mes": pd.Timestamp("2026-02-02"), "peso": 40.0, "valor_flete": 2000.0},
    ]
    return pd.DataFrame([{**base, **f} for f in filas])


def test_get_shipments_filtra_sede_mes_y_peso_pero_no_flete(monkeypatch):
    monkeypatch.setattr(dp, "process_shipments", _despachos_procesados)
    pedidos = dp.get_shipments("0111", "2026-01-01", "2026-01-31")
    # B se conserva aunque no tenga flete pagado (pedido nuevo en modo operación)
    assert pedidos["guia"].tolist() == ["A", "B"]
    assert pd.isna(pedidos.loc[1, "flete_pagado_actual"])
    assert pedidos.loc[0, "flete_pagado_actual"] == 5000.0
    assert "tipo_transportadora" in pedidos.columns
    assert pedidos["transportadora"].tolist() == ["ENVIA", "ENVIA"]   # Colvanes es ENVIA


def test_agregar_id_unico_numera_guias_repetidas():
    pedidos = pd.DataFrame({"guia": ["X", "X", "Y"]})
    assert dp.add_unique_id(pedidos)["id_pedido"].tolist() == ["X_0", "X_1", "Y_0"]


# ------------------------------------------------------ cotizador con caché

@pytest.fixture
def cotizador_falso(monkeypatch, tmp_path):
    """Reemplaza Excel por una función que registra lo que se le pide y
    redirige la caché a una carpeta temporal (no toca la caché real)."""
    llamadas = []

    def falso(consultas):
        llamadas.append(list(consultas))
        return pd.DataFrame([
            {"origen": o, "destino": d, "peso": p, "unidades": u, "transportadora": "PRUEBA",
             "flete": 1000.0 + p, "tiempo_entrega": 2.0, "promesa_envia": 3.0}
            for o, d, p, u in consultas
        ])

    monkeypatch.setattr(dp, "_excel_quoter", falso)
    monkeypatch.setattr(dp, "CACHE_COTIZACIONES", tmp_path / "cache.parquet")
    return llamadas


def test_cotizador_consulta_una_sola_vez_las_combinaciones_repetidas(cotizador_falso):
    consultas = [("BOGOTA", "CALI", 100.0, 2), ("BOGOTA", "CALI", 100.0, 2), ("BOGOTA", "PASTO", 50.0, 1)]
    resultado = dp.quote(consultas)
    assert len(resultado) == 3
    assert len(cotizador_falso) == 1 and len(cotizador_falso[0]) == 2


def test_cotizador_conserva_el_orden_de_las_consultas(cotizador_falso):
    consultas = [("BOGOTA", "PASTO", 50.0, 1), ("BOGOTA", "CALI", 100.0, 2), ("BOGOTA", "PASTO", 50.0, 1)]
    resultado = dp.quote(consultas)
    assert resultado["destino"].tolist() == ["PASTO", "CALI", "PASTO"]
    assert resultado["flete"].tolist() == [1050.0, 1100.0, 1050.0]


def test_cotizador_reutiliza_la_cache_y_solo_cotiza_lo_nuevo(cotizador_falso):
    dp.quote([("BOGOTA", "CALI", 100.0, 2)])
    dp.quote([("BOGOTA", "CALI", 100.0, 2), ("BOGOTA", "NEIVA", 30.0, 1)])
    assert cotizador_falso[1] == [("BOGOTA", "NEIVA", 30.0, 1)]


def test_cotizador_ignora_ruido_de_punto_flotante_en_el_peso(cotizador_falso):
    dp.quote([("BOGOTA", "CALI", 100.0, 2)])
    dp.quote([("BOGOTA", "CALI", 100.00000001, 2)])
    assert len(cotizador_falso) == 1


def test_cotizador_guarda_la_promesa_de_envia_en_la_cache(cotizador_falso):
    dp.quote([("BOGOTA", "CALI", 100.0, 2)])
    cache = pd.read_parquet(dp.CACHE_COTIZACIONES)
    assert list(cache.columns) == dp.COLUMNAS_COTIZACION
    assert cache.loc[0, "promesa_envia"] == 3.0


def test_cotizador_cotiza_por_tandas_y_guarda_la_cache_despues_de_cada_una(cotizador_falso, monkeypatch):
    # Una cotización larga (horas de Excel) no puede perder lo avanzado: se guarda la caché después de cada tanda
    monkeypatch.setattr(dp, "TAMANO_TANDA_COTIZACION", 2)
    consultas = [("BOGOTA", destino, 10.0, 1) for destino in ("CALI", "PASTO", "NEIVA", "TUNJA", "IBAGUE")]
    resultado = dp.quote(consultas)
    assert [len(tanda) for tanda in cotizador_falso] == [2, 2, 1]
    assert len(pd.read_parquet(dp.CACHE_COTIZACIONES)) == 5
    assert resultado["destino"].tolist() == ["CALI", "PASTO", "NEIVA", "TUNJA", "IBAGUE"]


def test_si_una_tanda_falla_lo_ya_cotizado_queda_en_la_cache(monkeypatch, tmp_path):
    # Un corte a mitad de camino (Excel que se cierra, falta de energía) deja guardadas las tandas terminadas
    llamadas = []

    def falso(consultas):
        llamadas.append(list(consultas))
        if len(llamadas) == 2:
            raise RuntimeError("Excel se cerró")
        return pd.DataFrame([{"origen": o, "destino": d, "peso": p, "unidades": u, "transportadora": "PRUEBA",
                              "flete": 1000.0, "tiempo_entrega": 2.0, "promesa_envia": 3.0} for o, d, p, u in consultas])

    monkeypatch.setattr(dp, "_excel_quoter", falso)
    monkeypatch.setattr(dp, "CACHE_COTIZACIONES", tmp_path / "cache.parquet")
    monkeypatch.setattr(dp, "TAMANO_TANDA_COTIZACION", 2)
    with pytest.raises(RuntimeError):
        dp.quote([("BOGOTA", destino, 10.0, 1) for destino in ("CALI", "PASTO", "NEIVA", "TUNJA")])
    assert sorted(pd.read_parquet(dp.CACHE_COTIZACIONES)["destino"]) == ["CALI", "PASTO"]


def test_cotizador_sin_consultas_devuelve_tabla_vacia_con_columnas(cotizador_falso):
    resultado = dp.quote([])
    assert resultado.empty
    assert list(resultado.columns) == dp.COLUMNAS_COTIZACION


# ------------------------------------------------------------ rate_shipments

NODOS = {"ANTIOQUIA_CHOCO": "MEDELLIN", "COSTA_CARIBE": "BARRANQUILLA", "SUR_OCCIDENTE": "CALI"}


def _pedidos_para_triangulacion():
    return pd.DataFrame([
        # guia, tipo, región, destino, sede_flete, peso
        {"guia": "P1", "tipo": "CLIENTES", "region_logistica": "ANTIOQUIA_CHOCO", "ciudad_destino": "ITAGUI", "sede_flete": "0111", "peso_kg": 10.0},
        {"guia": "P2", "tipo": "TRASLADOS", "region_logistica": "ANTIOQUIA_CHOCO", "ciudad_destino": "SABANETA", "sede_flete": "0106", "peso_kg": 500.0},
        {"guia": "P3", "tipo": "TRASLADOS AL GASTO", "region_logistica": "COSTA_CARIBE", "ciudad_destino": "BARRANQUILLA", "sede_flete": "0112", "peso_kg": 80.0},
        {"guia": "P4", "tipo": "CLIENTES", "region_logistica": "ANTIOQUIA_CHOCO", "ciudad_destino": "MEDELLIN", "sede_flete": "0212", "peso_kg": 20.0},
        {"guia": "P5", "tipo": "TRASLADOS", "region_logistica": "ANTIOQUIA_CHOCO", "ciudad_destino": "MEDELLIN", "sede_flete": "0109", "peso_kg": 30.0},
        {"guia": "P6", "tipo": "CLIENTES", "region_logistica": "ORIENTE", "ciudad_destino": "YOPAL", "sede_flete": "0111", "peso_kg": 15.0},
        {"guia": "P7", "tipo": "CLIENTES", "region_logistica": "SUR_OCCIDENTE", "ciudad_destino": "PUNTA BRAVA", "sede_flete": "0111", "peso_kg": 65.0},
    ]).assign(unidades=1, flete_pagado_actual=[1000.0, 5000.0, 900.0, 800.0, 700.0, 600.0, 2000.0])


@pytest.fixture
def triangulacion():
    consultadas = []

    def falso(consultas):
        consultadas.extend(consultas)
        return pd.DataFrame({
            "flete": [np.nan if d == "PUNTA BRAVA" else 1000.0 + p for _, d, p, _ in consultas],
            "tiempo_entrega": [2.0] * len(consultas),
        })

    return consultadas, falso


def test_rate_shipments_traslado_a_sede_del_hub_llega_directo(monkeypatch, triangulacion):
    consultadas, falso = triangulacion
    monkeypatch.setattr(dp, "quote", falso)
    pedidos = dp.rate_shipments(_pedidos_para_triangulacion(), NODOS)
    directos = pedidos.set_index("guia").loc[["P2", "P3"]]
    assert (directos["flete_triangulacion"] == 0).all()
    destinos_consultados = [d for _, d, _, _ in consultadas]
    assert "SABANETA" not in destinos_consultados and "BARRANQUILLA" not in destinos_consultados


def test_rate_shipments_cliente_se_cotiza_aunque_su_sede_sea_la_del_hub(monkeypatch, triangulacion):
    consultadas, falso = triangulacion
    monkeypatch.setattr(dp, "quote", falso)
    pedidos = dp.rate_shipments(_pedidos_para_triangulacion(), NODOS).set_index("guia")
    assert pedidos.loc["P4", "flete_triangulacion"] == 1020.0      # cliente con sede_flete 0212
    assert pedidos.loc["P5", "flete_triangulacion"] == 1030.0      # traslado a otra sede (0109)
    assert ("MEDELLIN", "MEDELLIN", 20.0, 1) in consultadas


def test_rate_shipments_region_sin_hub_no_se_cotiza(monkeypatch, triangulacion):
    consultadas, falso = triangulacion
    monkeypatch.setattr(dp, "quote", falso)
    pedidos = dp.rate_shipments(_pedidos_para_triangulacion(), NODOS).set_index("guia")
    assert pedidos.loc["P6", "flete_triangulacion"] == 0
    assert not pedidos.loc["P6", "sin_ruta_triangulacion"]
    assert "YOPAL" not in [d for _, d, _, _ in consultadas]


def test_rate_shipments_sin_ruta_recibe_big_m(monkeypatch, triangulacion):
    _, falso = triangulacion
    monkeypatch.setattr(dp, "quote", falso)
    pedidos = dp.rate_shipments(_pedidos_para_triangulacion(), NODOS).set_index("guia")
    assert pedidos.loc["P7", "sin_ruta_triangulacion"]
    # Big-M = FACTOR_BIG_M veces la mayor tarifa conocida (cotizada o pagada): aquí el pagado de 5000
    assert pedidos.loc["P7", "flete_triangulacion"] == 5000.0 * parametros.FACTOR_BIG_M
    # supera cualquier flete directo, así que el modelo nunca lo consolida
    assert pedidos.loc["P7", "flete_triangulacion"] > pedidos["flete_pagado_actual"].max()


def test_rate_shipments_big_m_funciona_sin_fletes_pagados(monkeypatch, triangulacion):
    _, falso = triangulacion
    monkeypatch.setattr(dp, "quote", falso)
    pedidos = _pedidos_para_triangulacion().assign(flete_pagado_actual=np.nan)
    pedidos = dp.rate_shipments(pedidos, NODOS).set_index("guia")
    assert pedidos.loc["P7", "flete_triangulacion"] == 1030.0 * parametros.FACTOR_BIG_M


# ----------------------------------------------------------- rate_ltl_direct

def test_rate_ltl_directo_promesa_estricta_y_sin_respaldo_del_pagado(monkeypatch):
    monkeypatch.setattr(dp, "quote", lambda consultas: pd.DataFrame({
        "transportadora": ["ENVIA", np.nan],
        "flete": [15000.0, np.nan],
        "tiempo_entrega": [2.0, np.nan],
        "promesa_envia": [3.0, np.nan],
    }))
    pedidos = pd.DataFrame({
        "ciudad_sede": ["BOGOTA", "BOGOTA"], "ciudad_destino": ["CALI", "PUNTA BRAVA"],
        "peso_kg": [10.0, 20.0], "unidades": [1, 1], "flete_pagado_actual": [9000.0, 8000.0],
    })
    resultado = dp.rate_ltl_direct(pedidos)
    assert resultado["promesa_estricta"].tolist() == [3.0, 8.0]   # ENVIA; sin ENVIA, reexpedido 8
    assert "promesa_maxima" not in resultado.columns            # se eliminó el día de gracia
    assert pd.isna(resultado.loc[1, "flete_directo_cotizado"])  # ya no se rellena con lo pagado
    assert resultado.loc[0, "transportadora_directo_cotizada"] == "ENVIA"


# ------------------------------------------------ prepare_unconsolidated_cost

def _pedidos_cotizados():
    return pd.DataFrame({
        "id_pedido": ["A_0", "B_0", "C_0"],
        "flete_pagado_actual": [900.0, 800.0, 700.0],
        "flete_directo_cotizado": [600.0, np.nan, 500.0],
        "transportadora": ["ENVIA", "SERVIENTREGA", "ENVIA"],
        "transportadora_directo_cotizada": ["SERVIENTREGA", np.nan, "ENVIA"],
    })


def test_modo_historico_usa_lo_pagado():
    pedidos, revision = dp.prepare_unconsolidated_cost(_pedidos_cotizados(), "historico")
    assert pedidos["flete_sin_consolidar"].tolist() == [900.0, 800.0, 700.0]
    assert pedidos["transportadora_directo"].tolist() == ["ENVIA", "SERVIENTREGA", "ENVIA"]  # la que despachó
    assert revision.empty


def test_modo_historico_se_detiene_si_falta_el_pagado():
    pedidos = _pedidos_cotizados()
    pedidos.loc[0, "flete_pagado_actual"] = np.nan
    with pytest.raises(ValueError, match="flete pagado"):
        dp.prepare_unconsolidated_cost(pedidos, "historico")


def test_modo_operacion_usa_lo_cotizado_y_separa_sin_cotizacion(capsys):
    pedidos, revision = dp.prepare_unconsolidated_cost(_pedidos_cotizados(), "operacion")
    assert pedidos["id_pedido"].tolist() == ["A_0", "C_0"]
    assert pedidos["flete_sin_consolidar"].tolist() == [600.0, 500.0]
    assert pedidos["transportadora_directo"].tolist() == ["SERVIENTREGA", "ENVIA"]  # la ganadora del cotizador
    assert revision["id_pedido"].tolist() == ["B_0"]
    assert "¿querías correr --modo historico?" in capsys.readouterr().out


def test_modo_operacion_sin_flete_pagado_no_advierte(capsys):
    pedidos = _pedidos_cotizados().assign(flete_pagado_actual=np.nan)
    dp.prepare_unconsolidated_cost(pedidos, "operacion")
    assert "traen flete pagado" not in capsys.readouterr().out   # el aviso de flete pagado no aparece
