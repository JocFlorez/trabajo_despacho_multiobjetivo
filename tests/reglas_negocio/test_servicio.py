"""Nivel de servicio: cumplimiento de las transportadoras, retraso esperado y probabilidad de llegar a tiempo."""
from datetime import date

import pandas as pd
import pytest

from despacho_multiobjetivo import parametros
from despacho_multiobjetivo.reglas_negocio import servicio

LUNES, MARTES = date(2026, 1, 5), date(2026, 1, 6)


def _pedidos():
    return pd.DataFrame({
        "id_pedido": ["E_0", "S_0", "M_0", "T_0"],
        "transportadora_directo": ["ENVIA", "SERVIENTREGA", "FRACOR", "ENVIA"],
        "tipo_transportadora": ["PAQUETEO", "PAQUETEO", "MASIVO", "PAQUETEO"],
        "sede": ["0111", "0111", "0111", "0111"],
        "region_logistica": ["ANTIOQUIA_CHOCO", "ANTIOQUIA_CHOCO", "COSTA_CARIBE", "SUR_OCCIDENTE"],
        "llega_directo": [False, False, False, True],   # T_0: traslado que llega directo a la sede del hub
    })


def _opciones():
    return pd.DataFrame({"id_pedido": ["E_0", "E_0", "T_0"],
                         "dia_salida": [LUNES, MARTES, LUNES],
                         "retraso": [0.0, 1.0, 0.0]})


# ---------------------------------------------------------------- cumplimiento por transportadora y región

@pytest.mark.parametrize("transportadora, tipo, sede, region, esperado", [
    ("FRACOR", "MASIVO", "0111", "COSTA_CARIBE", 1.0),             # carga masiva
    ("ENVIA", "PAQUETEO", "0111", "ANTIOQUIA_CHOCO", 0.738),       # ENVIA medido por sede y región
    ("ENVIA", "PAQUETEO", "0111", "REGION_SIN_DATO", 0.85),        # ENVIA sin dato: valor por defecto
    ("SERVIENTREGA", "PAQUETEO", "0111", "ANTIOQUIA_CHOCO", 0.60),
    ("OTRA", "PAQUETEO", "0111", "ANTIOQUIA_CHOCO", 0.85),         # otra transportadora: valor por defecto
])
def test_cumplimiento_directo(transportadora, tipo, sede, region, esperado):
    assert servicio.direct_on_time_rate(transportadora, tipo, sede, region) == pytest.approx(esperado)


def test_cumplimiento_del_ultimo_tramo_por_region():
    assert servicio.last_mile_on_time_rate("SUR_OCCIDENTE") == pytest.approx(0.90)
    assert servicio.last_mile_on_time_rate("CENTRO") == pytest.approx(0.85)   # sin dato: por defecto


# ---------------------------------------------------------------- castigo por llegar tarde

@pytest.fixture
def castigo_sencillo(monkeypatch):
    # Parámetros fáciles de verificar a mano: i = 0,5 (g = 0; 1; 2,5; 4,75; 8,125), súper castigo ×10 desde 4 días
    # y una falla suma 0 días (cuenta como 1) o 2 días con igual probabilidad
    monkeypatch.setattr(parametros, "TASA_CASTIGO_RETRASO", 0.5)
    monkeypatch.setattr(parametros, "DIAS_SUPER_CASTIGO", 4)
    monkeypatch.setattr(parametros, "FACTOR_SUPER_CASTIGO", 10)
    monkeypatch.setattr(parametros, "HISTOGRAMA_DIAS_TARDE", {0: 0.5, 2: 0.5})


def test_castigo_tasa_de_interes(castigo_sencillo):
    # Cada día adicional pesa 1,5 veces el anterior: 1, +1,5, +2,25
    assert [servicio.delay_penalty(k) for k in range(4)] == pytest.approx([0, 1, 2.5, 4.75])


def test_desde_cuatro_dias_se_multiplica_por_el_super_castigo(castigo_sencillo):
    assert servicio.delay_penalty(4) == pytest.approx(8.125 * 10)
    assert servicio.delay_penalty(5) == pytest.approx(13.1875 * 10)


def test_con_tasa_cero_el_castigo_es_lineal(castigo_sencillo, monkeypatch):
    monkeypatch.setattr(parametros, "TASA_CASTIGO_RETRASO", 0)
    assert [servicio.delay_penalty(k) for k in range(4)] == [0, 1, 2, 3]


def test_castigo_esperado_sin_retraso_planeado(castigo_sencillo):
    # Cumple el 80 %: sin castigo. Falla el 20 %: 0 días cuenta como 1 (castigo 1) o 2 días (castigo 2,5)
    assert servicio.expected_delay_penalty(0.8, 0) == pytest.approx(0.2 * (0.5 * 1 + 0.5 * 2.5))


def test_castigo_esperado_con_un_dia_planeado(castigo_sencillo):
    # Cumple: 1 día (castigo 1). Falla: 1 + 1 = 2 días (2,5) o 1 + 2 = 3 días (4,75)
    assert servicio.expected_delay_penalty(0.8, 1) == pytest.approx(0.8 * 1 + 0.2 * (0.5 * 2.5 + 0.5 * 4.75))


def test_entrega_tardia_en_fin_de_semana_cuenta_como_un_dia(castigo_sencillo, monkeypatch):
    monkeypatch.setattr(parametros, "HISTOGRAMA_DIAS_TARDE", {0: 1.0})
    assert servicio.expected_delay_penalty(0, 0) == pytest.approx(1)   # no 0


def test_el_super_castigo_aplica_al_retraso_planeado_mas_el_de_la_falla(castigo_sencillo, monkeypatch):
    # Planeado 3 días + 1 día de falla = 4 días → súper castigo
    monkeypatch.setattr(parametros, "HISTOGRAMA_DIAS_TARDE", {1: 1.0})
    assert servicio.expected_delay_penalty(0, 3) == pytest.approx(81.25)


def test_castigo_esperado_de_cada_despacho_directo(castigo_sencillo):
    castigo = servicio.expected_delay_direct(_pedidos())
    assert castigo.index.tolist() == ["E_0", "S_0", "M_0", "T_0"]
    assert castigo["E_0"] == pytest.approx((1 - 0.738) * 1.75)
    assert castigo["S_0"] == pytest.approx((1 - 0.60) * 1.75)
    assert castigo["M_0"] == pytest.approx(0)   # la carga masiva siempre cumple


def test_castigo_esperado_de_cada_opcion_de_camion(castigo_sencillo):
    castigo = servicio.expected_delay_options(_opciones(), _pedidos())
    assert castigo.index.names == ["id_pedido", "dia_salida"]
    assert castigo["E_0", LUNES] == pytest.approx(0.15 * 1.75)                      # a tiempo; falla el último tramo 15 %
    assert castigo["E_0", MARTES] == pytest.approx(0.85 * 1 + 0.15 * 3.625)         # 1 día planeado
    assert castigo["T_0", LUNES] == pytest.approx(0)                                # llega directo a la sede del hub


def test_el_castigo_usa_los_parametros_vigentes(castigo_sencillo, monkeypatch):
    # Los parámetros se leen al calcular: así se corre un escenario de sensibilidad
    antes = servicio.expected_delay_direct(_pedidos())["E_0"]
    monkeypatch.setattr(parametros, "TASA_CASTIGO_RETRASO", 1.0)   # g(2) pasa de 2,5 a 3
    assert servicio.expected_delay_direct(_pedidos())["E_0"] == pytest.approx((1 - 0.738) * (0.5 * 1 + 0.5 * 3))
    assert servicio.expected_delay_direct(_pedidos())["E_0"] > antes


# ---------------------------------------------------------------- probabilidad de llegar a tiempo

def test_probabilidad_directo_es_el_cumplimiento_de_su_transportadora():
    probabilidad = servicio.on_time_probability_direct(_pedidos())
    assert probabilidad.tolist() == pytest.approx([0.738, 0.60, 1.0, 0.817])


def test_probabilidad_consolidado_es_cero_si_ya_sale_tarde():
    probabilidad = servicio.on_time_probability_options(_opciones(), _pedidos())
    assert probabilidad["E_0", LUNES] == pytest.approx(0.85)
    assert probabilidad["E_0", MARTES] == 0                     # con 1 día planeado de retraso no llega a tiempo
    assert probabilidad["T_0", LUNES] == pytest.approx(1.0)     # llega directo a la sede del hub


# ---------------------------------------------------------------- histograma de días de retraso

def _despachos(entregas, captura="2026-01-13", promesa=3):
    # Captura mar 13 con promesa de 3 días hábiles (sábado no hábil) → fecha límite vie 16
    return pd.DataFrame({"fecha_captura": [captura] * len(entregas), "dias_promesa_entrega": [promesa] * len(entregas),
                         "fecha_entrega": entregas})


def test_viernes_a_lunes_es_un_dia_habil_y_no_tres():
    # Si se restaran fechas (entrega − límite) daría 3 días calendario; el sábado y el domingo no cuentan
    histograma = servicio.late_days_histogram(_despachos(["2026-01-19"]))
    assert histograma[1] == 1.0


def test_entrega_en_fin_de_semana_suma_cero_dias():
    # Vencía el viernes y llegó el sábado: llegó tarde, pero sin días hábiles de retraso (k = 0)
    histograma = servicio.late_days_histogram(_despachos(["2026-01-17"]))
    assert histograma[0] == 1.0


def test_el_festivo_no_cuenta_como_dia_de_retraso():
    # Captura mar 6 → límite vie 9; el lunes 12 es festivo: entregar el martes 13 es 1 día hábil tarde
    histograma = servicio.late_days_histogram(_despachos(["2026-01-13"], captura="2026-01-06"))
    assert histograma[1] == 1.0


def test_solo_cuenta_los_que_llegaron_tarde_y_agrupa_cuatro_o_mas():
    # A tiempo (jue 15) no entra; lun 19 = 1 día; sáb 17 = 0 días; vie 23 = 5 días → grupo "4 o más"
    histograma = servicio.late_days_histogram(_despachos(["2026-01-15", "2026-01-19", "2026-01-17", "2026-01-23"]))
    assert histograma.to_dict() == pytest.approx({0: 1 / 3, 1: 1 / 3, 2: 0, 3: 0, 4: 1 / 3})


def test_sin_retrasos_el_histograma_queda_en_ceros():
    histograma = servicio.late_days_histogram(_despachos(["2026-01-14", "2026-01-16"]))
    assert histograma.tolist() == [0, 0, 0, 0, 0]


# ---------------------------------------------------------------- días hábiles de retraso, sin castigo (para reportar)

def test_dias_de_retraso_esperados_sin_castigo(castigo_sencillo):
    # Una falla suma 0 días (cuenta como 1) o 2 días: en promedio 1,5 días; con 1 día planeado, 2 o 3 días
    assert servicio.expected_delay_days(0.8, 0) == pytest.approx(0.2 * 1.5)
    assert servicio.expected_delay_days(0.8, 1) == pytest.approx(0.8 * 1 + 0.2 * 2.5)


def test_directo_y_opciones_en_dias_sin_castigo(castigo_sencillo):
    dias = servicio.expected_delay_direct(_pedidos(), castigo=False)
    assert dias["E_0"] == pytest.approx((1 - 0.738) * 1.5)
    dias_opciones = servicio.expected_delay_options(_opciones(), _pedidos(), castigo=False)
    assert dias_opciones["E_0", MARTES] == pytest.approx(0.85 * 1 + 0.15 * 2.5)
    # Con el súper castigo los días no se multiplican: un retraso de 5 días cuenta 5
    assert servicio.expected_delay_days(0, 4) == pytest.approx(4 + 1.5)


# ---------------------------------------------------------------- directo tardío (fase 3 del Monte Carlo)
# Un pedido que esperó y salió directo llega tarde los días que esperó: su retraso planeado son los días hábiles
# entre el día en que entró a operación y el día en que salió, con el sábado como no hábil (igual que el retraso
# de las opciones en camión). Sin espera es el castigo directo de siempre.

LUN_F, MIE_F, VIE_F, LUN_F2 = date(2026, 2, 2), date(2026, 2, 4), date(2026, 2, 6), date(2026, 2, 9)


def _directos_tardios(llegadas, salidas, filas=(0, 2)):
    # E_0 (ENVIA, cumple 73,8 %) y M_0 (carga masiva, siempre cumple), con su día de llegada y de salida
    return _pedidos().iloc[list(filas)].assign(dia_llegada=llegadas, dia_salida=salidas).reset_index(drop=True)


def test_sin_espera_es_el_castigo_directo_de_siempre(castigo_sencillo):
    despachos = _directos_tardios([LUN_F, LUN_F], [LUN_F, LUN_F])
    tardio = servicio.expected_delay_late_direct(despachos)
    directo = servicio.expected_delay_direct(despachos)
    assert tardio.tolist() == pytest.approx(directo.tolist())


def test_cada_dia_esperado_es_un_dia_de_retraso_planeado(castigo_sencillo):
    # E_0 espera de lunes a miércoles: 2 días. Cumple con 2 días tarde (g = 2,5); si falla, 3 o 4 días
    # (g = 4,75 o 8,125 × 10 por el súper castigo desde 4 días)
    tardio = servicio.expected_delay_late_direct(_directos_tardios([LUN_F], [MIE_F], filas=(0,)))
    assert tardio["E_0"] == pytest.approx(0.738 * 2.5 + 0.262 * (0.5 * 4.75 + 0.5 * 81.25))


def test_de_viernes_a_lunes_es_un_dia_porque_el_sabado_no_cuenta(castigo_sencillo):
    tardio = servicio.expected_delay_late_direct(_directos_tardios([VIE_F], [LUN_F2], filas=(0,)))
    assert tardio["E_0"] == pytest.approx(0.738 * 1 + 0.262 * (0.5 * 2.5 + 0.5 * 4.75))


def test_la_carga_masiva_que_espera_llega_tarde_lo_que_espero(castigo_sencillo):
    # Siempre cumple su tránsito: el único retraso es el que se planeó al esperar
    tardio = servicio.expected_delay_late_direct(_directos_tardios([LUN_F], [MIE_F], filas=(2,)))
    assert tardio["M_0"] == pytest.approx(2.5)


def test_mas_espera_mas_castigo(castigo_sencillo):
    castigos = [servicio.expected_delay_late_direct(_directos_tardios([LUN_F], [salida], filas=(0,)))["E_0"]
                for salida in (LUN_F, date(2026, 2, 3), MIE_F)]
    assert castigos == sorted(castigos) and len(set(castigos)) == 3


def test_directo_tardio_en_dias_sin_castigo(castigo_sencillo):
    # Sin castigo: 2 días si cumple; si falla, 3 o 4 días (en promedio 3,5)
    dias = servicio.expected_delay_late_direct(_directos_tardios([LUN_F], [MIE_F], filas=(0,)), castigo=False)
    assert dias["E_0"] == pytest.approx(0.738 * 2 + 0.262 * 3.5)


def test_directo_tardio_por_pedido_sin_modificar_lo_que_recibe(castigo_sencillo):
    despachos = _directos_tardios([LUN_F, LUN_F], [MIE_F, LUN_F])
    antes = despachos.copy()
    tardio = servicio.expected_delay_late_direct(despachos)
    assert tardio.index.tolist() == ["E_0", "M_0"]
    pd.testing.assert_frame_equal(despachos, antes)
