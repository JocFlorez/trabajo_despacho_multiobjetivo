"""Escenarios de demanda para el Monte Carlo: remuestreo de días completos del histórico."""
import random
from datetime import date

import pandas as pd

from despacho_multiobjetivo.reglas_negocio.opciones_salida import build_departure_options
from despacho_multiobjetivo.simulacion.escenarios import (build_day_pool, candidate_days, day_stratum, relabel_day,
                                                        generar_demanda)

LUN = date(2026, 2, 2)
MAR = date(2026, 2, 3)
SAB = date(2026, 2, 7)
DOM = date(2026, 2, 1)


def _pedidos():
    return pd.DataFrame({"id_pedido": ["A", "B", "C", "D", "E"],
                         "fecha_captura": pd.to_datetime([LUN, LUN, MAR, DOM, SAB]),
                         "peso_kg": [100.0, 200.0, 300.0, 400.0, 500.0]})


# ---------------------------------------------------------------- días del histórico

def test_cada_dia_guarda_sus_pedidos_completos():
    # Los pedidos del mismo día quedan juntos, con todas sus columnas
    pool = build_day_pool(_pedidos())
    assert sorted(pool[MAR]["id_pedido"]) == ["C"]
    assert pool[MAR]["peso_kg"].tolist() == [300.0]
    assert list(pool[MAR].columns) == list(_pedidos().columns)


def test_el_pedido_capturado_en_domingo_cuenta_el_dia_que_entra_a_operacion():
    # El domingo no se despacha: D entra a operación el lunes, junto con A y B
    pool = build_day_pool(_pedidos())
    assert sorted(pool[LUN]["id_pedido"]) == ["A", "B", "D"]
    assert DOM not in pool


def test_el_sabado_es_un_dia_de_operacion():
    pool = build_day_pool(_pedidos())
    assert sorted(pool[SAB]["id_pedido"]) == ["E"]


def test_los_dias_quedan_en_orden_cronologico():
    assert list(build_day_pool(_pedidos())) == [LUN, MAR, SAB]


def test_pedido_sin_fecha_de_captura_no_entra_y_se_avisa(capsys):
    pedidos = pd.concat([_pedidos(), pd.DataFrame({"id_pedido": ["F"], "fecha_captura": [pd.NaT], "peso_kg": [1.0]})])
    pool = build_day_pool(pedidos)
    assert "F" not in pd.concat(pool.values())["id_pedido"].tolist()
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_no_cambia_los_pedidos_que_recibe():
    pedidos = _pedidos()
    copia = pedidos.copy()
    build_day_pool(pedidos)
    pd.testing.assert_frame_equal(pedidos, copia)


# ---------------------------------------------------------------- un día del histórico en otra fecha

def _dia():
    return pd.DataFrame({"id_pedido": ["A", "B"], "fecha_captura": pd.to_datetime([LUN, LUN]),
                         "fecha_entrega": pd.to_datetime([date(2026, 2, 5), date(2026, 2, 6)]),
                         "peso_kg": [100.0, 200.0]}, index=[7, 9])


def test_los_pedidos_quedan_con_la_fecha_nueva():
    nuevo = relabel_day(_dia(), MAR, "e1")
    assert nuevo["fecha_captura"].tolist() == [pd.Timestamp(MAR)] * 2


def test_conserva_lo_demas_del_pedido():
    assert relabel_day(_dia(), MAR, "e1")["peso_kg"].tolist() == [100.0, 200.0]


def test_el_mismo_dia_sorteado_dos_veces_no_repite_identificadores():
    # El remuestreo puede sacar un día varias veces, y el modelo necesita un identificador único por pedido
    uno, dos = relabel_day(_dia(), MAR, "e1"), relabel_day(_dia(), SAB, "e2")
    identificadores = pd.concat([uno, dos])["id_pedido"]
    assert identificadores.is_unique
    assert uno["id_pedido"].str.startswith(("A", "B")).all()


def test_la_entrega_historica_se_borra():
    # Esa entrega ocurrió en otra fecha: no debe confundirse con la del pedido simulado
    assert relabel_day(_dia(), MAR, "e1")["fecha_entrega"].isna().all()


def test_funciona_sin_columna_de_entrega():
    assert "fecha_entrega" not in relabel_day(_pedidos().head(2), MAR, "e1").columns


def test_la_numeracion_de_filas_empieza_de_cero():
    assert relabel_day(_dia(), MAR, "e1").index.tolist() == [0, 1]


def test_no_cambia_el_dia_que_recibe():
    dia = _dia()
    copia = dia.copy()
    relabel_day(dia, MAR, "e1")
    pd.testing.assert_frame_equal(dia, copia)


def test_las_opciones_de_salida_salen_de_la_fecha_nueva():
    # Lo que importa al final: el pedido simulado puede salir desde su fecha nueva, no desde la del histórico
    dia = pd.DataFrame({"id_pedido": ["A"], "fecha_captura": pd.to_datetime([LUN]), "sede": ["0111"],
                        "nodo_regional": ["MEDELLIN"], "sin_ruta_triangulacion": [False],
                        "tiempo_triangulacion": [1.0], "promesa_estricta": [3], "llega_directo": [False]})
    nuevo = relabel_day(dia, MAR, "e1")
    opciones = build_departure_options(nuevo, {("0111", "MEDELLIN"): 1}, espera_maxima=1, manipulacion=1)
    assert opciones["dia_salida"].min() == MAR
    assert opciones["id_pedido"].unique().tolist() == nuevo["id_pedido"].tolist()


# ---------------------------------------------------------------- tramo del mes y día de la semana

def test_estrato_de_un_dia_con_los_tramos_por_defecto():
    # Tramos: días 1-10 (0), 11-20 (1) y 21-fin (2). weekday: lunes = 0
    assert day_stratum(date(2026, 2, 2)) == (0, 0)
    assert day_stratum(date(2026, 2, 17)) == (1, 1)
    assert day_stratum(date(2026, 2, 25)) == (2, 2)


def test_los_limites_de_un_tramo_quedan_dentro_del_tramo():
    assert day_stratum(date(2026, 2, 10))[0] == 0
    assert day_stratum(date(2026, 2, 11))[0] == 1
    assert day_stratum(date(2026, 2, 20))[0] == 1
    assert day_stratum(date(2026, 2, 21))[0] == 2


def test_los_tramos_se_pueden_cambiar_si_cambia_el_cierre():
    # Con cierre desde el día 25: 1-24 es un tramo y 25-fin es otro
    assert day_stratum(date(2026, 2, 23), limites_tramos=(24,))[0] == 0
    assert day_stratum(date(2026, 2, 25), limites_tramos=(24,))[0] == 1


# ---------------------------------------------------------------- de qué días se puede sortear

LUNES_INICIO, LUNES_CIERRE, MARTES_CIERRE = date(2026, 2, 2), date(2026, 2, 23), date(2026, 2, 24)
OTRO_LUNES_CIERRE = date(2026, 3, 23)


def _pool():
    def dia(*identificadores):
        return pd.DataFrame({"id_pedido": list(identificadores), "fecha_captura": pd.Timestamp(LUNES_INICIO),
                             "peso_kg": 100.0})
    return {LUNES_INICIO: dia("I1", "I2"), LUNES_CIERRE: dia("C1"), MARTES_CIERRE: dia("M1"),
            OTRO_LUNES_CIERRE: dia("O1", "O2", "O3")}


def test_candidatos_del_mismo_tramo_y_dia_de_la_semana():
    assert candidate_days(_pool(), date(2026, 4, 27)) == [LUNES_CIERRE, OTRO_LUNES_CIERRE]


def test_sin_ese_dia_de_la_semana_usa_el_mismo_tramo_y_avisa(capsys):
    # Un viernes del tramo de cierre: el histórico no tiene, se usa cualquier día de ese tramo
    assert candidate_days(_pool(), date(2026, 4, 24)) == [LUNES_CIERRE, MARTES_CIERRE, OTRO_LUNES_CIERRE]
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_sin_ese_tramo_usa_cualquier_dia_y_avisa(capsys):
    # El tramo 11-20 no tiene días en este histórico
    assert candidate_days(_pool(), date(2026, 4, 13)) == list(_pool())
    assert "ADVERTENCIA" in capsys.readouterr().out


# ---------------------------------------------------------------- los futuros

def _futuros(n=5, semilla=0, fechas=(date(2026, 4, 27), date(2026, 4, 28))):
    return generar_demanda(_pool(), list(fechas), n, random.Random(semilla))


def test_devuelve_la_cantidad_de_futuros_pedida():
    assert len(_futuros(n=7)) == 7


def test_cada_futuro_trae_los_pedidos_de_todos_sus_dias_con_la_fecha_simulada():
    for futuro in _futuros():
        assert set(futuro["fecha_captura"]) == {pd.Timestamp(2026, 4, 27), pd.Timestamp(2026, 4, 28)}


def test_cada_fecha_solo_recibe_dias_de_su_mismo_estrato():
    # El lunes 27 solo puede traer pedidos de lunes de cierre (C1 u O); el martes 28, de martes de cierre (M1)
    for futuro in _futuros(n=20):
        lunes = futuro[futuro["fecha_captura"] == pd.Timestamp(2026, 4, 27)]["id_pedido"]
        martes = futuro[futuro["fecha_captura"] == pd.Timestamp(2026, 4, 28)]["id_pedido"]
        assert lunes.str.startswith(("C1", "O")).all()
        assert martes.str.startswith("M1").all()


def test_un_dia_sorteado_dos_veces_en_un_futuro_no_repite_identificadores():
    # 27 de abril y 4 de mayo no son del mismo tramo; 20 y 27 de abril tampoco: se usan dos lunes de cierre seguidos
    fechas = [date(2026, 4, 27), date(2026, 6, 29)]
    pool = {LUNES_CIERRE: _pool()[LUNES_CIERRE]}
    futuro = generar_demanda(pool, fechas, 1, random.Random(0))[0]
    assert futuro["id_pedido"].is_unique
    assert len(futuro) == 2


def test_futuros_distintos_tienen_identificadores_distintos():
    uno, dos = _futuros(n=2)
    assert not set(uno["id_pedido"]) & set(dos["id_pedido"])


def test_con_la_misma_semilla_salen_los_mismos_futuros():
    for a, b in zip(_futuros(semilla=3), _futuros(semilla=3)):
        pd.testing.assert_frame_equal(a, b)


def test_el_sorteo_varia_entre_futuros():
    # Con 20 futuros, el lunes de cierre debe salir de los dos días posibles
    tamanos = {len(futuro) for futuro in _futuros(n=20, fechas=(date(2026, 4, 27),))}
    assert tamanos == {1, 3}


def test_no_cambia_el_historico_que_recibe():
    pool = _pool()
    generar_demanda(pool, [date(2026, 4, 27)], 3, random.Random(0))
    for dia, pedidos in _pool().items():
        pd.testing.assert_frame_equal(pool[dia], pedidos)
