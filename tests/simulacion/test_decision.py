"""Decisión diaria con futuros (fase 4 del Monte Carlo): piezas comunes de las políticas A y C.

Cada pedido se describe con un diccionario:
  id, peso, flete_directo, flete_tramo (último tramo si va en camión), llegada, ultimo_dia,
  castigo_camion {día: castigo si sale ese día en camión} (solo los días con opción de camión) y
  castigo_directo {día: castigo si sale directo ese día} (sube si esperó: directo tardío)
Tabla de camiones de la instancia pequeña: Turbo de 4.500 kg por 2,0 M y Sencillo de 8.500 kg por 2,2 M.
"""
from datetime import date

import pytest

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import truck_cost_table
from despacho_multiobjetivo.simulacion.decision import _lote, decision_window, dispatch_when_pays, obj_promedio
from tests.metodos.instancia_pequena import CAPACIDADES, SEDE, TARIFAS

LUN, MAR, MIE, JUE, VIE, SAB = (date(2026, 2, d) for d in (2, 3, 4, 5, 6, 7))
LUN9, MAR10 = date(2026, 2, 9), date(2026, 2, 10)
TABLA = truck_cost_table(CAPACIDADES, TARIFAS, SEDE, "MEDELLIN", 20_000)


def _pedido(id_pedido, peso, ahorro, llegada=LUN, ultimo_dia=MIE, dias_camion=(LUN, MAR, MIE), flete_tramo=100_000):
    # Castigos sencillos para verificar a mano: en camión 0,5; directo 1,0 el día que llega y +1 por cada día esperado
    dias = [LUN, MAR, MIE, JUE, VIE, SAB, LUN9, MAR10]
    return {"id": id_pedido, "peso": peso, "flete_directo": flete_tramo + ahorro, "flete_tramo": flete_tramo,
            "llegada": llegada, "ultimo_dia": ultimo_dia,
            "castigo_camion": {dia: 0.5 for dia in dias_camion},
            "castigo_directo": {dia: 1.0 + max(0, dias.index(dia) - dias.index(llegada)) for dia in dias}}


# ---------------------------------------------------------------- ventana de días de los futuros

def test_la_ventana_va_de_manana_al_ultimo_dia_de_los_conocidos():
    conocidos = [_pedido("A", 1000, 1, ultimo_dia=JUE), _pedido("B", 1000, 1, ultimo_dia=MIE)]
    assert decision_window(conocidos, LUN) == [MAR, MIE, JUE]


def test_la_ventana_salta_el_domingo_y_cuenta_el_sabado():
    # Lo que llega el viernes puede esperar hasta el martes: sábado sí, domingo no
    assert decision_window([_pedido("A", 1000, 1, llegada=VIE, ultimo_dia=MAR10)], VIE) == [SAB, LUN9, MAR10]


def test_si_todos_vencen_hoy_no_hay_ventana():
    assert decision_window([_pedido("A", 1000, 1, ultimo_dia=LUN)], LUN) == []


def test_sin_conocidos_no_hay_ventana():
    assert decision_window([], LUN) == []


# ---------------------------------------------------------------- R1: el camión sale cuando ya se paga

def test_si_el_camion_no_se_paga_espera_y_al_vencer_sale_directo():
    # 2.000 kg que ahorran 1,5 M no pagan una Turbo de 2,0 M: espera y el miércoles sale directo, tarde
    resultado = dispatch_when_pays([_pedido("A", 2000, 1_500_000)], [MAR, MIE], TABLA)
    assert resultado["salidas"] == {"A": (MIE, "directo")}
    assert resultado["costo"] == pytest.approx(1_600_000)
    assert resultado["castigo"] == pytest.approx(3.0)   # directo el miércoles: 2 días esperados


def test_cuando_la_carga_ya_paga_el_camion_sale_ese_dia():
    # 4.000 kg que ahorran 3,0 M pagan la Turbo de 2,0 M
    resultado = dispatch_when_pays([_pedido("A", 2000, 1_500_000), _pedido("B", 2000, 1_500_000)], [MAR, MIE], TABLA)
    assert resultado["salidas"] == {"A": (MAR, "camion"), "B": (MAR, "camion")}
    assert resultado["costo"] == pytest.approx(2 * 100_000 + 2_100_000)
    assert resultado["castigo"] == pytest.approx(1.0)


def test_con_5_8_toneladas_conviene_la_turbo_y_el_resto_sigue_esperando():
    # B (1.500 kg, 600 $/kg) y A (3.000 kg, 500 $/kg) llenan la Turbo: ganan 2,4 − 2,0 = 0,4 M.
    # Sumar C (1.300 kg, casi sin ahorro) obliga a un Sencillo: 2,5 − 2,2 = 0,3 M. Gana la Turbo sin C
    pedidos = [_pedido("A", 3000, 1_500_000), _pedido("B", 1500, 900_000), _pedido("C", 1300, 100_000)]
    resultado = dispatch_when_pays(pedidos, [MAR, MIE], TABLA)
    assert resultado["salidas"]["A"] == (MAR, "camion") and resultado["salidas"]["B"] == (MAR, "camion")
    assert resultado["salidas"]["C"] == (MIE, "directo")


def test_un_pedido_que_llega_despues_puede_completar_el_camion():
    # A solo no paga el martes; el miércoles llega D y juntos sí pagan
    pedidos = [_pedido("A", 2000, 1_500_000, ultimo_dia=JUE, dias_camion=(MAR, MIE, JUE)),
               _pedido("D", 2000, 1_500_000, llegada=MIE, ultimo_dia=VIE, dias_camion=(MIE, JUE, VIE))]
    resultado = dispatch_when_pays(pedidos, [MAR, MIE, JUE], TABLA)
    assert resultado["salidas"] == {"A": (MIE, "camion"), "D": (MIE, "camion")}


def test_no_sube_un_pedido_sin_opcion_de_camion_ese_dia():
    # B no tiene camión el martes: A solo no paga; el miércoles los dos sí
    pedidos = [_pedido("A", 2000, 1_500_000), _pedido("B", 2000, 1_500_000, dias_camion=(MIE,))]
    resultado = dispatch_when_pays(pedidos, [MAR, MIE], TABLA)
    assert resultado["salidas"] == {"A": (MIE, "camion"), "B": (MIE, "camion")}


def test_lo_que_vence_despues_de_la_ventana_queda_directo_sin_espera():
    # Borde: E llega el miércoles y puede esperar hasta el viernes, pero la ventana termina el miércoles.
    # No se le cobra espera: queda directo el día en que llegó
    pedidos = [_pedido("E", 500, 100_000, llegada=MIE, ultimo_dia=VIE, dias_camion=(MIE, JUE, VIE))]
    resultado = dispatch_when_pays(pedidos, [MAR, MIE], TABLA)
    assert resultado["salidas"] == {"E": (MIE, "directo")}
    assert resultado["castigo"] == pytest.approx(1.0)


# ---------------------------------------------------------------- _lote: una decisión de hoy en todos los futuros

def _futuros():
    # Futuro 0: el martes llega D (2.000 kg) y completa el camión con A. Futuro 1: no llega nada
    d = _pedido("D", 2000, 1_500_000, llegada=MAR, ultimo_dia=JUE, dias_camion=(MAR, MIE, JUE))
    return [[d], []]


def test_esperar_se_juzga_con_lo_que_pasaria_en_cada_futuro():
    conocidos = [_pedido("A", 2000, 1_500_000)]
    costos, retrasos = _lote({"A": "esperar"}, conocidos, _futuros(), LUN, [MAR, MIE], TABLA)
    # Futuro 0: A y D salen el martes en una Turbo. Futuro 1: A sale directo el miércoles, tarde
    assert costos == pytest.approx([2 * 100_000 + 2_100_000, 1_600_000])
    assert retrasos == pytest.approx([(0.5 + 0.5) / 2, 3.0 / 1])


def test_directo_hoy_no_depende_del_futuro():
    conocidos = [_pedido("A", 2000, 1_500_000)]
    costos, retrasos = _lote({"A": "directo"}, conocidos, _futuros(), LUN, [MAR, MIE], TABLA)
    # A sale hoy directo (1,6 M, castigo 1,0); en el futuro 0, D solo no paga camión y queda directo en su borde
    assert costos[1] == pytest.approx(1_600_000) and retrasos[1] == pytest.approx(1.0)
    assert costos[0] == pytest.approx(1_600_000 + 1_600_000)


def test_camion_hoy_compra_el_camion_aunque_no_se_pague():
    # La decisión candidata se ejecuta tal cual: R1 solo decide los días siguientes
    conocidos = [_pedido("A", 2000, 1_500_000)]
    costos, _ = _lote({"A": "camion"}, conocidos, [[]], LUN, [MAR, MIE], TABLA)
    assert costos == pytest.approx([100_000 + 2_100_000])


def test_camion_hoy_sin_opcion_de_camion_es_un_error():
    conocidos = [_pedido("A", 2000, 1_500_000, dias_camion=(MAR,))]
    with pytest.raises(ValueError, match="no puede salir en camión"):
        _lote({"A": "camion"}, conocidos, [[]], LUN, [MAR, MIE], TABLA)


def test_un_resultado_por_futuro():
    conocidos = [_pedido("A", 2000, 1_500_000)]
    costos, retrasos = _lote({"A": "esperar"}, conocidos, [[]] * 7, LUN, [MAR, MIE], TABLA)
    assert len(costos) == len(retrasos) == 7


# ---------------------------------------------------------------- criterio: costo promedio con límite al retraso

def test_si_el_retraso_promedio_cumple_es_el_costo_promedio():
    assert obj_promedio([100.0, 300.0], [0.8, 1.0], epsilon=1.0, penalizacion=1e6) == pytest.approx(200.0)


def test_si_el_retraso_promedio_se_pasa_suma_el_castigo():
    # Retraso promedio 1,2 con ε = 1,0: se pasa 0,2
    assert obj_promedio([100.0, 300.0], [1.0, 1.4], epsilon=1.0, penalizacion=1e6) == pytest.approx(200.0 + 0.2e6)


def test_el_limite_es_sobre_el_promedio_no_sobre_cada_futuro():
    # Un futuro se pasa (1,5) pero el promedio (0,9) cumple: no hay castigo
    assert obj_promedio([100.0, 300.0], [0.3, 1.5], epsilon=1.0, penalizacion=1e6) == pytest.approx(200.0)


def test_el_retraso_puede_medirse_solo_en_los_conocidos():
    # La decisión de hoy responde por los pedidos que decide: futuro 0, A sale en camión con D el martes (0,5);
    # futuro 1, A sale directo el miércoles, tarde (3,0). D no entra en la medida
    conocidos = [_pedido("A", 2000, 1_500_000)]
    _, retrasos = _lote({"A": "esperar"}, conocidos, _futuros(), LUN, [MAR, MIE], TABLA, solo_conocidos=True)
    assert retrasos == pytest.approx([0.5, 3.0])
