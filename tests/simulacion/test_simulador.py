"""Simulador de horizonte rodante (fase 2 del Monte Carlo): recorrer los días, decidir y ejecutar solo lo de hoy."""
from datetime import date

import pandas as pd
import pytest

from despacho_multiobjetivo.metodos.metaheuristicas.evaluador import evaluate_plan
from despacho_multiobjetivo.reglas_negocio import servicio
from despacho_multiobjetivo.reglas_negocio.opciones_salida import possible_departure_days
from despacho_multiobjetivo.simulacion.simulador import (apply_decision, costo_escenario, decide_all_direct,
                                                       executed_plan, fill_trucks_today, known_orders,
                                                       operation_days, simular_escenario)
from tests.metodos import instancia_pequena as pequena

DOM = date(2026, 2, 1)
LUN = date(2026, 2, 2)
MAR = date(2026, 2, 3)
MIE = date(2026, 2, 4)
VIE = date(2026, 2, 6)
COLUMNAS_REGISTRO = ["id_pedido", "dia_llegada", "dia_salida", "modo", "fraccion"]


def _pedidos():
    # A y B llegan el lunes, C el martes y D se captura el domingo (entra a operación el lunes)
    return pd.DataFrame({"id_pedido": ["A", "B", "C", "D"],
                         "fecha_captura": pd.to_datetime([LUN, LUN, MAR, DOM]),
                         "peso_kg": [100.0, 200.0, 300.0, 400.0]})


def _conocidos(ids, dia_llegada, ultimo_dia):
    return pd.DataFrame({"id_pedido": ids, "dia_llegada": dia_llegada, "ultimo_dia": ultimo_dia,
                         "fraccion": [1.0] * len(ids), "peso_kg": [100.0] * len(ids)})


# ---------------------------------------------------------------- días de operación

def test_los_dias_cubren_todas_las_salidas_posibles_de_los_pedidos():
    esperados = sorted({dia for captura in (LUN, MAR) for dia in possible_departure_days(captura, 1)})
    pedidos = _pedidos().iloc[:3]   # A, B (lunes) y C (martes)
    assert operation_days(pedidos, espera_maxima=1) == esperados


def test_el_horizonte_se_extiende_despues_del_ultimo_pedido():
    # Lo que llega el último día puede esperar: el horizonte va hasta su última salida posible
    pedidos = pd.DataFrame({"id_pedido": ["Z"], "fecha_captura": pd.to_datetime([VIE])})
    dias = operation_days(pedidos, espera_maxima=3)
    assert dias == possible_departure_days(VIE, 3)
    assert dias[-1] > VIE


def test_no_hay_salidas_en_domingo():
    pedidos = pd.DataFrame({"id_pedido": ["Z"], "fecha_captura": pd.to_datetime([VIE])})
    assert all(dia.weekday() != 6 for dia in operation_days(pedidos, espera_maxima=3))


def test_lo_capturado_en_domingo_empieza_el_lunes():
    pedidos = pd.DataFrame({"id_pedido": ["D"], "fecha_captura": pd.to_datetime([DOM])})
    assert operation_days(pedidos, espera_maxima=0) == [LUN]


def test_sin_pedidos_no_hay_dias():
    assert operation_days(_pedidos().iloc[:0], espera_maxima=3) == []


# ---------------------------------------------------------------- pedidos conocidos

def test_se_conoce_lo_que_llega_hoy_mas_lo_que_sigue_esperando():
    pedidos = _pedidos().assign(dia_llegada=[LUN, LUN, MAR, LUN])
    pendientes = pedidos[pedidos["id_pedido"] == "A"]
    conocidos = known_orders(pedidos, pendientes, MAR)
    assert sorted(conocidos["id_pedido"]) == ["A", "C"]


def test_no_se_conoce_lo_que_llega_otro_dia():
    pedidos = _pedidos().assign(dia_llegada=[LUN, LUN, MAR, LUN])
    conocidos = known_orders(pedidos, pedidos.iloc[:0], LUN)
    assert sorted(conocidos["id_pedido"]) == ["A", "B", "D"]


def test_conocer_no_modifica_lo_que_recibe():
    pedidos = _pedidos().assign(dia_llegada=[LUN, LUN, MAR, LUN])
    pendientes = pedidos.iloc[:1].copy()
    antes_pedidos, antes_pendientes = pedidos.copy(), pendientes.copy()
    known_orders(pedidos, pendientes, MAR)
    pd.testing.assert_frame_equal(pedidos, antes_pedidos)
    pd.testing.assert_frame_equal(pendientes, antes_pendientes)


# ---------------------------------------------------------------- política de prueba: todo directo

def test_todo_directo_manda_directo_a_todos_los_conocidos():
    conocidos = _conocidos(["A", "B"], LUN, MIE)
    assert decide_all_direct(conocidos, LUN) == {"A": "directo", "B": "directo"}


def test_todo_directo_sin_conocidos_no_decide_nada():
    assert decide_all_direct(_conocidos([], LUN, MIE), LUN) == {}


# ---------------------------------------------------------------- ejecutar solo lo de hoy

def test_sale_lo_directo_y_lo_de_camion_y_espera_el_resto():
    conocidos = _conocidos(["A", "B", "C"], LUN, MIE)
    salen, pendientes = apply_decision(conocidos, {"A": "directo", "B": "camion", "C": "esperar"}, LUN)
    assert list(salen.columns) == COLUMNAS_REGISTRO
    assert salen.set_index("id_pedido")["modo"].to_dict() == {"A": "directo", "B": "camion"}
    assert (salen["dia_salida"] == LUN).all()
    assert pendientes["id_pedido"].tolist() == ["C"]


def test_lo_pendiente_conserva_sus_datos():
    conocidos = _conocidos(["C"], LUN, MIE)
    _, pendientes = apply_decision(conocidos, {"C": "esperar"}, LUN)
    assert pendientes.iloc[0]["dia_llegada"] == LUN and pendientes.iloc[0]["ultimo_dia"] == MIE


def test_sale_la_fraccion_que_estaba_pendiente():
    # Un traslado al que ya le subió el 75 % a un camión solo tiene pendiente el 25 %
    conocidos = _conocidos(["T"], LUN, MIE).assign(fraccion=[0.25])
    salen, _ = apply_decision(conocidos, {"T": "directo"}, MAR)
    assert salen.iloc[0]["fraccion"] == pytest.approx(0.25)


def test_esperar_en_el_ultimo_dia_sale_directo_y_avisa(capsys):
    conocidos = _conocidos(["C"], LUN, MIE)
    salen, pendientes = apply_decision(conocidos, {"C": "esperar"}, MIE)
    assert salen.set_index("id_pedido")["modo"].to_dict() == {"C": "directo"}
    assert pendientes.empty
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_cada_pedido_conocido_necesita_una_decision():
    with pytest.raises(ValueError, match="sin decisión"):
        apply_decision(_conocidos(["A", "B"], LUN, MIE), {"A": "directo"}, LUN)


def test_no_se_puede_decidir_sobre_un_pedido_que_no_se_conoce():
    with pytest.raises(ValueError, match="no se conoce"):
        apply_decision(_conocidos(["A"], LUN, MIE), {"A": "directo", "Z": "directo"}, LUN)


def test_decision_no_valida_es_un_error():
    with pytest.raises(ValueError, match="decisión no válida"):
        apply_decision(_conocidos(["A"], LUN, MIE), {"A": "mañana"}, LUN)


# ---------------------------------------------------------------- ciclo completo

def test_todo_directo_cada_pedido_sale_el_dia_que_llega():
    registro = simular_escenario(_pedidos(), decide_all_direct, espera_maxima=3)
    assert list(registro.columns) == COLUMNAS_REGISTRO
    assert sorted(registro["id_pedido"]) == ["A", "B", "C", "D"]          # ninguno se pierde ni se repite
    assert (registro["dia_salida"] == registro["dia_llegada"]).all()
    assert (registro["modo"] == "directo").all()
    assert registro.set_index("id_pedido").loc["D", "dia_llegada"] == LUN  # domingo → lunes


def test_una_politica_que_espera_un_dia_despacha_al_dia_siguiente():
    def esperar_un_dia(conocidos, dia):
        return {fila.id_pedido: ("esperar" if fila.dia_llegada == dia else "camion") for fila in conocidos.itertuples()}

    registro = simular_escenario(_pedidos(), esperar_un_dia, espera_maxima=3).set_index("id_pedido")
    assert registro.loc["A", "dia_salida"] == MAR and registro.loc["A", "modo"] == "camion"
    assert registro.loc["C", "dia_salida"] == MIE


def test_una_politica_que_siempre_espera_termina_despachando_todo(capsys):
    registro = simular_escenario(_pedidos(), lambda conocidos, dia: dict.fromkeys(conocidos["id_pedido"], "esperar"),
                                   espera_maxima=1).set_index("id_pedido")
    assert sorted(registro.index) == ["A", "B", "C", "D"]
    for id_pedido, fila in registro.iterrows():
        assert fila["dia_salida"] == possible_departure_days(fila["dia_llegada"], 1)[-1]
        assert fila["modo"] == "directo"
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_los_pedidos_sin_fecha_quedan_fuera_con_aviso(capsys):
    pedidos = pd.concat([_pedidos(), pd.DataFrame({"id_pedido": ["X"], "fecha_captura": [pd.NaT], "peso_kg": [1.0]})],
                        ignore_index=True)
    registro = simular_escenario(pedidos, decide_all_direct, espera_maxima=3)
    assert "X" not in set(registro["id_pedido"])
    assert "ADVERTENCIA" in capsys.readouterr().out


def test_el_ciclo_no_modifica_los_pedidos():
    pedidos = _pedidos()
    antes = pedidos.copy()
    simular_escenario(pedidos, decide_all_direct, espera_maxima=3)
    pd.testing.assert_frame_equal(pedidos, antes)


# ---------------------------------------------------------------- de lo ejecutado al plan del evaluador

def test_el_plan_ejecutado_solo_lleva_lo_que_salio_en_camion():
    registro = pd.DataFrame({"id_pedido": ["A", "B", "C"], "dia_llegada": [LUN, LUN, MAR],
                             "dia_salida": [LUN, MAR, MAR], "modo": ["directo", "camion", "camion"],
                             "fraccion": [1.0, 1.0, 1.0]})
    assert executed_plan(registro) == {"B": MAR, "C": MAR}


def test_todo_directo_es_el_plan_vacio():
    registro = simular_escenario(_pedidos(), decide_all_direct, espera_maxima=3)
    assert executed_plan(registro) == {}


# ---------------------------------------------------------------- relleno con traslados (fase 3, opción B2)
# Después de la decisión de cada día, el espacio libre de los camiones que salen hoy se llena con fracciones de
# traslados presentes ese día: los que hoy salen directo y los que siguen esperando. Se llena primero el que más
# ahorra por kilo; nunca se compra un camión solo para rellenar. La parte que sube sale hoy en camión; el resto
# sale directo hoy (si iba directo) o sigue esperando (si esperaba).
# Instancia pequeña: C_0 (3.000 kg, lunes o martes), C_1 (1.000 kg, martes), T_0 (traslado, 2.000 kg, lunes o
# miércoles) y O_0 (Oriente, siempre directo). Turbo de 4.500 kg; C_0 solo en un Turbo deja 1.500 kg libres.

P_LUN, P_MAR, P_MIE = pequena.LUN, pequena.MAR, pequena.MIE


def _salen(filas):
    return pd.DataFrame(filas, columns=COLUMNAS_REGISTRO)


def _pendientes(filas):
    return pd.DataFrame(filas, columns=["id_pedido", "dia_llegada", "ultimo_dia", "fraccion"])


def _fracciones(tabla, columna="modo"):
    return {(fila.id_pedido, getattr(fila, columna)): round(fila.fraccion, 9) for fila in tabla.itertuples()}


def test_el_traslado_que_sale_directo_hoy_llena_lo_que_cabe_y_el_resto_sale_directo():
    salen = _salen([("C_0", P_LUN, P_LUN, "camion", 1.0), ("T_0", P_LUN, P_LUN, "directo", 1.0)])
    salen, pendientes = fill_trucks_today(salen, _pendientes([]), P_LUN, pequena.instancia())
    assert _fracciones(salen) == {("C_0", "camion"): 1.0, ("T_0", "camion"): 0.75, ("T_0", "directo"): 0.25}
    assert (salen["dia_salida"] == P_LUN).all()
    assert pendientes.empty


def test_el_traslado_que_espera_llena_lo_que_cabe_y_el_resto_sigue_esperando():
    salen = _salen([("C_0", P_LUN, P_LUN, "camion", 1.0)])
    pendientes = _pendientes([("T_0", P_LUN, P_MIE, 1.0)])
    salen, pendientes = fill_trucks_today(salen, pendientes, P_LUN, pequena.instancia())
    assert _fracciones(salen) == {("C_0", "camion"): 1.0, ("T_0", "camion"): 0.75}
    assert pendientes["id_pedido"].tolist() == ["T_0"] and pendientes.iloc[0]["fraccion"] == pytest.approx(0.25)


def test_lo_que_queda_pendiente_de_un_traslado_puede_subir_completo():
    # Solo quedan 800 kg de T_0 (40 %) y caben 1.500: sube todo lo que queda y deja de estar pendiente
    salen = _salen([("C_0", P_LUN, P_LUN, "camion", 1.0)])
    salen, pendientes = fill_trucks_today(salen, _pendientes([("T_0", P_LUN, P_MIE, 0.4)]), P_LUN, pequena.instancia())
    assert _fracciones(salen)[("T_0", "camion")] == pytest.approx(0.4)
    assert pendientes.empty


def test_sin_camion_hoy_no_se_rellena():
    salen = _salen([("T_0", P_LUN, P_LUN, "directo", 1.0)])
    salen_despues, _ = fill_trucks_today(salen, _pendientes([]), P_LUN, pequena.instancia())
    assert _fracciones(salen_despues) == {("T_0", "directo"): 1.0}


def test_los_clientes_no_se_parten_para_rellenar():
    # Martes: C_1 en un Turbo deja 3.500 kg libres, pero C_0 es un cliente y no se puede partir
    salen = _salen([("C_1", P_MAR, P_MAR, "camion", 1.0), ("C_0", P_LUN, P_MAR, "directo", 1.0)])
    salen_despues, _ = fill_trucks_today(salen, _pendientes([]), P_MAR, pequena.instancia())
    assert _fracciones(salen_despues) == {("C_1", "camion"): 1.0, ("C_0", "directo"): 1.0}


def test_un_traslado_sin_opcion_ese_dia_no_rellena():
    # T_0 puede salir lunes o miércoles: el martes no puede subir aunque haya espacio
    salen = _salen([("C_1", P_MAR, P_MAR, "camion", 1.0), ("T_0", P_LUN, P_MAR, "directo", 1.0)])
    salen_despues, _ = fill_trucks_today(salen, _pendientes([]), P_MAR, pequena.instancia())
    assert _fracciones(salen_despues) == {("C_1", "camion"): 1.0, ("T_0", "directo"): 1.0}


@pytest.mark.parametrize("filtrar, rellena", [(True, False), (False, True)])
def test_con_el_filtro_de_retraso_no_rellena_si_en_camion_llega_peor(filtrar, rellena):
    instancia = pequena.instancia()
    instancia["opciones"].loc[("T_0", P_LUN), "castigo"] = 5.0   # en camión llegaría peor que directo
    salen = _salen([("C_0", P_LUN, P_LUN, "camion", 1.0), ("T_0", P_LUN, P_LUN, "directo", 1.0)])
    salen_despues, _ = fill_trucks_today(salen, _pendientes([]), P_LUN, instancia, filtrar_retraso=filtrar)
    assert (("T_0", "camion") in _fracciones(salen_despues)) is rellena


def test_rellenar_no_modifica_lo_que_recibe():
    salen = _salen([("C_0", P_LUN, P_LUN, "camion", 1.0)])
    pendientes = _pendientes([("T_0", P_LUN, P_MIE, 1.0)])
    antes_salen, antes_pendientes = salen.copy(), pendientes.copy()
    fill_trucks_today(salen, pendientes, P_LUN, pequena.instancia())
    pd.testing.assert_frame_equal(salen, antes_salen)
    pd.testing.assert_frame_equal(pendientes, antes_pendientes)


# ---------------------------------------------------------------- el ciclo con relleno

def _pedidos_pequenos():
    pedidos, _ = pequena.datos()
    return pedidos.assign(fecha_captura=pd.to_datetime([P_LUN, P_MAR, P_LUN, P_LUN]))   # C_0, C_1, T_0, O_0


def test_con_la_instancia_el_ciclo_rellena_y_el_traslado_termina_completo(capsys):
    def politica(conocidos, dia):
        return {fila.id_pedido: ("camion" if fila.id_pedido == "C_0" and dia == P_LUN
                                 else "esperar" if fila.id_pedido == "T_0" else "directo")
                for fila in conocidos.itertuples()}

    registro = simular_escenario(_pedidos_pequenos(), politica, espera_maxima=3, instancia=pequena.instancia())
    assert registro.groupby("id_pedido")["fraccion"].sum().to_dict() == pytest.approx(
        {"C_0": 1.0, "C_1": 1.0, "T_0": 1.0, "O_0": 1.0})                     # cada pedido sale completo
    t_0 = registro[registro["id_pedido"] == "T_0"].set_index("modo")
    assert t_0.loc["camion", "fraccion"] == pytest.approx(0.75) and t_0.loc["camion", "dia_salida"] == P_LUN
    assert t_0.loc["directo", "fraccion"] == pytest.approx(0.25)              # lo que quedó, en su último día


def test_sin_la_instancia_no_hay_relleno():
    registro = simular_escenario(_pedidos_pequenos(), decide_all_direct, espera_maxima=3)
    assert (registro["fraccion"] == 1.0).all()


# ---------------------------------------------------------------- costo y servicio de lo ejecutado

def _registro_pequeno(salida_o_0=P_LUN):
    # C_0 en camión el lunes; T_0: 75 % rellena ese camión y 25 % sale directo; C_1 y O_0 directo
    return _salen([("C_0", P_LUN, P_LUN, "camion", 1.0), ("T_0", P_LUN, P_LUN, "camion", 0.75),
                   ("T_0", P_LUN, P_LUN, "directo", 0.25), ("C_1", P_MAR, P_MAR, "directo", 1.0),
                   ("O_0", P_LUN, salida_o_0, "directo", 1.0)])


def test_todo_directo_sin_espera_cuesta_lo_de_la_operacion_actual():
    pedidos, instancia = _pedidos_pequenos(), pequena.instancia()
    registro = simular_escenario(pedidos, decide_all_direct, espera_maxima=3)
    costo, actual = costo_escenario(registro, pedidos, instancia), evaluate_plan({}, instancia)
    assert (costo["z1"], costo["z2"]) == pytest.approx((actual["z1"], actual["z2"]), rel=1e-12)


def test_sin_esperas_cuesta_lo_mismo_que_el_evaluador_con_su_relleno():
    # evaluate_plan({"C_0": lunes}) compra el mismo Turbo y rellena el mismo 75 % de T_0
    pedidos, instancia = _pedidos_pequenos(), pequena.instancia()
    costo, plan = costo_escenario(_registro_pequeno(), pedidos, instancia), evaluate_plan({"C_0": P_LUN}, instancia)
    assert (costo["z1"], costo["z2"]) == pytest.approx((plan["z1"], plan["z2"]), rel=1e-12)


def test_el_directo_tardio_suma_su_castigo_pero_no_cambia_el_costo():
    # O_0 esperó de lunes a miércoles y salió directo: mismo flete, más castigo
    pedidos, instancia = _pedidos_pequenos(), pequena.instancia()
    a_tiempo = costo_escenario(_registro_pequeno(), pedidos, instancia)
    tardio = costo_escenario(_registro_pequeno(salida_o_0=P_MIE), pedidos, instancia)
    o_0 = pedidos[pedidos["id_pedido"] == "O_0"].assign(dia_llegada=[P_LUN], dia_salida=[P_MIE])
    extra = (servicio.expected_delay_late_direct(o_0)["O_0"] - instancia["pedidos"].loc["O_0", "castigo_directo"]) / 4
    assert tardio["z1"] == pytest.approx(a_tiempo["z1"])
    assert tardio["z2"] == pytest.approx(a_tiempo["z2"] + extra)


def test_lo_que_no_paso_por_el_simulador_cuenta_como_directo():
    # Un pedido sin fecha no entra a la simulación: va directo, como en el modelo
    pedidos, instancia = _pedidos_pequenos(), pequena.instancia()
    sin_o_0 = _registro_pequeno()[lambda tabla: tabla["id_pedido"] != "O_0"]
    con_o_0 = costo_escenario(_registro_pequeno(), pedidos, instancia)
    assert costo_escenario(sin_o_0, pedidos, instancia) == pytest.approx(con_o_0)


def test_el_plan_ejecutado_lleva_las_fracciones_de_los_traslados_partidos():
    assert executed_plan(_registro_pequeno()) == {"C_0": P_LUN, "T_0": {P_LUN: 0.75}}
