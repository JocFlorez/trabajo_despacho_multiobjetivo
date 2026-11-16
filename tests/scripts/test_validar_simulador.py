"""Validación del simulador: la política "todo directo" debe reproducir la operación actual."""
from scripts.validar_simulador import reference_name, validate
from tests.metodos.instancia_pequena import LUN, MAR
from tests.metodos.instancia_pequena import datos as _datos
from tests.metodos.instancia_pequena import instancia as _instancia


def _pedidos():
    pedidos, _ = _datos()
    return pedidos.assign(fecha_captura=[LUN, LUN, MAR, LUN])


def test_todo_directo_pasa_todas_las_revisiones():
    revisiones = validate(_pedidos(), _instancia(), espera_maxima=3)
    assert [fila["revision"] for fila in revisiones] == ["costo (Z1)", "días de retraso ponderados (Z2)",
                                                         "cada pedido despachado completo",
                                                         "todo sale directo el día que llega"]
    assert all(fila["cumple"] for fila in revisiones)


def test_la_referencia_es_la_operacion_actual():
    pedidos, instancia = _pedidos(), _instancia()
    costo = next(fila for fila in validate(pedidos, instancia, espera_maxima=3) if fila["revision"] == "costo (Z1)")
    assert costo["esperado"] == pedidos["flete_sin_consolidar"].sum()


def test_la_referencia_dice_de_donde_sale_el_costo_directo():
    # Modo histórico: lo pagado (la operación actual); modo operación: todo directo con el cotizador
    assert reference_name("historico") == "operación actual (lo pagado)"
    assert reference_name("operacion") == "todo directo cotizado"
