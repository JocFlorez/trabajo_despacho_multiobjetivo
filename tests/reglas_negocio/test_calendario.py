from datetime import date

import pytest

from despacho_multiobjetivo.reglas_negocio import calendario as cal

# Enero de 2026: el jueves 1 y el lunes 12 (Reyes Magos, trasladado) son festivos.
# Abril de 2026: jueves 2 y viernes 3 son Jueves y Viernes Santo.


# ------------------------------------------------------------ business_day_factor

@pytest.mark.parametrize("fecha, esperado", [
    (date(2026, 1, 14), 1.0),   # miércoles
    (date(2026, 1, 16), 1.0),   # viernes
    (date(2026, 1, 17), 0.5),   # sábado
    (date(2026, 1, 18), 0.0),   # domingo
    (date(2026, 1, 12), 0.0),   # lunes festivo
    (date(2026, 1, 1), 0.0),    # jueves festivo (Año Nuevo)
    (date(2026, 4, 3), 0.0),    # Viernes Santo
])
def test_factor_dia_habil(fecha, esperado):
    assert cal.business_day_factor(fecha) == esperado


# ---------------------------------------------------------- business_days_between

@pytest.mark.parametrize("inicio, fin, esperado", [
    (date(2026, 1, 16), date(2026, 1, 19), 1.5),   # vie → lun: sáb 0,5 + dom 0 + lun 1
    (date(2026, 1, 9), date(2026, 1, 12), 0.5),    # vie → lun festivo: solo cuenta el sábado
    (date(2026, 1, 20), date(2026, 1, 20), 0.0),   # mismo día: la captura no cuenta
    (date(2026, 1, 19), date(2026, 1, 23), 4.0),   # lun → vie de una semana normal
    (date(2026, 1, 16), date(2026, 1, 17), 0.5),   # vie → sáb
])
def test_dias_habiles_entre(inicio, fin, esperado):
    assert cal.business_days_between(inicio, fin) == esperado


# ---------------------------------------------------------------- fecha_limite

@pytest.mark.parametrize("inicio, promesa, esperado", [
    (date(2026, 1, 16), 2, date(2026, 1, 20)),    # vie: lun queda en 1,5, mar en 2,5
    (date(2026, 1, 9), 2, date(2026, 1, 14)),     # vie con lunes festivo en medio
    (date(2026, 1, 14), 0, date(2026, 1, 14)),    # sin promesa: el mismo día
    (date(2026, 1, 19), 2, date(2026, 1, 21)),    # lun: cae justo en 2 el miércoles (sin día extra)
    (date(2026, 1, 16), 1.5, date(2026, 1, 19)),  # promesa fraccionaria
    (date(2026, 1, 16), 0.5, date(2026, 1, 17)),  # medio día: el sábado alcanza
])
def test_fecha_limite(inicio, promesa, esperado):
    assert cal.due_date(inicio, promesa) == esperado


def test_fecha_limite_coincide_con_dias_habiles_entre():
    # El acumulado hasta la fecha límite alcanza la promesa, y el del día anterior no
    inicio, promesa = date(2026, 1, 9), 3
    limite = cal.due_date(inicio, promesa)
    assert cal.business_days_between(inicio, limite) >= promesa
    assert cal.business_days_between(inicio, date.fromordinal(limite.toordinal() - 1)) < promesa


# ---------------------------------------------------------------- unloading_day

@pytest.mark.parametrize("salida, transito, esperado", [
    (date(2026, 1, 15), 2, date(2026, 1, 17)),    # jue + 2 = sáb → se descarga el sábado
    (date(2026, 1, 9), 1, date(2026, 1, 10)),     # vie + 1 = sáb (aunque el lunes sea festivo)
    (date(2026, 1, 19), 1, date(2026, 1, 20)),    # llega en día hábil
    (date(2026, 1, 16), 2, date(2026, 1, 19)),    # vie + 2 = dom → lunes
    (date(2026, 4, 1), 1, date(2026, 4, 4)),      # llega Jueves Santo → Viernes Santo también festivo → sábado
    (date(2026, 1, 20), 0, date(2026, 1, 20)),    # sin tránsito en día hábil
])
def test_dia_descarga(salida, transito, esperado):
    assert cal.unloading_day(salida, transito) == esperado


def test_dia_descarga_nunca_es_domingo_ni_festivo():
    salida = date(2026, 1, 1)
    for dias in range(0, 60):
        for transito in (1, 2):
            descarga = cal.unloading_day(date.fromordinal(salida.toordinal() + dias), transito)
            assert cal.business_day_factor(descarga) > 0   # puede ser sábado (0,5) o día hábil (1)


# ------------------------------------------- sábado no hábil (medición del servicio)

def test_sabado_vale_cero_para_medir_el_servicio():
    sabado = date(2026, 1, 17)
    assert cal.business_day_factor(sabado) == 0.5
    assert cal.business_day_factor(sabado, valor_sabado=0) == 0.0


def test_dias_habiles_entre_viernes_y_lunes_segun_el_sabado():
    viernes, lunes = date(2026, 1, 16), date(2026, 1, 19)
    assert cal.business_days_between(viernes, lunes) == 1.5
    assert cal.business_days_between(viernes, lunes, valor_sabado=0) == 1.0


def test_fecha_limite_estricta_no_cuenta_los_sabados():
    # Promesa de 6 días desde el viernes 16: con sábados de 0,5 se completa el sábado 24;
    # sin sábados hay que esperar al lunes 26
    viernes = date(2026, 1, 16)
    assert cal.due_date(viernes, 6) == date(2026, 1, 24)
    assert cal.due_date(viernes, 6, valor_sabado=0) == date(2026, 1, 26)
