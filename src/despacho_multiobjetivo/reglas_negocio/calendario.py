import holidays
from datetime import date
from datetime import timedelta
from typing import Literal

# Festivos de Colombia; el rango cubre con holgura los datos (2026) y corridas futuras
FESTIVOS = holidays.country_holidays("CO", years=range(2020, 2041))

def business_day_factor(fecha: date, valor_sabado: Literal[0.5,0] = 0.5) -> float:
    dia_semana = fecha.weekday()

    if fecha in FESTIVOS or dia_semana==6:
        return 0.0
    elif dia_semana == 5:
        return valor_sabado
    else:
        return 1.0

def business_days_between(inicio: date, fin: date, valor_sabado: Literal[0.5,0] = 0.5) -> float:
    rango_dias = (fin - inicio).days
    dias_habiles = 0.0

    for dias in range(1, rango_dias+1):
        dia_actual = inicio + timedelta(days=dias)
        dias_habiles += business_day_factor(dia_actual, valor_sabado)

    return dias_habiles

def due_date(inicio: date, promesa: int, valor_sabado: Literal[0.5,0] = 0.5) -> date:
    dias_habiles = 0.0
    fecha = inicio

    while dias_habiles < promesa:
        fecha = fecha + timedelta(days=1)
        dias_habiles += business_day_factor(fecha, valor_sabado)

    return fecha

def unloading_day(salida: date, dias_transito: int) -> date:
    # El camión descarga en la sede regional cualquier día con operación, incluido el sábado;
    # si llega domingo o festivo, descarga el siguiente día con operación
    llegada = salida + timedelta(days=dias_transito)

    while business_day_factor(llegada) == 0:
        llegada = llegada + timedelta(days=1)

    return llegada

