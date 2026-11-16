import pandas as pd

from despacho_multiobjetivo import parametros
from despacho_multiobjetivo.reglas_negocio import calendario as cal


# Los cumplimientos, el histograma de días de retraso y el castigo están en parametros.py
# (se leen al calcular, así sirven para la sensibilidad)

def direct_on_time_rate(transportadora: str, tipo_transportadora: str, sede: str, region: str) -> float:
    # Probabilidad de que el despacho directo llegue a tiempo, según quien lo despacha
    if tipo_transportadora == "MASIVO":
        return parametros.CUMPLIMIENTO_TRANSPORTE_MASIVO
    elif transportadora == "ENVIA":
        return parametros.CUMPLIMIENTO_ENVIA.get((sede, region), parametros.CUMPLIMIENTO_POR_DEFECTO)
    else:
        return parametros.CUMPLIMIENTO_TRANSPORTADORA.get(transportadora, parametros.CUMPLIMIENTO_POR_DEFECTO)

def last_mile_on_time_rate(region: str) -> float:
    # Probabilidad de que el último tramo llegue a tiempo, según la región
    return parametros.CUMPLIMIENTO_ULTIMO_TRAMO.get(region, parametros.CUMPLIMIENTO_POR_DEFECTO)

def on_time_probability_direct(pedidos: pd.DataFrame) -> pd.Series:
    # Probabilidad de que el despacho directo llegue a tiempo: el cumplimiento de su transportadora
    probabilidad = pedidos.apply(lambda fila: direct_on_time_rate(fila["transportadora_directo"], fila["tipo_transportadora"],
                                                                  fila["sede"], fila["region_logistica"]), axis=1)
    return probabilidad.set_axis(pedidos["id_pedido"])

def _option_on_time_rates(opciones: pd.DataFrame, pedidos: pd.DataFrame) -> pd.DataFrame:
    # Para cada opción (pedido, día): su retraso planeado y la probabilidad de que la ruta consolidada cumpla
    # P = camión × manipulación × último tramo (el último tramo vale 1 si el traslado llega directo a la sede del hub)
    datos = opciones.merge(pedidos[["id_pedido", "region_logistica", "llega_directo"]],
                           on="id_pedido", how="left", validate="many_to_one")
    ultimo_tramo = datos["region_logistica"].map(last_mile_on_time_rate).where(~datos["llega_directo"].astype(bool), 1)
    datos["cumplimiento"] = parametros.CUMPLIMIENTO_CAMION * parametros.CUMPLIMIENTO_MANIPULACION * ultimo_tramo
    return datos.set_index(["id_pedido", "dia_salida"])[["retraso", "cumplimiento"]]

def delay_penalty(dias: float) -> float:
    # Castigo de llegar 'dias' días hábiles tarde: g(k) = ((1 + i)^k − 1) / i, con i = TASA_CASTIGO_RETRASO
    # (i = 0: lineal, g(k) = k). Desde DIAS_SUPER_CASTIGO días se multiplica por FACTOR_SUPER_CASTIGO
    i = parametros.TASA_CASTIGO_RETRASO
    castigo = dias if i == 0 else ((1 + i) ** dias - 1) / i
    if dias >= parametros.DIAS_SUPER_CASTIGO:
        castigo *= parametros.FACTOR_SUPER_CASTIGO
    return castigo

def expected_delay_penalty(cumplimiento: float, retraso_planeado: float) -> float:
    # Castigo esperado de una ruta: con probabilidad 'cumplimiento' llega con su retraso planeado;
    # si falla, suma K días según el histograma. Una falla suma al menos 1 día: el grupo 0
    # (tarde, pero en fin de semana o festivo) cuenta como 1
    al_cumplir = delay_penalty(retraso_planeado)
    al_fallar = sum(p * delay_penalty(retraso_planeado + max(k, 1))
                    for k, p in parametros.HISTOGRAMA_DIAS_TARDE.items())
    return cumplimiento * al_cumplir + (1 - cumplimiento) * al_fallar

def expected_delay_days(cumplimiento: float, retraso_planeado: float) -> float:
    # Días hábiles de retraso esperados, sin castigo (para reportar en lenguaje natural): mismo histograma,
    # una falla suma al menos 1 día
    al_fallar = sum(p * (retraso_planeado + max(k, 1)) for k, p in parametros.HISTOGRAMA_DIAS_TARDE.items())
    return cumplimiento * retraso_planeado + (1 - cumplimiento) * al_fallar

def expected_delay_direct(pedidos: pd.DataFrame, castigo: bool = True) -> pd.Series:
    # Castigo esperado de un despacho directo: sin retraso planeado; si la transportadora falla, según el histograma.
    # castigo=False: días hábiles de retraso esperados, sin castigo
    esperado = expected_delay_penalty if castigo else expected_delay_days
    return on_time_probability_direct(pedidos).map(lambda cumplimiento: esperado(cumplimiento, 0))

def expected_delay_late_direct(despachos: pd.DataFrame, castigo: bool = True) -> pd.Series:
    # Castigo esperado de un despacho directo que salió después de esperar: los días hábiles que esperó
    # (sábado no hábil, como el retraso de las opciones) son retraso planeado; si la transportadora falla,
    # se suman los días del histograma. Sin espera es el castigo directo de siempre.
    # despachos: las columnas que usa on_time_probability_direct, más dia_llegada y dia_salida
    # castigo=False: días hábiles de retraso esperados, sin castigo
    esperado = expected_delay_penalty if castigo else expected_delay_days
    cumplimiento = on_time_probability_direct(despachos)
    espera = [cal.business_days_between(llegada, salida, valor_sabado=0)
              for llegada, salida in zip(despachos["dia_llegada"], despachos["dia_salida"])]
    return pd.Series([esperado(c, k) for c, k in zip(cumplimiento, espera)], index=cumplimiento.index)

def expected_delay_options(opciones: pd.DataFrame, pedidos: pd.DataFrame, castigo: bool = True) -> pd.Series:
    # Castigo esperado de salir en camión el día t: su retraso planeado y, si la ruta falla, los días del histograma.
    # castigo=False: días hábiles de retraso esperados, sin castigo
    esperado = expected_delay_penalty if castigo else expected_delay_days
    datos = _option_on_time_rates(opciones, pedidos)
    valores = [esperado(c, r) for c, r in zip(datos["cumplimiento"], datos["retraso"])]
    return pd.Series(valores, index=datos.index)

def on_time_probability_options(opciones: pd.DataFrame, pedidos: pd.DataFrame) -> pd.Series:
    # Probabilidad de que la ruta consolidada llegue a tiempo saliendo el día t:
    # el cumplimiento de la ruta si la entrega planeada cumple la promesa; 0 si ya llega tarde
    datos = _option_on_time_rates(opciones, pedidos)
    return datos["cumplimiento"].where(datos["retraso"] == 0, 0)

def late_days_histogram(despachos: pd.DataFrame, max_dias: int = 4) -> pd.Series:
    # Distribución de los días hábiles de retraso de los despachos que llegaron tarde (promesa estricta, sábado no hábil).
    # despachos: columnas fecha_captura, fecha_entrega y dias_promesa_entrega, sin vacíos (se filtran antes).
    # Devuelve P(K = k | llegó tarde) para k = 0, 1, ..., max_dias; el último valor agrupa "max_dias o más".
    # k = 0: llegó tarde, pero en fin de semana o festivo (no suma días hábiles)
    captura = pd.to_datetime(despachos["fecha_captura"]).dt.date
    entrega = pd.to_datetime(despachos["fecha_entrega"]).dt.date
    limite = [cal.due_date(c, int(p), valor_sabado=0) for c, p in zip(captura, despachos["dias_promesa_entrega"])]

    dias_tarde = [cal.business_days_between(l, e, valor_sabado=0) for l, e in zip(limite, entrega) if e > l]
    dias_tarde = pd.Series(dias_tarde, dtype=float).clip(upper=max_dias)
    return dias_tarde.value_counts(normalize=True).reindex(range(max_dias + 1), fill_value=0.0)
