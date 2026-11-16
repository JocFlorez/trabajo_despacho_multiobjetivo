"""Parámetros del modelo: todo lo que se ajusta o se sensibiliza está aquí, en un solo lugar.

Los módulos los leen como parametros.NOMBRE en el momento de calcular (no los copian al importar),
así que para un escenario de sensibilidad basta con cambiar el valor aquí o asignarlo en el script,
por ejemplo: parametros.TASA_CASTIGO_RETRASO = 1.

Los valores medidos en el histórico de la empresa (cumplimiento de ENVIA, histograma de días tarde y valor declarado)
no se publican: vienen de data/external/datos_empresa.json (ver datos_empresa.py).
"""
from despacho_multiobjetivo.datos_empresa import DATOS

# ------------------------------------------------------------ cumplimiento de la paquetería (reglas_negocio/servicio.py)
# ENVIA (Colvanes) por sede y región, medido en el histórico ene–mar 2026 (a tiempo = promesa estricta, sábado no hábil)
CUMPLIMIENTO_ENVIA = {(fila["sede"], fila["region"]): fila["valor"] for fila in DATOS["cumplimiento_envia"]}
CUMPLIMIENTO_TRANSPORTADORA = {"SERVIENTREGA": 0.60}   # indicador del autor
CUMPLIMIENTO_TRANSPORTE_MASIVO = 1.0                    # despachos MASIVO (FRACOR, COLTANQUES...); sensibilidad 0,95
CUMPLIMIENTO_ULTIMO_TRAMO = {"SUR_OCCIDENTE": 0.90,     # criterio experto del autor
                             "ANTIOQUIA_CHOCO": 0.85,
                             "COSTA_CARIBE": 0.85}
CUMPLIMIENTO_POR_DEFECTO = 0.85                         # provisional donde no hay dato
CUMPLIMIENTO_CAMION = 1.0                               # sensibilidad 0,95
CUMPLIMIENTO_MANIPULACION = 1.0                         # sensibilidad 0,95

# Días hábiles de retraso de los despachos de ENVIA que llegan tarde, medido en ene–mar 2026 (scripts/medir_cumplimiento.py)
# P(K = k | llegó tarde); 0 = tarde pero en fin de semana o festivo; 4 = "4 o más". Servientrega y regionales usan
# la misma forma, escalada con su cumplimiento
HISTOGRAMA_DIAS_TARDE = {int(dias): p for dias, p in DATOS["histograma_dias_tarde"].items()}

# Castigo por llegar tarde (reglas_negocio/servicio.py): g(k) = ((1 + i)^k − 1) / i, como una tasa de interés:
# cada día adicional de retraso pesa (1 + i) veces el anterior. i = 0 es lineal (g(k) = k)
TASA_CASTIGO_RETRASO = 0.5     # PROVISIONAL: pendiente consultarla con la empresa ("¿cuántos pedidos 1 día tarde = uno 2 días tarde?" → i = respuesta − 2)
DIAS_SUPER_CASTIGO = 4         # desde cuántos días de retraso se aplica el súper castigo
FACTOR_SUPER_CASTIGO = 10      # el castigo de 4 o más días se multiplica por este factor (para que el modelo lo evite)

# ------------------------------------------------------------ tiempos de la ruta consolidada (reglas_negocio/opciones_salida.py)
DIAS_MANIPULACION_HUB = 1      # días hábiles entre la descarga en la sede regional y la entrega a la transportadora final

# ------------------------------------------------------------ costos y promesas (datos/data_processing.py)
FACTOR_BIG_M = 2               # costo de triangulación sin ruta cotizada = FACTOR_BIG_M × la mayor tarifa conocida
PROMESA_REEXPEDIDO = 8         # promesa (días hábiles) cuando ENVIA no cubre el destino: se trata como reexpedido
VALOR_DECLARADO_COTIZACION = DATOS["valor_declarado_cotizacion"]   # valor declarado de la mercancía en el cotizador (celda D7)
