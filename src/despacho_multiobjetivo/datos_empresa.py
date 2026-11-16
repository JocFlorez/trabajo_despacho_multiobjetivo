"""Datos propios de la empresa que no se publican: guías con correcciones manuales, cumplimientos medidos en el
histórico, histograma de días de retraso, valor declarado del cotizador y nombre del archivo de despachos.

Se leen de data/external/datos_empresa.json, que no se versiona (data/ está excluido de git). Si no existe, se usa
datos_empresa_ejemplo.json (junto a este archivo), con el mismo formato y valores inventados, y se avisa.
Variables:
  DATOS: el contenido del JSON como diccionario, con las llaves archivo_despachos, guia_region, guia_ciudad,
    cumplimiento_envia (lista de {sede, region, valor}), histograma_dias_tarde ({días: probabilidad}) y
    valor_declarado_cotizacion
"""
import json
from pathlib import Path

from despacho_multiobjetivo.config import EXTERNAL

RUTA_PRIVADA = EXTERNAL / "datos_empresa.json"
RUTA_EJEMPLO = Path(__file__).parent / "datos_empresa_ejemplo.json"


def load_company_data(ruta_privada: Path = RUTA_PRIVADA) -> dict:
    # El archivo privado si existe; si no, el de ejemplo (valores inventados) con una advertencia
    ruta = ruta_privada if ruta_privada.exists() else RUTA_EJEMPLO
    if ruta == RUTA_EJEMPLO:
        print(f"ADVERTENCIA: no existe {ruta_privada}; se usan los valores de ejemplo (inventados)")
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


DATOS = load_company_data()
