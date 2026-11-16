"""Mide en el histórico los parámetros de servicio de ENVIA que usa el modelo.

- Cumplimiento por sede y región: % de despachos entregados a más tardar en la fecha límite
  (promesa estricta, sábado no hábil) → parametros.CUMPLIMIENTO_ENVIA
- Histograma de días hábiles de retraso de los que llegaron tarde → parametros.HISTOGRAMA_DIAS_TARDE

Se corre cuando hay datos nuevos; los valores impresos se copian en data/external/datos_empresa.json
(cumplimiento_envia e histograma_dias_tarde), que no se publica (ver datos_empresa.py).
La medición completa queda en data/processed/medicion_cumplimiento_envia.csv.
"""
import pandas as pd

from despacho_multiobjetivo.config import PROCESSED
from despacho_multiobjetivo.datos import data_processing
from despacho_multiobjetivo.reglas_negocio import calendario as cal
from despacho_multiobjetivo.reglas_negocio import servicio

RUTA_MEDICION = PROCESSED / "medicion_cumplimiento_envia.csv"


def envia_with_dates() -> pd.DataFrame:
    # Despachos de ENVIA con fecha de captura, fecha de entrega y promesa; se descartan los que
    # figuran entregados antes de capturados (error de digitación)
    despachos = data_processing.process_shipments()
    transportadora = despachos["transportadora"].str.upper().str.strip().replace(data_processing.NOMBRE_TRANSPORTADORA)
    envia = despachos[(transportadora == "ENVIA")
                      & despachos["fecha_captura"].notna()
                      & despachos["fecha_entrega"].notna()
                      & despachos["dias_promesa_entrega"].notna()].copy()
    captura = pd.to_datetime(envia["fecha_captura"]).dt.date
    entrega = pd.to_datetime(envia["fecha_entrega"]).dt.date
    return envia[entrega >= captura]


def mark_on_time(envia: pd.DataFrame) -> pd.DataFrame:
    # Agrega la fecha límite (promesa estricta, sábado no hábil) y si llegó a tiempo
    envia = envia.copy()
    captura = pd.to_datetime(envia["fecha_captura"]).dt.date
    envia["fecha_limite"] = [cal.due_date(c, int(p), valor_sabado=0)
                             for c, p in zip(captura, envia["dias_promesa_entrega"])]
    envia["a_tiempo"] = pd.to_datetime(envia["fecha_entrega"]).dt.date <= envia["fecha_limite"]
    return envia


if __name__ == "__main__":
    print("Cargando despachos")
    envia = mark_on_time(envia_with_dates())
    meses = pd.to_datetime(envia["mes"]).dt.strftime("%Y-%m")
    print(f"Despachos de ENVIA con fechas completas: {len(envia):,} · meses {meses.min()} a {meses.max()}")
    print(f"Cumplimiento total: {envia['a_tiempo'].mean():.1%}")
    print(envia.groupby("sede")["a_tiempo"].mean().map("{:.1%}".format).to_string())

    por_region = (envia.groupby(["sede", "region_logistica"])
                  .agg(despachos=("a_tiempo", "size"), cumplimiento=("a_tiempo", "mean"))
                  .reset_index())
    histograma = servicio.late_days_histogram(envia)
    tarde = (~envia["a_tiempo"]).sum()

    print(f"\nDespachos que llegaron tarde: {tarde:,}")
    print((histograma * 100).round(1).rename("% de los tardíos").rename_axis("días hábiles tarde").to_string())

    # Medición completa (para documentar de dónde sale cada parámetro)
    medicion = pd.concat([
        por_region.assign(parametro="CUMPLIMIENTO_ENVIA"),
        pd.DataFrame({"parametro": "HISTOGRAMA_DIAS_TARDE", "dias_tarde": histograma.index,
                      "proporcion": histograma.values, "despachos": (histograma * tarde).round().astype(int).values}),
    ], ignore_index=True)
    medicion.to_csv(RUTA_MEDICION, index=False)

    # Valores listos para copiar en parametros.py
    print("\n# ---- para copiar en src/despacho_multiobjetivo/parametros.py ----")
    print("CUMPLIMIENTO_ENVIA = {")
    for fila in por_region.itertuples():
        print(f'    ("{fila.sede}", "{fila.region_logistica}"): {fila.cumplimiento:.3f},   # {fila.despachos:,} despachos')
    print("}")
    print("HISTOGRAMA_DIAS_TARDE = {" + ", ".join(f"{k}: {p:.3f}" for k, p in histograma.items()) + "}"
          f"   # {tarde:,} despachos tardíos; {len(histograma) - 1} = '{len(histograma) - 1} o más'")
    print(f"\nMedición guardada en {RUTA_MEDICION}")
