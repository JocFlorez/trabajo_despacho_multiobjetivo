"""Frontera costo–servicio en lo que se le muestra a la empresa: días hábiles reales, % a tiempo y flete por kilo.

No resuelve nada: lee los puntos ya guardados por main_multiobjetivo.py y frontera_puntos_manuales.py y deja solo los
no dominados en (Z1, Z2). La optimización sigue usando los días ponderados; esta es la lectura pedida en la reunión
del 21 de octubre.
Resultados en data/processed/: frontera_{sede}_{mes}_E{espera}_dias_reales.csv y .png
"""
import pandas as pd

from despacho_multiobjetivo.config import SEDE, MES_INICIO, MES_FIN, MODO, ESPERA_MAXIMA, PREFIJO_FRONTERA as PREFIJO
from despacho_multiobjetivo.datos.cargar import load_data
from despacho_multiobjetivo.reglas_negocio import servicio
from despacho_multiobjetivo.reportes.frontera import plot_frontier_real_terms, real_terms_table


if __name__ == "__main__":
    pedidos, _, _, _, _ = load_data(SEDE, MES_INICIO, MES_FIN, MODO)
    peso_total = pedidos["peso_kg"].sum()
    linea_base = {"costo": pedidos["flete_sin_consolidar"].sum(),
                  "retraso": servicio.expected_delay_direct(pedidos).mean(),
                  "retraso_promedio_dias": servicio.expected_delay_direct(pedidos, castigo=False).mean(),
                  "nivel_servicio": servicio.on_time_probability_direct(pedidos).mean()}

    puntos = pd.concat([pd.read_csv(f"{PREFIJO}_puntos.csv"), pd.read_csv(f"{PREFIJO}_manuales.csv")])
    no_dominado = [not ((puntos["retraso"] <= r) & (puntos["costo"] < c - 1)).any()
                   for r, c in zip(puntos["retraso"], puntos["costo"])]
    puntos = puntos[no_dominado]

    tabla = real_terms_table(puntos, peso_total)
    tabla.to_csv(f"{PREFIJO}_dias_reales.csv", index=False)
    plot_frontier_real_terms(puntos, linea_base,
                             f"Frontera en días reales · sede {SEDE} · {MES_INICIO[:7]} · espera máxima {ESPERA_MAXIMA}",
                             f"{PREFIJO}_dias_reales.png")

    pd.set_option("display.width", 200)
    print(f"Operación actual: costo {linea_base['costo']:,.0f} · {linea_base['nivel_servicio']:.1%} a tiempo · "
          f"{linea_base['retraso_promedio_dias']:.3f} días reales · {linea_base['retraso']:.3f} ponderados · "
          f"{linea_base['costo'] / peso_total:,.0f} pesos por kilo")
    print(tabla.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
