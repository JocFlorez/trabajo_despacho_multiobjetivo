"""Salidas de la Etapa 3: métricas de cada plan de la frontera y gráfica del espacio de objetivos."""
import matplotlib
matplotlib.use("Agg")   # guarda la imagen sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd

ETIQUETA_Z2 = "Días de retraso ponderados por pedido"


def plan_metrics(pedidos: pd.DataFrame, opciones: pd.DataFrame, x: dict, u: dict, y: dict,
                 prob_directo: pd.Series, prob_opciones: pd.Series,
                 dias_directo: pd.Series, dias_opciones: pd.Series) -> dict:
    # Qué cambia en la operación con el plan resuelto, en lenguaje natural: servicio esperado (% a tiempo),
    # días hábiles de retraso promedio por pedido (sin castigo), pedidos en camión, espera y camiones
    valor = lambda variable: variable.varValue or 0          # valor de la variable en la solución (None → 0)
    espera = opciones.set_index(["id_pedido", "dia_salida"])["espera"]

    a_tiempo = (sum(prob_directo[i] * valor(variable) for i, variable in u.items())
                + sum(prob_opciones[llave] * valor(variable) for llave, variable in x.items()))
    dias_retraso = (sum(dias_directo[i] * valor(variable) for i, variable in u.items())
                    + sum(dias_opciones[llave] * valor(variable) for llave, variable in x.items()))
    en_camion = sum(valor(variable) for variable in x.values())
    espera_total = sum(espera[llave] * valor(variable) for llave, variable in x.items())

    return {
        "nivel_servicio": a_tiempo / len(pedidos),
        "retraso_promedio_dias": dias_retraso / len(pedidos),
        "pedidos_en_camion": en_camion,
        "espera_promedio": espera_total / en_camion if en_camion else 0,
        "camiones": sum(valor(variable) for variable in y.values()),
    }


def plot_frontier(matriz: pd.DataFrame, puntos: pd.DataFrame, linea_base: dict, titulo: str, ruta: str) -> None:
    # Espacio de objetivos, como en el taller: frontera, ideal, nadir y la operación actual
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(puntos["retraso"], puntos["costo"] / 1e6, "o-", color="darkred", label="Frontera de Pareto")
    ax.scatter(matriz.loc["Ideal", "retraso"], matriz.loc["Ideal", "costo"] / 1e6, marker="*", s=250, color="green", label="Ideal (inalcanzable)")
    ax.scatter(matriz.loc["Nadir", "retraso"], matriz.loc["Nadir", "costo"] / 1e6, marker="v", s=120, color="purple", label="Nadir")
    ax.scatter(linea_base["retraso"], linea_base["costo"] / 1e6, marker="s", s=120, color="gray", label="Operación actual (todo directo)")
    ax.set_xlabel(ETIQUETA_Z2)
    ax.set_ylabel("Costo total (millones de pesos)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def real_terms_table(puntos: pd.DataFrame, peso_total_kg: float) -> pd.DataFrame:
    # Puntos de la frontera en lo que se le muestra a la empresa: % a tiempo, días hábiles reales y flete por kilo.
    # Quedan ordenados de mejor a peor Z2; los días reales no tienen que quedar en orden (Z2 es lo que se optimiza)
    ordenados = puntos.sort_values("retraso")
    return pd.DataFrame({"costo": ordenados["costo"], "ahorro": ordenados["ahorro"], "retraso": ordenados["retraso"],
                         "a_tiempo_pct": ordenados["nivel_servicio"] * 100,
                         "retraso_promedio_dias": ordenados["retraso_promedio_dias"],
                         "camiones": ordenados["camiones"],
                         "flete_por_kg": ordenados["costo"] / peso_total_kg}).reset_index(drop=True)


def plot_frontier_real_terms(puntos: pd.DataFrame, linea_base: dict, titulo: str, ruta: str) -> None:
    # La misma frontera con el servicio en días hábiles reales (izquierda) y en % a tiempo (derecha)
    ordenados = puntos.sort_values("retraso")
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    paneles = [(ejes[0], ordenados["retraso_promedio_dias"], linea_base["retraso_promedio_dias"],
                "Días hábiles reales de retraso por pedido"),
               (ejes[1], ordenados["nivel_servicio"] * 100, linea_base["nivel_servicio"] * 100, "Pedidos a tiempo (%)")]
    for ax, servicio_puntos, servicio_base, etiqueta in paneles:
        ax.plot(servicio_puntos, ordenados["costo"] / 1e6, "o-", color="darkred", label="Frontera de Pareto")
        ax.scatter(servicio_base, linea_base["costo"] / 1e6, marker="s", s=120, color="gray",
                   label="Operación actual (todo directo)", zorder=5)
        ax.set_xlabel(etiqueta)
        ax.grid(alpha=0.3)
    ejes[0].set_ylabel("Costo total (millones de pesos)")
    ejes[0].legend()
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
