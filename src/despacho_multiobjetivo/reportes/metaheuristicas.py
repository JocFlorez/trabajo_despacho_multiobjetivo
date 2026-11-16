"""Salidas de las metaheurísticas: curva de convergencia de una corrida."""
import matplotlib
matplotlib.use("Agg")   # guarda la imagen sin abrir ventanas
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def plot_convergence(historial: pd.DataFrame, epsilon: float, costo_milp: float | None, titulo: str,
                     ruta: str) -> None:
    # Convergencia de una corrida de la tabú (historial de tabu_search, una fila por iteración):
    #   arriba, el costo del plan actual (verde si cumple Z2 <= ε, rojo si se pasa) y del mejor plan que cumple,
    #     con el óptimo del MILP como referencia: la distancia final entre las dos líneas es la brecha
    #   abajo, el Z2 del plan actual frente al límite ε: si oscila alrededor de la línea, el castigo adaptativo
    #     mantiene la búsqueda en el borde, donde está el óptimo
    # costo_milp: None si no hay referencia exacta
    fig, (ax_costo, ax_retraso) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                               gridspec_kw={"height_ratios": [2, 1]})
    iteracion = historial["iteracion"]
    cumple = historial["cumple"].astype(bool)

    # 1. Costo
    ax_costo.scatter(iteracion[cumple], historial["z1"][cumple] / 1e6, s=8, color="seagreen",
                     label="Plan actual (cumple el límite)")
    ax_costo.scatter(iteracion[~cumple], historial["z1"][~cumple] / 1e6, s=8, color="indianred",
                     label="Plan actual (se pasa del límite)")
    ax_costo.step(iteracion, historial["z1_mejor"] / 1e6, where="post", color="darkred", linewidth=2,
                  label="Mejor plan que cumple")
    if costo_milp is not None:
        ax_costo.axhline(costo_milp / 1e6, color="black", linestyle="--", linewidth=1, label="Óptimo del MILP")
    ax_costo.set_ylabel("Costo total (millones de pesos)")
    ax_costo.set_title(titulo)
    ax_costo.grid(alpha=0.3)
    ax_costo.legend()

    # 2. Castigo por retraso frente al límite
    ax_retraso.plot(iteracion, historial["z2"], color="steelblue", linewidth=1, label="Plan actual")
    ax_retraso.axhline(epsilon, color="black", linewidth=1.5, label="Límite ε")
    ax_retraso.set_xlabel("Iteración")
    ax_retraso.xaxis.set_major_locator(MaxNLocator(integer=True))   # iteraciones enteras
    ax_retraso.set_ylabel("Castigo por retraso\n(días equivalentes por pedido)")
    ax_retraso.grid(alpha=0.3)
    ax_retraso.legend()

    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
