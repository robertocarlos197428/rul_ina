"""
Script 2: analisis de percentiles de las 48 areas de banda.
Lee areas_ordenadas.csv (salida del script 1), sin separar por eje.

Salida:
  - percentiles_areas.csv      (P0, P10, P25, P50, P75, P90, P100 de las 48 areas)
  - percentil_por_feature.csv  (percentil y cuartil de cada feature)
  - histograma_areas.png       (distribucion de las 48 areas con P25/P50/P75)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION
# ============================================================

CARPETA = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina"
    r"\analisis_features\areas_banda"
)
ARCHIVO_ENTRADA = CARPETA / "areas_ordenadas.csv"

PERCENTILES = [0, 10, 25, 50, 75, 90, 100]
N_BINS = 10
DPI = 150


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(f"No se encuentra el archivo:\n{ARCHIVO_ENTRADA}")

    tabla = pd.read_csv(ARCHIVO_ENTRADA)
    areas = tabla["area"].to_numpy(dtype=float)
    n = len(areas)
    print(f"Features: {n}")

    # --------------------------------------------------------
    # 1. Percentiles de la distribucion de areas
    # --------------------------------------------------------
    valores = np.percentile(areas, PERCENTILES)
    resumen = pd.DataFrame({
        "percentil": [f"P{p}" for p in PERCENTILES],
        "area": valores,
    })
    resumen.to_csv(CARPETA / "percentiles_areas.csv", index=False)

    # --------------------------------------------------------
    # 2. Percentil y cuartil de cada feature
    #    percentil = (posicion - 1) / (n - 1) * 100
    #    (la area mas chica queda en 0 y la mas grande en 100)
    # --------------------------------------------------------
    p25, p50, p75 = np.percentile(areas, [25, 50, 75])

    por_feature = tabla.sort_values("area").reset_index(drop=True).copy()
    por_feature["percentil"] = (
        np.arange(n) / (n - 1) * 100.0 if n > 1 else 0.0
    )
    por_feature["cuartil"] = np.select(
        [por_feature["area"] <= p25,
         por_feature["area"] <= p50,
         por_feature["area"] <= p75],
        ["Q1", "Q2", "Q3"],
        default="Q4",
    )
    por_feature = por_feature[["feature", "area", "percentil", "cuartil"]]
    por_feature.to_csv(CARPETA / "percentil_por_feature.csv", index=False)

    # --------------------------------------------------------
    # 3. Histograma de las 48 areas
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(areas, bins=N_BINS, color="tab:blue", alpha=0.35,
            edgecolor="tab:blue")
    # Cada feature como una marca en la base (para ver los huecos)
    ax.plot(areas, np.zeros(n) - 0.15, "|", color="tab:blue", markersize=10,
            clip_on=False)

    for valor, nombre in [(p25, "P25"), (p50, "P50"), (p75, "P75")]:
        ax.axvline(valor, color="tab:red", linestyle="--", linewidth=1.2)
        ax.text(valor, ax.get_ylim()[1], f" {nombre}\n {valor:.3f}",
                color="tab:red", va="top", fontsize=9)

    ax.set_xlabel("Area de la banda (P25-P75)")
    ax.set_ylabel("N de features")
    ax.set_title(f"Distribucion de las {n} areas de banda")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    destino = CARPETA / "histograma_areas.png"
    fig.savefig(destino, dpi=DPI)
    plt.close(fig)

    print()
    print(resumen.to_string(index=False))
    print()
    print(por_feature["cuartil"].value_counts().sort_index().to_string())
    print(f"\nGuardado en: {CARPETA}")
    print("Proceso terminado.")


if __name__ == "__main__":
    main()
    