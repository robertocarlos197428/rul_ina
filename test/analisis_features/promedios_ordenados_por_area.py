"""
Script 1: area de la banda (percentiles 25-75) de cada feature, ordenadas
de menor a mayor. Misma normalizacion que promedio_total_suavizado.py.

Salida:
  - areas_ordenadas.csv  (48 features ordenadas por area)
  - areas_ordenadas.png  (un solo grafico, las de area mas chica arriba)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACION
# ============================================================

ARCHIVO_ENTRADA = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed"
    r"\features_48_suavizado.parquet"
)

CARPETA_SALIDA = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina"
    r"\analisis_features\areas_banda"
)

N_PUNTOS = 101  # 0 %, 1 %, ..., 100 %
VIDA_UTIL = np.linspace(0, 100, N_PUNTOS)

# Cuadricula de la imagen: 8 filas x 6 columnas = 48 graficos.
# Se llena por filas: la primera fila tiene las areas mas chicas.
N_FILAS = 8
N_COLUMNAS = 6

DPI = 150

# ============================================================
# FEATURES (las 48)
# ============================================================

BASE = [
    "RMS", "Kurtosis", "Skewness", "CrestFactor",
    "SpectralCentroid", "SpectralStd", "SpectralEntropy",
]

FEATURES = (
    [f"{b}_H" for b in BASE]
    + [f"{b}_V" for b in BASE]
    + ["Pearson_HV", "EnergyRatio_HV"]
    + [f"WPD_Erel_{i}_H" for i in range(16)]
    + [f"WPD_Erel_{i}_V" for i in range(16)]
)


# ============================================================
# NORMALIZACION (igual que promedio_total_suavizado.py)
# ============================================================

def preparar_rodamiento(grupo, feature):
    """Min-Max por rodamiento y remuestreo a 101 puntos de vida."""
    datos = grupo[["time_seconds", feature]].copy()
    datos["time_seconds"] = pd.to_numeric(datos["time_seconds"], errors="coerce")
    datos[feature] = pd.to_numeric(datos[feature], errors="coerce")
    datos = datos.replace([np.inf, -np.inf], np.nan).dropna()

    if len(datos) < 2:
        return None

    datos = datos.sort_values("time_seconds")
    datos = datos.groupby("time_seconds", as_index=False)[feature].mean()

    tiempo = datos["time_seconds"].to_numpy(dtype=float)
    valores = datos[feature].to_numpy(dtype=float)

    if len(tiempo) < 2 or tiempo[-1] <= tiempo[0]:
        return None

    minimo, maximo = np.min(valores), np.max(valores)
    rango = maximo - minimo
    if np.isclose(rango, 0.0):
        normalizados = np.zeros_like(valores)
    else:
        normalizados = (valores - minimo) / rango

    porcentaje = (tiempo - tiempo[0]) / (tiempo[-1] - tiempo[0]) * 100.0
    return np.interp(VIDA_UTIL, porcentaje, normalizados)


def construir_matriz(df, feature):
    """Filas = rodamientos, columnas = % de vida."""
    filas = []
    for _, grupo in df.groupby("bearing_id", sort=True):
        serie = preparar_rodamiento(grupo, feature)
        if serie is not None:
            filas.append(serie)
    if not filas:
        return np.empty((0, N_PUNTOS))
    return np.vstack(filas)


# ============================================================
# AREA DE LA BANDA
# ============================================================

def calcular_banda(matriz):
    mediana = np.median(matriz, axis=0)
    p25 = np.percentile(matriz, 25, axis=0)
    p75 = np.percentile(matriz, 75, axis=0)

    # Integral (trapecios) del ancho de la banda sobre 0-100 %,
    # dividida por 100: ancho promedio de la banda, entre 0 y 1.
    ancho = p75 - p25
    integral = np.sum((ancho[1:] + ancho[:-1]) / 2 * np.diff(VIDA_UTIL))
    area = integral / 100.0
    return mediana, p25, p75, area


# ============================================================
# GRAFICO ORDENADO
# ============================================================

def graficar(resultados, destino):
    fig, axes = plt.subplots(
        N_FILAS, N_COLUMNAS,
        figsize=(4.0 * N_COLUMNAS, 3.2 * N_FILAS),
        sharex=True, sharey=True,
    )

    for ax, r in zip(axes.flat, resultados):
        ax.fill_between(VIDA_UTIL, r["p25"], r["p75"], alpha=0.25)
        ax.plot(VIDA_UTIL, r["mediana"], linewidth=1.5)
        ax.set_title(f"{r['feature']}  |  area {r['area']:.3f}", fontsize=9)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

    # Celdas sobrantes (si hubiera menos de 48 features)
    for ax in list(axes.flat)[len(resultados):]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Vida util (%)", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("Valor normalizado", fontsize=8)

    fig.suptitle(
        "Features ordenadas por area de la banda (P25-P75), "
        "de menor a mayor",
        fontsize=16,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(destino, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {destino}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(f"No se encuentra el archivo:\n{ARCHIVO_ENTRADA}")

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    print("Leyendo archivo Parquet...")
    df = pd.read_parquet(ARCHIVO_ENTRADA)

    faltantes = [c for c in ["bearing_id", "time_seconds", *FEATURES]
                 if c not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas:\n" + "\n".join(faltantes))

    df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce")
    df = df.dropna(subset=["bearing_id", "time_seconds"])

    print(f"Rodamientos: {df['bearing_id'].nunique()} | Features: {len(FEATURES)}")

    resultados = []
    for feature in FEATURES:
        matriz = construir_matriz(df, feature)
        if matriz.shape[0] == 0:
            print(f"Sin datos suficientes: {feature}")
            continue
        mediana, p25, p75, area = calcular_banda(matriz)
        resultados.append({
            "feature": feature, "area": area,
            "mediana": mediana, "p25": p25, "p75": p75,
        })

    # Orden: de menor a mayor area
    resultados.sort(key=lambda r: r["area"])

    tabla = pd.DataFrame({
        "posicion": range(1, len(resultados) + 1),
        "feature": [r["feature"] for r in resultados],
        "area": [r["area"] for r in resultados],
    })
    destino_csv = CARPETA_SALIDA / "areas_ordenadas.csv"
    tabla.to_csv(destino_csv, index=False)
    print(f"Guardado: {destino_csv}")

    graficar(resultados, CARPETA_SALIDA / "areas_ordenadas.png")

    print()
    print(tabla.to_string(index=False))
    print("\nProceso terminado.")


if __name__ == "__main__":
    main()