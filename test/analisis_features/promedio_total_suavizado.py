
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURACIÓN
# ============================================================

ARCHIVO_ENTRADA = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed"
    r"\features_48_suavizado.parquet"
)

CARPETA_SALIDA = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina"
    r"\analisis_features\suavizado_moderado"
)

N_PUNTOS = 101  # 0 %, 1 %, ..., 100 %

# Resolución de las imágenes
DPI = 180

# ============================================================
# FEATURES POR FAMILIA
# ============================================================

FEATURES_H = [
    "RMS_H",
    "Kurtosis_H",
    "Skewness_H",
    "CrestFactor_H",
    "SpectralCentroid_H",
    "SpectralStd_H",
    "SpectralEntropy_H",
]

FEATURES_V = [
    "RMS_V",
    "Kurtosis_V",
    "Skewness_V",
    "CrestFactor_V",
    "SpectralCentroid_V",
    "SpectralStd_V",
    "SpectralEntropy_V",
]

FEATURES_CRUZADAS = [
    "Pearson_HV",
    "EnergyRatio_HV",
]

FEATURES_WPD_H = [
    f"WPD_Erel_{i}_H" for i in range(16)
]

FEATURES_WPD_V = [
    f"WPD_Erel_{i}_V" for i in range(16)
]

FEATURES = (
    FEATURES_H
    + FEATURES_V
    + FEATURES_CRUZADAS
    + FEATURES_WPD_H
    + FEATURES_WPD_V
)

# Eje temporal común para todos los rodamientos
VIDA_UTIL = np.linspace(0, 100, N_PUNTOS)


# ============================================================
# NORMALIZACIÓN TEMPORAL Y DE AMPLITUD
# ============================================================

def preparar_rodamiento(grupo, feature):
    """
    Devuelve la feature normalizada en amplitud y remuestreada
    a 101 puntos entre el 0 % y el 100 % de vida observada.

    La normalización Min-Max se calcula independientemente
    para cada rodamiento.
    """

    datos = grupo[["time_seconds", feature]].copy()

    datos["time_seconds"] = pd.to_numeric(
        datos["time_seconds"], errors="coerce"
    )
    datos[feature] = pd.to_numeric(
        datos[feature], errors="coerce"
    )

    datos = datos.replace([np.inf, -np.inf], np.nan)
    datos = datos.dropna()

    if len(datos) < 2:
        return None

    # Orden cronológico
    datos = datos.sort_values("time_seconds")

    # Si existen tiempos repetidos, se promedian sus valores
    datos = datos.groupby(
        "time_seconds", as_index=False
    )[feature].mean()

    tiempo = datos["time_seconds"].to_numpy(dtype=float)
    valores = datos[feature].to_numpy(dtype=float)

    if len(tiempo) < 2 or tiempo[-1] <= tiempo[0]:
        return None

    # Normalización Min-Max por rodamiento
    minimo = np.min(valores)
    maximo = np.max(valores)
    rango = maximo - minimo

    if np.isclose(rango, 0.0):
        # Una feature constante no contiene evolución observable
        normalizados = np.zeros_like(valores)
    else:
        normalizados = (valores - minimo) / rango

    # Normalización temporal: porcentaje de vida observada
    porcentaje = (
        (tiempo - tiempo[0]) / (tiempo[-1] - tiempo[0])
    ) * 100.0

    # Interpolación a 101 puntos comunes
    interpolados = np.interp(
        VIDA_UTIL,
        porcentaje,
        normalizados
    )

    return interpolados


# ============================================================
# MATRIZ: RODAMIENTOS x PORCENTAJE DE VIDA
# ============================================================

def construir_matriz(df, feature):
    """
    Cada fila representa un rodamiento.
    Cada columna representa un porcentaje de vida.
    """

    filas = []

    for bearing_id, grupo in df.groupby(
        "bearing_id", sort=True
    ):
        serie = preparar_rodamiento(grupo, feature)

        if serie is not None:
            filas.append(serie)

    if not filas:
        return np.empty((0, N_PUNTOS))

    return np.vstack(filas)


# ============================================================
# GRÁFICO DE MEDIANA Y DISPERSIÓN
# ============================================================

def dibujar_feature(ax, df, feature):
    matriz = construir_matriz(df, feature)

    if matriz.shape[0] == 0:
        ax.text(
            0.5, 0.5,
            "Sin datos suficientes",
            ha="center", va="center",
            transform=ax.transAxes
        )
        ax.set_title(feature, fontsize=9)
        return

    mediana = np.median(matriz, axis=0)
    p25 = np.percentile(matriz, 25, axis=0)
    p75 = np.percentile(matriz, 75, axis=0)

    ax.fill_between(
        VIDA_UTIL,
        p25,
        p75,
        alpha=0.25,
        label="Percentiles 25–75"
    )

    ax.plot(
        VIDA_UTIL,
        mediana,
        linewidth=1.8,
        label="Mediana"
    )

    ax.set_title(feature, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


# ============================================================
# IMAGEN 1: TEMPORALES Y ESPECTRALES
# 7 COLUMNAS x 2 FILAS: H ARRIBA, V ABAJO
# ============================================================

def generar_temporales_espectrales(df):
    fig, axes = plt.subplots(
        2, 7,
        figsize=(24, 7.5),
        sharex=True,
        sharey=True
    )

    for columna, feature in enumerate(FEATURES_H):
        dibujar_feature(
            axes[0, columna], df, feature
        )

    for columna, feature in enumerate(FEATURES_V):
        dibujar_feature(
            axes[1, columna], df, feature
        )

    axes[0, 0].set_ylabel(
        "H\nValor normalizado", fontsize=10
    )
    axes[1, 0].set_ylabel(
        "V\nValor normalizado", fontsize=10
    )

    for ax in axes[1, :]:
        ax.set_xlabel("Vida útil (%)", fontsize=9)

    fig.suptitle(
        "Evolución colectiva de features temporales y espectrales",
        fontsize=15,
        y=1.02
    )

    fig.tight_layout()

    destino = CARPETA_SALIDA / "01_temporales_espectrales.png"

    fig.savefig(
        destino, dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Guardado: {destino}")


# ============================================================
# IMAGEN 2: FEATURES CRUZADAS
# DOS GRÁFICOS, UNO ENCIMA DEL OTRO
# ============================================================

def generar_cruzadas(df):
    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 9),
        sharex=True,
        sharey=True
    )

    for ax, feature in zip(axes, FEATURES_CRUZADAS):
        dibujar_feature(ax, df, feature)
        ax.set_ylabel("Valor normalizado")

    axes[-1].set_xlabel("Vida útil (%)")

    fig.suptitle(
        "Evolución colectiva de features cruzadas H–V",
        fontsize=15
    )

    fig.tight_layout()

    destino = CARPETA_SALIDA / "02_features_cruzadas.png"

    fig.savefig(
        destino, dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Guardado: {destino}")


# ============================================================
# IMÁGENES 3 Y 4: WAVELETS, CUADRÍCULAS 4 x 4
# ============================================================

def generar_wavelets(df, features, eje, numero_imagen):
    fig, axes = plt.subplots(
        4, 4,
        figsize=(16, 13),
        sharex=True,
        sharey=True
    )

    for ax, feature in zip(axes.flat, features):
        dibujar_feature(ax, df, feature)

    for ax in axes[-1, :]:
        ax.set_xlabel("Vida útil (%)", fontsize=9)

    for ax in axes[:, 0]:
        ax.set_ylabel("Valor normalizado", fontsize=9)

    fig.suptitle(
        f"Evolución colectiva de las wavelets — eje {eje}",
        fontsize=15
    )

    fig.tight_layout()

    destino = (
        CARPETA_SALIDA
        / f"{numero_imagen}_wavelets_{eje}.png"
    )

    fig.savefig(
        destino, dpi=DPI, bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Guardado: {destino}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not ARCHIVO_ENTRADA.exists():
        raise FileNotFoundError(
            f"No se encuentra el archivo:\n{ARCHIVO_ENTRADA}"
        )

    CARPETA_SALIDA.mkdir(
        parents=True, exist_ok=True
    )

    print("Leyendo archivo Parquet...")
    df = pd.read_parquet(ARCHIVO_ENTRADA)

    columnas_requeridas = [
        "bearing_id",
        "time_seconds",
        *FEATURES
    ]

    faltantes = [
        col for col in columnas_requeridas
        if col not in df.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan columnas en el archivo:\n"
            + "\n".join(faltantes)
        )

    df["time_seconds"] = pd.to_numeric(
        df["time_seconds"], errors="coerce"
    )

    df = df.dropna(
        subset=["bearing_id", "time_seconds"]
    )

    print(f"Filas leídas: {len(df):,}")
    print(f"Rodamientos: {df['bearing_id'].nunique()}")
    print(f"Features: {len(FEATURES)}")
    print(f"Salida: {CARPETA_SALIDA}")
    print()

    generar_temporales_espectrales(df)
    generar_cruzadas(df)

    generar_wavelets(
        df, FEATURES_WPD_H, "H", "03"
    )

    generar_wavelets(
        df, FEATURES_WPD_V, "V", "04"
    )

    print()
    print("Proceso terminado.")
    print("Se generaron las 4 imágenes PNG.")


if __name__ == "__main__":
    main()
