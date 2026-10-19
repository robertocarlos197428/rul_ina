"""
Script 3: las 17 features seleccionadas (corte de area 0,31), agrupadas por
forma de la mediana a lo largo de la vida:
  - Suben (como RMS)
  - Bajan (como Erel_5_H)
  - Suben y bajan (cambian de sentido)
Dentro de cada grupo van ordenadas de menor a mayor area de banda.

Una fila por grupo. Linea = mediana, sombra = percentiles 25-75.

Salida:
  - grupos_17_features.png
  - grupos_17_features.csv
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
DPI = 150

# ------------------------------------------------------------
# GRUPOS (clasificacion segun lo que se ve en los graficos
# colectivos; se puede editar a mano moviendo features de lista)
# ------------------------------------------------------------
GRUPOS = {
    "Suben": [
        "RMS_H", "RMS_V",
        "Kurtosis_H", "Kurtosis_V",
        "CrestFactor_H", "CrestFactor_V",
        "SpectralStd_H",
    ],
    "Bajan": [
        "WPD_Erel_3_H", "WPD_Erel_4_H", "WPD_Erel_5_H", "WPD_Erel_6_H",
        "Skewness_H",
        "WPD_Erel_5_V",
    ],
    "Suben y bajan (cambian de sentido)": [
        "WPD_Erel_3_V", "WPD_Erel_4_V",     # bajan y suben al final
        "WPD_Erel_11_V", "WPD_Erel_12_V",   # suben y caen al final
    ],
}


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


def calcular_banda(matriz):
    mediana = np.median(matriz, axis=0)
    p25 = np.percentile(matriz, 25, axis=0)
    p75 = np.percentile(matriz, 75, axis=0)
    ancho = p75 - p25
    integral = np.sum((ancho[1:] + ancho[:-1]) / 2 * np.diff(VIDA_UTIL))
    return mediana, p25, p75, integral / 100.0


# ============================================================
# GRAFICO: UNA FILA POR GRUPO
# ============================================================

def graficar(grupos_ordenados, destino):
    n_filas = len(grupos_ordenados)
    n_cols = max(len(items) for items in grupos_ordenados.values())

    fig, axes = plt.subplots(
        n_filas, n_cols,
        figsize=(4.0 * n_cols, 3.4 * n_filas),
        sharex=True, sharey=True,
        squeeze=False,
    )

    for r, (nombre, items) in enumerate(grupos_ordenados.items()):
        for c in range(n_cols):
            ax = axes[r, c]
            if c >= len(items):
                ax.axis("off")
                continue

            it = items[c]
            ax.fill_between(VIDA_UTIL, it["p25"], it["p75"], alpha=0.25)
            ax.plot(VIDA_UTIL, it["mediana"], linewidth=1.6)
            ax.set_title(f"{it['feature']}  |  area {it['area']:.3f}", fontsize=9)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=7)

            if c == 0:
                ax.set_ylabel(f"{nombre}\nValor normalizado", fontsize=9)
            if r == n_filas - 1:
                ax.set_xlabel("Vida util (%)", fontsize=8)

    for c in range(n_cols):
        axes[-1, c].set_xlabel("Vida util (%)", fontsize=8)

    fig.suptitle(
        "Las 17 features seleccionadas, agrupadas por forma "
        "(dentro de cada grupo, de menor a mayor area)",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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

    df = pd.read_parquet(ARCHIVO_ENTRADA)
    df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce")
    df = df.dropna(subset=["bearing_id", "time_seconds"])

    todas = [f for lista in GRUPOS.values() for f in lista]
    faltantes = [c for c in todas if c not in df.columns]
    if faltantes:
        raise ValueError("Faltan columnas:\n" + "\n".join(faltantes))

    print(f"Rodamientos: {df['bearing_id'].nunique()} | Features: {len(todas)}")

    grupos_ordenados = {}
    filas_csv = []

    for nombre, lista in GRUPOS.items():
        items = []
        for feature in lista:
            matriz = construir_matriz(df, feature)
            if matriz.shape[0] == 0:
                print(f"Sin datos suficientes: {feature}")
                continue
            mediana, p25, p75, area = calcular_banda(matriz)
            items.append({
                "feature": feature, "area": area,
                "mediana": mediana, "p25": p25, "p75": p75,
            })

        items.sort(key=lambda it: it["area"])
        grupos_ordenados[nombre] = items

        for pos, it in enumerate(items, start=1):
            filas_csv.append({
                "grupo": nombre,
                "posicion_en_grupo": pos,
                "feature": it["feature"],
                "area": it["area"],
            })

    tabla = pd.DataFrame(filas_csv)
    destino_csv = CARPETA_SALIDA / "grupos_17_features.csv"
    tabla.to_csv(destino_csv, index=False)
    print(f"Guardado: {destino_csv}")

    graficar(grupos_ordenados, CARPETA_SALIDA / "grupos_17_features.png")

    print()
    print(tabla.to_string(index=False))
    print("\nProceso terminado.")


if __name__ == "__main__":
    main()