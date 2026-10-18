"""
Grafica features de tiempo + espectral restantes (7 por eje).
Una imagen por eje (H y V). Filas = rodamientos, columnas = features.
Eje X = tiempo en horas.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path(r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed")
INPUT_FILE = DATA_DIR / "features_48.parquet"
OUT_H = DATA_DIR / "time_spectral_features_axis_H.png"
OUT_V = DATA_DIR / "time_spectral_features_axis_V.png"

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


def load_data():
    if not INPUT_FILE.exists():
        print(f"❌ No se encuentra: {INPUT_FILE}")
        sys.exit(1)

    df = pd.read_parquet(INPUT_FILE)
    need = ["bearing_id", "time_seconds"] + FEATURES_H + FEATURES_V
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"❌ Faltan columnas: {missing}")
        print(f"   Disponibles: {list(df.columns)}")
        sys.exit(1)

    df = df.sort_values(["bearing_id", "time_seconds"]).reset_index(drop=True)
    print(f"✅ Cargado: {len(df):,} filas | bearings: {df['bearing_id'].nunique()}")
    return df


def plot_axis(df: pd.DataFrame, features: list, axis_name: str, out_path: Path):
    bearings = sorted(df["bearing_id"].unique())
    n_rows = len(bearings)
    n_cols = len(features)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.4 * n_cols, 1.5 * n_rows),
        sharex=False,
        squeeze=False,
    )

    for i, bid in enumerate(bearings):
        sub = df[df["bearing_id"] == bid]
        t_h = sub["time_seconds"].to_numpy(dtype=float) / 3600.0

        for j, feat in enumerate(features):
            ax = axes[i, j]
            y = sub[feat].to_numpy(dtype=float)
            ax.plot(t_h, y, color="C0", lw=0.8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.25, lw=0.4)

            if i == 0:
                ax.set_title(feat.replace(f"_{axis_name}", ""), fontsize=9, pad=3)
            if j == 0:
                ax.set_ylabel(bid, fontsize=7, rotation=90, labelpad=4)
            if i == n_rows - 1:
                ax.set_xlabel("Tiempo (horas)", fontsize=7)

    fig.suptitle(
        f"Time + Spectral Features — Axis {axis_name}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Guardado: {out_path}")


def main():
    df = load_data()
    plot_axis(df, FEATURES_H, "H", OUT_H)
    plot_axis(df, FEATURES_V, "V", OUT_V)
    print("Listo.")


if __name__ == "__main__":
    main()