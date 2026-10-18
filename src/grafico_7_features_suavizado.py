"""
Grafica 7 features tiempo+espectral por eje, con suavizado moderado
(no deforma la tendencia de degradación).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# RUTAS
# ============================================================
INPUT_FILE = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed\features_48.parquet"
)
OUTPUT_DIR = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina\analisis_features\suavizado_moderado"
)

# ============================================================
# CONFIGURACION
# ============================================================
FEATURES = [
    "RMS", "Kurtosis", "Skewness", "CrestFactor",
    "SpectralCentroid", "SpectralStd", "SpectralEntropy",
]

# --- Suavizado moderado (antes era agresivo) ---
# Hampel: solo picos aislados, ventana fija pequeña
HAMPEL_WINDOW = 11
HAMPEL_K = 4.0          # más tolerante; solo outliers claros

# Mediana y media: fracciones bajas de la vida
MEDIAN_FRAC = 0.02      # era 0.10 → aplastaba la curva
MEAN_FRAC = 0.01        # era 0.05
MIN_WINDOW = 5          # era 15

SHOW_RAW = True         # ver cruda tenue detrás ayuda a validar
CHANNELS = ["H", "V"]


def odd_window(n_points, frac, min_window=MIN_WINDOW):
    window = max(min_window, int(round(n_points * frac)))
    return window if window % 2 == 1 else window + 1


def hampel(series, window=HAMPEL_WINDOW, k=HAMPEL_K):
    med = series.rolling(window, center=True, min_periods=1).median()
    abs_dev = (series - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    # MAD=0 → no marcar todo como outlier
    thresh = k * 1.4826 * mad.replace(0, np.nan)
    outliers = abs_dev > thresh
    cleaned = series.copy()
    cleaned[outliers.fillna(False)] = med[outliers.fillna(False)]
    return cleaned


def smooth(series):
    n = len(series)
    w_median = odd_window(n, MEDIAN_FRAC)
    w_mean = odd_window(n, MEAN_FRAC)

    out = hampel(series)
    out = out.rolling(w_median, center=True, min_periods=1).median()
    out = out.rolling(w_mean, center=True, min_periods=1).mean()
    return out


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE).sort_values(["bearing_id", "time_seconds"])
    bearings = sorted(df["bearing_id"].unique())
    n_rows, n_cols = len(bearings), len(FEATURES)
    print(f"Rodamientos: {n_rows} | Features por canal: {n_cols}")
    print(f"Suavizado: Hampel w={HAMPEL_WINDOW} k={HAMPEL_K} | "
          f"mediana {MEDIAN_FRAC:.0%} | media {MEAN_FRAC:.0%}")

    for canal in CHANNELS:
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(20, 2.0 * n_rows), sharex=False
        )

        for r, bearing in enumerate(bearings):
            part = df[df["bearing_id"] == bearing]
            t = part["time_seconds"] / 3600.0

            for c, feat in enumerate(FEATURES):
                ax = axes[r, c]
                col = f"{feat}_{canal}"
                raw = part[col].astype(float).reset_index(drop=True)
                t_plot = t.reset_index(drop=True)
                smoothed = smooth(raw)

                if SHOW_RAW:
                    ax.plot(t_plot, raw, color="lightgray", linewidth=0.6, zorder=1)
                ax.plot(t_plot, smoothed, color="tab:blue", linewidth=1.2, zorder=2)

                # Límites Y según cruda+suave para no cortar picos reales
                y_ref = pd.concat([raw, smoothed], axis=0)
                lo, hi = np.nanmin(y_ref), np.nanmax(y_ref)
                margin = 0.08 * (hi - lo) if hi > lo else 1.0
                ax.set_ylim(lo - margin, hi + margin)

                ax.tick_params(labelsize=6)
                ax.grid(True, alpha=0.3)

                if r == 0:
                    ax.set_title(feat, fontsize=9)
                if c == 0:
                    ax.set_ylabel(bearing, fontsize=7)
                if r == n_rows - 1:
                    ax.set_xlabel("Tiempo (horas)", fontsize=7)

        fig.suptitle(
            f"Time + Spectral Features - Axis {canal} "
            f"(suavizado moderado: Hampel + mediana {MEDIAN_FRAC:.0%} + media {MEAN_FRAC:.0%})",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        out = OUTPUT_DIR / f"features_axis_{canal}_suavizado_moderado.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Guardado: {out}")

    print("Listo.")


if __name__ == "__main__":
    main()