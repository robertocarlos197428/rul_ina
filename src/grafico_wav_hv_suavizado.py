"""
Grafica WPD (4 imágenes × 8 features) + cruzadas H-V (1 imagen).
Mismo suavizado moderado que el script de tiempo/espectral.
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
# GRUPOS DE FEATURES (5 imágenes)
# ============================================================
FEATURE_GROUPS = {
    "wpd_H_0_7": [f"WPD_Erel_{i}_H" for i in range(0, 8)],
    "wpd_H_8_15": [f"WPD_Erel_{i}_H" for i in range(8, 16)],
    "wpd_V_0_7": [f"WPD_Erel_{i}_V" for i in range(0, 8)],
    "wpd_V_8_15": [f"WPD_Erel_{i}_V" for i in range(8, 16)],
    "cruzadas_HV": ["Pearson_HV", "EnergyRatio_HV"],
}

GROUP_TITLES = {
    "wpd_H_0_7": "WPD Erel 0-7 — Axis H",
    "wpd_H_8_15": "WPD Erel 8-15 — Axis H",
    "wpd_V_0_7": "WPD Erel 0-7 — Axis V",
    "wpd_V_8_15": "WPD Erel 8-15 — Axis V",
    "cruzadas_HV": "Cruzadas H-V (Pearson, EnergyRatio)",
}

# ============================================================
# SUAVIZADO MODERADO
# ============================================================
HAMPEL_WINDOW = 11
HAMPEL_K = 4.0
MEDIAN_FRAC = 0.02
MEAN_FRAC = 0.01
MIN_WINDOW = 5
SHOW_RAW = True


def odd_window(n_points, frac, min_window=MIN_WINDOW):
    window = max(min_window, int(round(n_points * frac)))
    return window if window % 2 == 1 else window + 1


def hampel(series, window=HAMPEL_WINDOW, k=HAMPEL_K):
    med = series.rolling(window, center=True, min_periods=1).median()
    abs_dev = (series - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
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


def plot_group(df, bearings, features, title, out_path):
    n_rows, n_cols = len(bearings), len(features)
    # Ancho proporcional al número de columnas (cruzadas solo tiene 2)
    fig_w = max(6.0, 2.5 * n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, 2.0 * n_rows),
        sharex=False,
        squeeze=False,
    )

    for r, bearing in enumerate(bearings):
        part = df[df["bearing_id"] == bearing]
        t = part["time_seconds"].to_numpy(dtype=float) / 3600.0

        for c, feat in enumerate(features):
            ax = axes[r, c]
            raw = part[feat].astype(float).reset_index(drop=True)
            t_plot = pd.Series(t).reset_index(drop=True)
            smoothed = smooth(raw)

            if SHOW_RAW:
                ax.plot(t_plot, raw, color="lightgray", linewidth=0.6, zorder=1)
            ax.plot(t_plot, smoothed, color="tab:blue", linewidth=1.2, zorder=2)

            y_ref = pd.concat([raw, smoothed], axis=0)
            lo, hi = np.nanmin(y_ref), np.nanmax(y_ref)
            margin = 0.08 * (hi - lo) if hi > lo else 1.0
            ax.set_ylim(lo - margin, hi + margin)

            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)

            if r == 0:
                # Título corto de columna
                short = (
                    feat.replace("WPD_Erel_", "Erel")
                    .replace("_H", "")
                    .replace("_V", "")
                    .replace("Pearson_HV", "Pearson")
                    .replace("EnergyRatio_HV", "EnergyRatio")
                )
                ax.set_title(short, fontsize=9)
            if c == 0:
                ax.set_ylabel(bearing, fontsize=7)
            if r == n_rows - 1:
                ax.set_xlabel("Tiempo (horas)", fontsize=7)

    fig.suptitle(
        f"{title}  |  suavizado moderado "
        f"(Hampel + mediana {MEDIAN_FRAC:.0%} + media {MEAN_FRAC:.0%})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE).sort_values(["bearing_id", "time_seconds"])
    bearings = sorted(df["bearing_id"].unique())
    print(f"Rodamientos: {len(bearings)}")

    # Verificar columnas
    all_feats = [f for feats in FEATURE_GROUPS.values() for f in feats]
    missing = [c for c in all_feats if c not in df.columns]
    if missing:
        print(f"❌ Faltan columnas: {missing}")
        return

    for key, features in FEATURE_GROUPS.items():
        out = OUTPUT_DIR / f"features_{key}_suavizado_moderado.png"
        plot_group(df, bearings, features, GROUP_TITLES[key], out)

    print("Listo. 5 imágenes generadas.")


if __name__ == "__main__":
    main()