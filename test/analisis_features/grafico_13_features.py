from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

# ============================================================
# RUTAS (Actualizadas a tu nueva PC)
# ============================================================
INPUT_FILE = Path(
    r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed\features_48.parquet"
)
OUTPUT_DIR = Path(
    r"C:\Users\emaker\Downloads\RUL\rul_ina\analisis_features\suavizado_moderado"
)

# ============================================================
# GRUPOS DE LAS 13 FEATURES (2 IMÁGENES)
# ============================================================
FEATURE_GROUPS = {
    "subset_1_7_features": [
        "RMS_H",
        "RMS_V",
        "Kurtosis_H",
        "Kurtosis_V",
        "CrestFactor_H",
        "CrestFactor_V",
        "SpectralStd_H",
    ],
    "subset_2_6_features": [
        "WPD_Erel_5_H",
        "WPD_Erel_3_H",
        "WPD_Erel_6_H",
        "Skewness_H",
        "WPD_Erel_5_V",
        "WPD_Erel_4_H",
    ],
}

GROUP_TITLES = {
    "subset_1_7_features": "Features Temporales y Espectrales (1-7)",
    "subset_2_6_features": "Features WPD y Skewness (8-13)",
}

# ============================================================
# SUAVIZADO MODERADO (Fórmulas originales)
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
                short = (
                    feat.replace("WPD_Erel_", "Erel_")
                    .replace("_H", "_H")
                    .replace("_V", "_V")
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

    all_feats = [f for feats in FEATURE_GROUPS.values() for f in feats]
    missing = [c for c in all_feats if c not in df.columns]
    if missing:
        print(f"❌ Faltan columnas: {missing}")
        return

    for key, features in FEATURE_GROUPS.items():
        out = OUTPUT_DIR / f"features_{key}_suavizado_moderado.png"
        plot_group(df, bearings, features, GROUP_TITLES[key], out)

    print("Listo. 2 imágenes generadas con las 13 features.")


if __name__ == "__main__":
    main()