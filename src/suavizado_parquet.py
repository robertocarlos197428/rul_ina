"""
Suaviza features_48.parquet (por rodamiento) y guarda
features_48_suavizado.parquet + .csv en data/processed.
Mantiene IDs, tiempo y el mismo número de filas.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ============================================================
# RUTAS
# ============================================================
DATA_DIR = Path(r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed")
INPUT_FILE = DATA_DIR / "features_48.parquet"
OUTPUT_PARQUET = DATA_DIR / "features_48_suavizado.parquet"
OUTPUT_CSV = DATA_DIR / "features_48_suavizado.csv"

ID_COLS = ["bearing_id", "burst_id", "burst_index", "time_seconds"]

# ============================================================
# SUAVIZADO MODERADO
# ============================================================
HAMPEL_WINDOW = 11
HAMPEL_K = 4.0
MEDIAN_FRAC = 0.02
MEAN_FRAC = 0.01
MIN_WINDOW = 5


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


def smooth(series: pd.Series) -> pd.Series:
    n = len(series)
    w_median = odd_window(n, MEDIAN_FRAC)
    w_mean = odd_window(n, MEAN_FRAC)
    out = hampel(series.astype(float))
    out = out.rolling(w_median, center=True, min_periods=1).median()
    out = out.rolling(w_mean, center=True, min_periods=1).mean()
    return out


def main():
    if not INPUT_FILE.exists():
        print(f"❌ No se encuentra: {INPUT_FILE}")
        sys.exit(1)

    df = pd.read_parquet(INPUT_FILE)
    missing_id = [c for c in ID_COLS if c not in df.columns]
    if missing_id:
        print(f"❌ Faltan columnas de ID: {missing_id}")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in ID_COLS]
    if not feature_cols:
        print("❌ No hay columnas de features para suavizar.")
        sys.exit(1)

    print(f"Entrada: {INPUT_FILE}")
    print(f"Filas: {len(df):,} | Features: {len(feature_cols)} | Bearings: {df['bearing_id'].nunique()}")
    print(
        f"Suavizado: Hampel w={HAMPEL_WINDOW} k={HAMPEL_K} | "
        f"mediana {MEDIAN_FRAC:.0%} | media {MEAN_FRAC:.0%}"
    )

    df = df.sort_values(["bearing_id", "time_seconds"]).reset_index(drop=True)
    parts = []

    for bid, g in df.groupby("bearing_id", sort=True):
        g = g.copy()
        for col in feature_cols:
            g[col] = smooth(g[col].reset_index(drop=True)).to_numpy()
        parts.append(g)
        print(f"  · {bid}: {len(g)} ráfagas")

    out = pd.concat(parts, ignore_index=True)

    # Orden de columnas: IDs primero, luego features
    out = out[ID_COLS + feature_cols]

    out.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"✅ Parquet: {OUTPUT_PARQUET}")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ CSV:     {OUTPUT_CSV}")

    print(f"Listo. Filas entrada={len(df):,} | salida={len(out):,} (sin reducción).")


if __name__ == "__main__":
    main()