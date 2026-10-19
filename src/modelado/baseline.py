"""
Baseline lineal con validacion LOBO (dejar un rodamiento fuera).

- Entrada : features_13_rul.parquet (13 features + rul_frac)
- Modelo  : regresion lineal (minimos cuadrados) sobre features normalizadas
- Etiqueta: rul_frac (fraccion de vida, 1 -> 0)
- LOBO    : 17 veces, entrena con 16 rodamientos y prueba con el que queda fuera.
            La normalizacion (media y desviacion) se calcula SOLO con los 16
            de entrenamiento.

MODO_ENTRADA:
  "promedio"  -> media causal de las ultimas VENTANA rafagas (solo mira atras)
  "actual"    -> valor de la rafaga actual, sin suavizar

Salida (en resultados/<modo>/):
  - errores_por_rodamiento.csv
  - predicciones.csv
  - pesos_regresion.csv
  - baseline_lobo.png
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


# ============================================================
# CONFIGURACION
# ============================================================

BASE_DIR = Path(r"C:\Users\a\Downloads\CursoDL\rul_ina")
INPUT_FILE = BASE_DIR / "data" / "processed" / "features_13_rul.parquet"
RESULTS_DIR = BASE_DIR / "modelado" / "01_baseline_lineal" / "resultados"

MODO_ENTRADA = "promedio"   # "promedio" o "actual"
VENTANA = 50                # rafagas (solo se usa si MODO_ENTRADA = "promedio")
CLIPEAR_PREDICCION = False  # True = limita la prediccion a [0, 1]

BEARING_COL = "bearing_id"
TIME_COL = "time_seconds"
TARGET_COL = "rul_frac"

FEATURES = [
    "RMS_H", "RMS_V",
    "Kurtosis_H", "Kurtosis_V",
    "CrestFactor_H", "CrestFactor_V",
    "SpectralStd_H",
    "WPD_Erel_5_H", "WPD_Erel_3_H", "WPD_Erel_6_H",
    "Skewness_H",
    "WPD_Erel_5_V", "WPD_Erel_4_H",
]

DPI = 150


# ============================================================
# FUNCIONES
# ============================================================

def preparar_entrada(df):
    """Devuelve el DataFrame ordenado, con las features segun MODO_ENTRADA.
    La media es causal: cada rodamiento por separado y solo hacia atras."""
    df = df.sort_values([BEARING_COL, TIME_COL]).reset_index(drop=True)

    if MODO_ENTRADA == "actual":
        return df

    if MODO_ENTRADA == "promedio":
        df = df.copy()
        df[FEATURES] = (
            df.groupby(BEARING_COL)[FEATURES]
            .transform(lambda s: s.rolling(VENTANA, min_periods=1).mean())
        )
        return df

    raise ValueError(f"MODO_ENTRADA invalido: {MODO_ENTRADA}")


def nombre_modo():
    return f"promedio_{VENTANA}" if MODO_ENTRADA == "promedio" else "actual"


def ajustar(X, y):
    """Normaliza con X (entrenamiento) y ajusta minimos cuadrados.
    Devuelve media, desviacion, pesos (con intercepto al final)."""
    media = X.mean(axis=0)
    desv = X.std(axis=0)
    desv[desv == 0] = 1.0
    Z = (X - media) / desv
    A = np.column_stack([Z, np.ones(len(Z))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return media, desv, coef


def predecir(X, media, desv, coef):
    Z = (X - media) / desv
    A = np.column_stack([Z, np.ones(len(Z))])
    pred = A @ coef
    if CLIPEAR_PREDICCION:
        pred = np.clip(pred, 0.0, 1.0)
    return pred


def graficar(pred_df, errores, destino):
    bearings = list(errores["bearing_id"])
    n = len(bearings)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows),
        sharey=True, squeeze=False,
    )

    for ax, b in zip(axes.flat, bearings):
        part = pred_df[pred_df[BEARING_COL] == b]
        vida = (1.0 - part["rul_real"].to_numpy()) * 100.0
        mae = float(errores.loc[errores["bearing_id"] == b, "MAE"].iloc[0])

        ax.plot(vida, part["rul_real"], color="black", linewidth=1.5, label="Real")
        ax.plot(vida, part["rul_pred"], color="tab:blue", linewidth=0.9,
                alpha=0.9, label="Predicho")
        ax.set_title(f"{b}  |  MAE {mae:.3f}", fontsize=10)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for ax in list(axes.flat)[n:]:
        ax.axis("off")

    for ax in axes[-1, :]:
        ax.set_xlabel("Vida util (%)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("rul_frac", fontsize=9)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        f"Baseline lineal, LOBO - entrada: {nombre_modo()}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(destino, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Guardado: {destino}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not INPUT_FILE.exists():
        print(f"❌ No se encuentra el archivo:\n   {INPUT_FILE}")
        sys.exit(1)

    print(f"📂 Leyendo: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Filas: {len(df):,}  |  Columnas: {len(df.columns)}")

    requeridas = [BEARING_COL, TIME_COL, TARGET_COL] + FEATURES
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        print("❌ Faltan columnas:")
        for c in faltantes:
            print(f"   - {c}")
        sys.exit(1)

    if df[requeridas].isna().any().any():
        print("❌ Hay NaN en las columnas usadas.")
        sys.exit(1)

    out_dir = RESULTS_DIR / nombre_modo()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"⚙️  Entrada: {nombre_modo()}  |  Features: {len(FEATURES)}")
    df = preparar_entrada(df)

    bearings = sorted(df[BEARING_COL].unique())
    print(f"   Rodamientos: {len(bearings)}  ->  {len(bearings)} entrenamientos LOBO")

    X_all = df[FEATURES].to_numpy(dtype=float)
    y_all = df[TARGET_COL].to_numpy(dtype=float)
    ids = df[BEARING_COL].to_numpy()

    # --------------------------------------------------------
    # LOBO
    # --------------------------------------------------------
    pred_filas = []
    err_filas = []

    for b in bearings:
        test = ids == b
        train = ~test

        media, desv, coef = ajustar(X_all[train], y_all[train])
        pred = predecir(X_all[test], media, desv, coef)
        real = y_all[test]

        error = pred - real
        err_filas.append({
            "bearing_id": b,
            "filas": int(test.sum()),
            "MAE": float(np.mean(np.abs(error))),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
        })

        part = df.loc[test, [BEARING_COL, TIME_COL]].copy()
        part["rul_real"] = real
        part["rul_pred"] = pred
        pred_filas.append(part)

    errores = pd.DataFrame(err_filas)
    pred_df = pd.concat(pred_filas, ignore_index=True)

    # Promedios: por rodamiento y global (todas las rafagas juntas)
    err_glob = pred_df["rul_pred"] - pred_df["rul_real"]
    resumen = pd.DataFrame([
        {"bearing_id": "PROMEDIO_RODAMIENTOS", "filas": int(errores["filas"].sum()),
         "MAE": errores["MAE"].mean(), "RMSE": errores["RMSE"].mean()},
        {"bearing_id": "GLOBAL_TODAS_LAS_RAFAGAS", "filas": len(pred_df),
         "MAE": float(np.mean(np.abs(err_glob))),
         "RMSE": float(np.sqrt(np.mean(err_glob ** 2)))},
    ])
    errores_out = pd.concat([errores, resumen], ignore_index=True)

    # --------------------------------------------------------
    # Pesos con los 17 rodamientos (para ver que features pesan mas)
    # --------------------------------------------------------
    _, _, coef_all = ajustar(X_all, y_all)
    pesos = pd.DataFrame({
        "feature": FEATURES + ["intercepto"],
        "peso": coef_all,
    })
    pesos["peso_abs"] = pesos["peso"].abs()
    pesos = pesos.sort_values("peso_abs", ascending=False).drop(columns="peso_abs")

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------
    errores_out.to_csv(out_dir / "errores_por_rodamiento.csv", index=False)
    pred_df.to_csv(out_dir / "predicciones.csv", index=False)
    pesos.to_csv(out_dir / "pesos_regresion.csv", index=False)
    print(f"✅ Guardado en: {out_dir}")

    graficar(pred_df, errores, out_dir / "baseline_lobo.png")

    print()
    print(errores_out.round(4).to_string(index=False))
    print()
    print("Pesos (features normalizadas, ordenados por magnitud):")
    print(pesos.round(4).to_string(index=False))
    print("\nListo.")


if __name__ == "__main__":
    main()