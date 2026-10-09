import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

# Rodamientos a revisar (editar esta lista si quieres otros)
RODAMIENTOS = ["Bearing3_2", "Bearing2_5", "Bearing1_1", "Bearing2_7"]
N_SANO = 70

BASE_DIR = r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed"
OUTPUT_DIR = r"C:\Users\emaker\Downloads\RUL\rul_ina\hi_rul"

FEATURES = [
    "SpectralStd_H",
    "WPD_Erel_5_H",
    "SpectralCentroid_V",
    "WPD_Erel_8_H",
    "WPD_Erel_6_H",
    "WPD_Erel_7_H",
]


def calcular_md(X, n_sano=N_SANO):
    """Misma lógica de hi_rul.py: referencia sana = primeras n_sano ráfagas,
    Ledoit-Wolf sobre covarianza cruda, sin estandarizar."""
    X_sano = X[:n_sano]
    mu_sano = np.mean(X_sano, axis=0)
    lw = LedoitWolf().fit(X_sano)
    prec = lw.precision_
    diff = X - mu_sano
    md = np.sqrt(np.einsum("ij,jk,ik->i", diff, prec, diff))
    return np.maximum(md, 0.0)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(BASE_DIR, "features_68.parquet")
    print(f"Cargando: {path}")
    df = pd.read_parquet(path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, b_id in zip(axes, RODAMIENTOS):
        df_b = df[df["bearing_id"] == b_id].sort_values("burst_index")
        if df_b.empty:
            ax.set_title(f"{b_id}: no encontrado")
            continue

        X = df_b[FEATURES].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        n = len(X)
        md = calcular_md(X)
        tau = md[:N_SANO].max() if md[:N_SANO].max() > 0 else 1.0

        vida_pct = np.arange(n) / (n - 1) * 100
        pct_70 = (N_SANO - 1) / (n - 1) * 100

        ax.plot(vida_pct, md, color="#1f77b4", linewidth=1)
        ax.axvline(pct_70, color="red", linestyle="--", linewidth=1.2,
                   label=f"Ráfaga {N_SANO} ({pct_70:.1f} % de la vida)")
        ax.axhline(tau, color="green", linestyle="--", linewidth=1.2,
                   label=f"τ = {tau:.2f}")
        ax.set_title(f"{b_id} ({n} ráfagas)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Progreso de Vida (%)")
        ax.set_ylabel("MD (Mahalanobis)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

        print(f"{b_id}: n={n}, MD media sano={md[:N_SANO].mean():.2f}, "
              f"MD max sano (tau)={tau:.2f}, MD final={md[-1]:.2f}")

    plt.suptitle("Distancia de Mahalanobis cruda (antes de exp(-MD/τ))",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(OUTPUT_DIR, "diagnostico_md_4_rodamientos.png")
    plt.savefig(out, dpi=200)
    print(f"Guardado: {out}")
    plt.show()


if __name__ == "__main__":
    main()