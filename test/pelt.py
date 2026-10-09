import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import ruptures as rpt

# ---------------- Parámetros ----------------
N_SANO = 70            # fin del tramo sano (ráfagas)
PCT_INICIO = 0.05      # la búsqueda del FPT empieza en max(N_SANO, 5 % de la vida)
VENTANA = 11           # media móvil centrada (ráfagas)
MIN_SIZE = 10          # largo mínimo de un tramo (ráfagas)
JUMP = 5               # PELT evalúa cambios cada 5 ráfagas
PENALTY_MULT = 50      # multiplicador de la penalización (v1 usaba 11). Subir = menos cambios
JITTER = 1e-3          # ruido mínimo para evitar varianza cero en tramos planos
CERCA_DE_INICIO = 20   # marca el FPT si cae dentro de estas ráfagas del inicio de búsqueda

OUTPUT_DIR = r"C:\Users\emaker\Downloads\RUL\rul_ina\hi_rul"
PARQUET_IN = os.path.join(OUTPUT_DIR, "dataset_hi_rul_features.parquet")


def detectar_cambios(senal):
    """PELT con costo 'normal' (cambios de nivel y de variabilidad).
    Penalización automática tipo BIC: 2 parámetros (media y varianza) por cambio."""
    n = len(senal)
    pen = PENALTY_MULT * 2 * np.log(n)
    algo = rpt.Pelt(model="normal", min_size=MIN_SIZE, jump=JUMP)
    algo.fit(senal.reshape(-1, 1))
    bkps = algo.predict(pen=pen)
    return [b for b in bkps if b < n]  # el último elemento es n, se descarta


def main():
    print(f"Cargando: {PARQUET_IN}")
    df = pd.read_parquet(PARQUET_IN)

    rng = np.random.default_rng(0)
    resultados = []
    datos = {}

    for b_id in sorted(df["bearing_id"].unique()):
        df_b = df[df["bearing_id"] == b_id].sort_values("burst_index")
        hi = df_b["Health_Indicator"].values.astype(float)
        n = len(hi)

        suave = pd.Series(hi).rolling(VENTANA, center=True, min_periods=1).mean().values
        senal = suave + JITTER * rng.standard_normal(n)

        inicio = max(N_SANO, int(np.ceil(PCT_INICIO * n)))
        cambios = detectar_cambios(senal)
        en_zona = [c for c in cambios if c >= inicio]
        fpt = en_zona[0] if en_zona else None

        datos[b_id] = dict(hi=hi, suave=suave, cambios=cambios, fpt=fpt,
                           n=n, inicio=inicio)
        resultados.append({
            "bearing_id": b_id,
            "n_rafagas": n,
            "inicio_busqueda": inicio,
            "n_cambios_total": len(cambios),
            "n_cambios_en_zona": len(en_zona),
            "FPT_rafaga": fpt,
            "FPT_pct_vida": round(fpt / (n - 1) * 100, 1) if fpt is not None else None,
            "FPT_cerca_de_inicio": (fpt - inicio < CERCA_DE_INICIO) if fpt is not None else None,
        })
        print(f"{b_id}: n={n}, inicio búsqueda={inicio}, cambios totales={len(cambios)}, "
              f"en zona={len(en_zona)}, FPT={fpt}")

    tabla = pd.DataFrame(resultados)
    out_csv = os.path.join(OUTPUT_DIR, "tabla_fpt_pelt_v2.csv")
    tabla.to_csv(out_csv, index=False)
    print(f"Guardado: {out_csv}")

    # ---------------- Grilla ----------------
    fig, axes = plt.subplots(6, 3, figsize=(15, 18))
    axes = axes.flatten()

    for idx, (b_id, d) in enumerate(datos.items()):
        ax = axes[idx]
        n = d["n"]
        pct = np.arange(n) / (n - 1) * 100

        ax.plot(pct, d["hi"], color="lightgray", linewidth=0.7)
        ax.plot(pct, d["suave"], color="#2ca02c", linewidth=1.3)
        ax.axvline((N_SANO - 1) / (n - 1) * 100, color="red",
                   linestyle="--", linewidth=1)
        ax.axvline(d["inicio"] / (n - 1) * 100, color="orange",
                   linestyle="--", linewidth=1)
        for c in d["cambios"]:
            ax.axvline(c / (n - 1) * 100, color="gray", linestyle=":", linewidth=1)
        if d["fpt"] is not None:
            ax.axvline(d["fpt"] / (n - 1) * 100, color="purple", linewidth=2.2)
            titulo = f"{b_id} | FPT: ráfaga {d['fpt']} ({d['fpt'] / (n - 1) * 100:.1f} %)"
        else:
            titulo = f"{b_id} | sin FPT detectado"
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.set_ylabel("HI", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx >= 15:
            ax.set_xlabel("Progreso de Vida (%)", fontsize=8)

    # Panel 18: leyenda
    ax = axes[17]
    ax.axis("off")
    handles = [
        Line2D([0], [0], color="lightgray", lw=1.5, label="HI original"),
        Line2D([0], [0], color="#2ca02c", lw=1.5, label=f"HI suavizado (media móvil {VENTANA})"),
        Line2D([0], [0], color="red", lw=1.5, ls="--", label=f"Ráfaga {N_SANO} (fin tramo sano)"),
        Line2D([0], [0], color="orange", lw=1.5, ls="--",
               label=f"Inicio de búsqueda (máx. ráfaga {N_SANO} y {int(PCT_INICIO * 100)} % de vida)"),
        Line2D([0], [0], color="gray", lw=1.5, ls=":", label="Cambios detectados por PELT"),
        Line2D([0], [0], color="purple", lw=2.5, label="FPT (primer cambio desde el inicio de búsqueda)"),
    ]
    ax.legend(handles=handles, loc="center", fontsize=9, frameon=False)

    plt.suptitle(f"PELT sobre el HI - v2 (penalización x{PENALTY_MULT})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])

    out_png = os.path.join(OUTPUT_DIR, "grilla_pelt_fpt_v2.png")
    plt.savefig(out_png, dpi=200)
    print(f"Guardado: {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
