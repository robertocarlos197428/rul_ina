"""
Detección de FPT (RMS_H) y etiqueta RUL piecewise linear por rodamiento.

Lee   : <raiz>/data/processed/features_13.parquet
Escribe: <raiz>/analisis_features/
           - fpt_por_rodamiento.csv
           - etiquetas_rul_piecewise.parquet
           - fpt_revision.png

Las rutas se resuelven solas: el script sube por las carpetas desde donde está
guardado hasta encontrar data/processed/features_13.parquet. Así funciona en
cualquier computador mientras mantengas la estructura rul_ina/data/processed.
También puedes forzar la raíz con la variable de entorno RUL_ROOT.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- Parámetros -----------------------------
ARCHIVO = "features_13.parquet"
FRAC_RODAJE = 0.15   # decidido: se descarta el 15% inicial
FRAC_SANA = 0.10     # ventana sana tras el rodaje (valor inicial, validar en el gráfico)
K_MAD = 3.0          # umbral = mediana + K_MAD * (1.4826 * MAD), robusto a transitorios
SEG_A_HORAS = 3600.0 # el tiempo del parquet está en segundos; se convierte a horas
N_CONSEC = 10        # muestras consecutivas sobre el umbral para confirmar FPT
SUAVIZAR = True      # mediana 2% + media 1% (centrada; solo para construir etiquetas offline)

# Si la detección automática de columnas falla, escribe aquí los nombres exactos:
COL_RODAMIENTO = None   # ej. "bearing"
COL_TIEMPO = None       # ej. "tiempo_h"
COL_RMS = None          # ej. "RMS_H"


# ----------------------------- Rutas -----------------------------
def encontrar_raiz():
    env = os.environ.get("RUL_ROOT")
    if env:
        return Path(env)
    inicio = Path(__file__).resolve().parent
    for carpeta in [inicio, *inicio.parents]:
        if (carpeta / "data" / "processed" / ARCHIVO).exists():
            return carpeta
    sys.exit(
        f"No encontré data/processed/{ARCHIVO} subiendo desde {inicio}.\n"
        "Guarda este script dentro de la carpeta rul_ina (o una subcarpeta) "
        "o define la variable de entorno RUL_ROOT con la ruta de rul_ina."
    )


# ----------------------------- Columnas -----------------------------
def buscar_columna(cols, candidatos, exacto=False):
    bajo = {c.lower(): c for c in cols}
    for cand in candidatos:
        if cand.lower() in bajo:
            return bajo[cand.lower()]
    if not exacto:
        for cand in candidatos:
            for low, orig in bajo.items():
                if cand.lower() in low:
                    return orig
    return None


def resolver_columnas(df):
    cols = list(df.columns)
    rod = COL_RODAMIENTO or buscar_columna(
        cols, ["bearing", "rodamiento", "bearing_id", "id", "name", "nombre"])
    tie = COL_TIEMPO or buscar_columna(
        cols, ["tiempo_h", "tiempo", "time_h", "time", "horas", "hora", "t_h", "t"])
    rms = COL_RMS or buscar_columna(cols, ["RMS_H"], exacto=True) \
        or buscar_columna(cols, ["rms_h", "rms"])
    if rod is None or rms is None:
        sys.exit(
            "No pude identificar las columnas.\n"
            f"  rodamiento={rod}  tiempo={tie}  rms={rms}\n"
            f"Columnas disponibles: {cols}\n"
            "Edita COL_RODAMIENTO / COL_TIEMPO / COL_RMS al inicio del script."
        )
    return rod, tie, rms


# ----------------------------- Lógica FPT -----------------------------
def suavizar(x, frac_med=0.02, frac_media=0.01):
    s = pd.Series(x)
    n = len(s)
    w1 = max(3, int(n * frac_med) | 1)       # impar
    w2 = max(3, int(n * frac_media) | 1)
    s = s.rolling(w1, center=True, min_periods=1).median()
    s = s.rolling(w2, center=True, min_periods=1).mean()
    return s.to_numpy()


def detectar_fpt(rms, frac_rodaje=FRAC_RODAJE, frac_sana=FRAC_SANA,
                 k=K_MAD, n_consec=N_CONSEC):
    rms = np.asarray(rms, dtype=float)
    n = len(rms)
    i0 = int(n * frac_rodaje)
    i1 = i0 + max(2, int(n * frac_sana))
    ventana = rms[i0:i1]
    mediana = np.median(ventana)
    mad = 1.4826 * np.median(np.abs(ventana - mediana))
    if mad == 0:
        mad = ventana.std()
    umbral = mediana + k * mad

    sobre = rms > umbral
    sobre[:i1] = False
    corrida = np.convolve(sobre.astype(int), np.ones(n_consec, dtype=int), mode="valid")
    cand = np.where(corrida == n_consec)[0]
    idx = int(cand[0]) if len(cand) else None
    return idx, umbral, (i0, i1)


def rul_piecewise(n, idx_fpt):
    if idx_fpt is None:
        return np.linspace(1.0, 0.0, n)
    rul = np.ones(n)
    rul[idx_fpt:] = np.linspace(1.0, 0.0, n - idx_fpt)
    return rul


# ----------------------------- Principal -----------------------------
def main():
    raiz = encontrar_raiz()
    ruta_in = raiz / "data" / "processed" / ARCHIVO
    dir_out = raiz / "analisis_features"
    dir_out.mkdir(parents=True, exist_ok=True)

    print(f"Raíz      : {raiz}")
    print(f"Entrada   : {ruta_in}")
    print(f"Salida    : {dir_out}")

    df = pd.read_parquet(ruta_in)
    col_rod, col_tie, col_rms = resolver_columnas(df)
    print(f"Columnas  : rodamiento='{col_rod}'  tiempo='{col_tie}'  rms='{col_rms}'")

    filas, etiquetas, series = [], [], {}
    for nombre, g in df.groupby(col_rod, sort=False):
        if col_tie is not None:
            g = g.sort_values(col_tie)
        rms = g[col_rms].to_numpy(dtype=float)
        t = (g[col_tie].to_numpy(dtype=float) / SEG_A_HORAS) if col_tie is not None \
            else np.arange(len(g), dtype=float)
        rms_s = suavizar(rms) if SUAVIZAR else rms

        idx, umbral, (i0, i1) = detectar_fpt(rms_s)
        rul = rul_piecewise(len(rms_s), idx)

        filas.append(dict(
            rodamiento=nombre, n_muestras=len(rms_s), idx_fpt=idx,
            frac_fpt=None if idx is None else round(idx / len(rms_s), 4),
            t_fpt_h=None if idx is None else round(float(t[idx]), 3),
            umbral=float(umbral), detectado=idx is not None,
        ))
        etiquetas.append(pd.DataFrame(dict(
            rodamiento=nombre, idx=np.arange(len(rms_s)), t=t, rul_frac=rul)))
        series[nombre] = (t, rms_s, idx, umbral, i0, i1)

    tabla = pd.DataFrame(filas)
    tabla.to_csv(dir_out / "fpt_por_rodamiento.csv", index=False)
    pd.concat(etiquetas, ignore_index=True).to_parquet(
        dir_out / "etiquetas_rul_piecewise.parquet", index=False)
    print("\n", tabla.to_string(index=False))
    if not tabla["detectado"].all():
        print("\nOJO: rodamientos sin FPT (se les dejó etiqueta lineal pura):",
              list(tabla.loc[~tabla["detectado"], "rodamiento"]))

    # Gráfico de revisión
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = int(np.ceil(len(series) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    for ax, (nombre, (t, r, idx, umbral, i0, i1)) in zip(axs.ravel(), series.items()):
        ax.plot(t, r, lw=1)
        ax.axvspan(t[i0], t[min(i1, len(t) - 1)], color="green", alpha=0.15, label="ventana sana")
        ax.axhline(umbral, color="orange", ls="--", lw=1, label="umbral")
        if idx is not None:
            ax.axvline(t[idx], color="red", lw=1.5, label="FPT")
            ax.set_title(f"{nombre}  FPT={idx / len(r):.0%} de la vida", fontsize=9)
        else:
            ax.set_title(f"{nombre}  (SIN FPT)", color="red", fontsize=9)
        ax.legend(fontsize=6)
    for ax in axs.ravel()[len(series):]:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(dir_out / "fpt_revision.png", dpi=120)
    print(f"\nListo. Archivos en: {dir_out}")


if __name__ == "__main__":
    main()