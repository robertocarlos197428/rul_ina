"""
Crea el dataset de entrenamiento con 6 rodamientos (Bearing1_1 a Bearing1_6):
features_13.parquet filtrado + columna rul_frac piecewise.

Ubicación: rul_ina/test/grafico_rms_fpt_V1.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- Parámetros y Rutas -----------------------------
# Como este script está en <raiz>/test/, subimos un nivel para encontrar la raíz (<raiz>)
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent

ARCHIVO_FEATURES = "features_13.parquet"
RUTA_FEAT = RAIZ_PROYECTO / "data" / "processed" / ARCHIVO_FEATURES
RUTA_ETQ = RAIZ_PROYECTO / "analisis_features" / "etiquetas_rul_piecewise.parquet"
RUTA_OUT = RAIZ_PROYECTO / "data" / "processed" / "etiquetas_rul_piecewise.parquet"

RODAMIENTOS = [f"Bearing1_{i}" for i in range(1, 7)]


def resolver_columnas(df):
    """Detecta automáticamente los nombres estándar de las columnas en el dataframe."""
    col_rod = next((c for c in ["bearing_id", "rodamiento", "id_rodamiento"] if c in df.columns), None)
    col_tie = next((c for c in ["time_seconds", "tiempo", "time"] if c in df.columns), None)
    if not col_rod:
        sys.exit(f"❌ No se encontró columna de identificación de rodamiento en el DataFrame. Columnas: {df.columns.tolist()}")
    return col_rod, col_tie


def main():
    if not RUTA_ETQ.exists():
        sys.exit(f"❌ No existe {RUTA_ETQ}. Genera antes las etiquetas piecewise correspondientes.")
    if not RUTA_FEAT.exists():
        sys.exit(f"❌ No existe el archivo de características en {RUTA_FEAT}.")

    df = pd.read_parquet(RUTA_FEAT)
    etq = pd.read_parquet(RUTA_ETQ)
    col_rod, col_tie = resolver_columnas(df)

    print(f"Entrada features : {RUTA_FEAT}")
    print(f"Entrada etiquetas: {RUTA_ETQ}")
    print(f"Salida           : {RUTA_OUT}")

    # Verificar que los 6 rodamientos estén presentes
    disponibles = set(df[col_rod])
    faltan = [r for r in RODAMIENTOS if r not in disponibles]
    if faltan:
        sys.exit(f"❌ Rodamientos no encontrados en features: {faltan}\nDisponibles: {sorted(disponibles)}")

    partes = []
    for nombre in RODAMIENTOS:
        g = df[df[col_rod] == nombre]
        if col_tie is not None:
            g = g.sort_values(col_tie)
        g = g.reset_index(drop=True)

        # Filtrar etiquetas para el rodamiento actual
        etq_rod = etq[etq["rodamiento"] == nombre] if "rodamiento" in etq.columns else etq.loc[etq.iloc[:, 0] == nombre]
        
        if "idx" in etq_rod.columns:
            rul = etq_rod.sort_values("idx")["rul_frac"].to_numpy()
        else:
            rul = etq_rod["rul_frac"].to_numpy()

        if len(rul) != len(g):
            sys.exit(f"❌ {nombre}: features tiene {len(g)} filas y las etiquetas {len(rul)}. "
                     "Verifica que correspondan al mismo dataset.")
        
        g["rul_frac"] = rul
        partes.append(g)

    out = pd.concat(partes, ignore_index=True)
    RUTA_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RUTA_OUT, index=False)

    resumen = out.groupby(col_rod, sort=False).agg(
        n_muestras=("rul_frac", "size"),
        rul_min=("rul_frac", "min"),
        rul_max=("rul_frac", "max"),
        frac_en_1=("rul_frac", lambda s: float(np.mean(s == 1.0))),
    )
    print("\n", resumen.round(3).to_string())
    print(f"\nListo: {len(out)} filas, {out.shape[1]} columnas -> {RUTA_OUT}")


if __name__ == "__main__":
    main()