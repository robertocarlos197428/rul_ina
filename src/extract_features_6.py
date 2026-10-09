"""
Extrae 6 features + columnas de identificación desde features_68.parquet
y guarda features_6.parquet y features_6.csv en la misma carpeta.
"""

from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    print("❌ Falta pandas. Instálalo con: pip install pandas pyarrow")
    sys.exit(1)

# ------------------------------------------------------------------------------
# Rutas y columnas
# ------------------------------------------------------------------------------
DATA_DIR = Path(r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed")
INPUT_FILE = DATA_DIR / "features_68.parquet"
OUTPUT_PARQUET = DATA_DIR / "features_6.parquet"
OUTPUT_CSV = DATA_DIR / "features_6.csv"

ID_COLS = ["bearing_id", "burst_id", "burst_index", "time_seconds"]
FEATURE_COLS = [
    "SpectralStd_H",
    "WPD_Erel_5_H",
    "SpectralCentroid_V",
    "WPD_Erel_8_H",
    "WPD_Erel_6_H",
    "WPD_Erel_7_H",
]
REQUIRED_COLS = ID_COLS + FEATURE_COLS


def main():
    # 1) Verificar carpeta y archivo de entrada
    if not DATA_DIR.exists():
        print(f"❌ No se encuentra la carpeta:\n   {DATA_DIR}")
        sys.exit(1)

    if not INPUT_FILE.exists():
        print(f"❌ No se encuentra el archivo:\n   {INPUT_FILE}")
        print("   Revisa el nombre o la ruta.")
        sys.exit(1)

    print(f"📂 Leyendo: {INPUT_FILE}")

    # 2) Cargar parquet
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"❌ Error al leer el parquet: {e}")
        sys.exit(1)

    print(f"   Filas: {len(df):,}  |  Columnas: {len(df.columns)}")

    # 3) Verificar columnas requeridas
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print("\n❌ Faltan las siguientes columnas en features_68.parquet:")
        for c in missing:
            print(f"   - {c}")
        print("\nColumnas disponibles en el archivo:")
        for c in sorted(df.columns):
            print(f"   · {c}")
        sys.exit(1)

    # 4) Seleccionar y ordenar columnas
    df_out = df[REQUIRED_COLS].copy()

    # 5) Guardar
    try:
        df_out.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"✅ Guardado: {OUTPUT_PARQUET}")
    except Exception as e:
        print(f"❌ Error al guardar parquet: {e}")
        sys.exit(1)

    try:
        df_out.to_csv(OUTPUT_CSV, index=False)
        print(f"✅ Guardado: {OUTPUT_CSV}")
    except Exception as e:
        print(f"❌ Error al guardar CSV: {e}")
        sys.exit(1)

    print(f"\nResumen salida: {len(df_out):,} filas × {len(df_out.columns)} columnas")
    print("Columnas:", ", ".join(df_out.columns.tolist()))


if __name__ == "__main__":
    main()