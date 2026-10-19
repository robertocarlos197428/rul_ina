"""
Lee features_48.parquet, se queda con las 13 features finales y guarda
features_13.parquet + .csv.
Sin modificar filas ni los valores. Se conservan las columnas de
identificacion (bearing_id, burst_id, burst_index, time_seconds).
"""

from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    print("❌ Falta pandas. Instala con: pip install pandas pyarrow")
    sys.exit(1)

DATA_DIR = Path(r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed")
INPUT_FILE = DATA_DIR / "features_48.parquet"
OUTPUT_PARQUET = DATA_DIR / "features_13.parquet"
OUTPUT_CSV = DATA_DIR / "features_13.csv"

# Columnas de identificacion (se conservan tal cual)
ID_COLS = [
    "bearing_id",
    "burst_id",
    "burst_index",
    "time_seconds",
]

# Las 13 features finales
KEEP_COLS = [
    # Suben (7)
    "RMS_H",
    "RMS_V",
    "Kurtosis_H",
    "Kurtosis_V",
    "CrestFactor_H",
    "CrestFactor_V",
    "SpectralStd_H",
    # Bajan (6)
    "WPD_Erel_5_H",
    "WPD_Erel_3_H",
    "WPD_Erel_6_H",
    "Skewness_H",
    "WPD_Erel_5_V",
    "WPD_Erel_4_H",
]


def main():
    if not DATA_DIR.exists():
        print(f"❌ No se encuentra la carpeta:\n   {DATA_DIR}")
        sys.exit(1)

    if not INPUT_FILE.exists():
        print(f"❌ No se encuentra el archivo:\n   {INPUT_FILE}")
        sys.exit(1)

    print(f"📂 Leyendo: {INPUT_FILE}")
    try:
        df = pd.read_parquet(INPUT_FILE)
    except Exception as e:
        print(f"❌ Error al leer parquet: {e}")
        sys.exit(1)

    print(f"   Filas: {len(df):,}  |  Columnas: {len(df.columns)}")

    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print("❌ Estas features a conservar no están en el archivo:")
        for c in missing:
            print(f"   - {c}")
        print("\nColumnas disponibles:")
        for c in df.columns:
            print(f"   · {c}")
        sys.exit(1)

    id_present = [c for c in ID_COLS if c in df.columns]
    id_missing = [c for c in ID_COLS if c not in df.columns]
    if id_missing:
        print(f"⚠️  Columnas de identificación que no existen (se omiten): {id_missing}")

    df_out = df[id_present + KEEP_COLS].copy()

    # Verificaciones: no se tocaron filas ni valores
    assert len(df_out) == len(df), "Cambió el número de filas"
    assert df_out[KEEP_COLS].equals(df[KEEP_COLS]), "Cambiaron los valores"

    print(f"   Columnas finales: {len(df_out.columns)} "
          f"({len(id_present)} de identificación + {len(KEEP_COLS)} features)")
    print(f"   Columnas eliminadas: {len(df.columns) - len(df_out.columns)}")

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

    print("Listo. No se modificaron filas ni valores.")


if __name__ == "__main__":
    main()