"""
Lee features_68.parquet, elimina 20 features y guarda features_48.parquet + .csv
Sin modificar filas ni el resto de columnas.
"""

from pathlib import Path
import sys

try:
    import pandas as pd
except ImportError:
    print("❌ Falta pandas. Instala con: pip install pandas pyarrow")
    sys.exit(1)

DATA_DIR = Path(r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed")
INPUT_FILE = DATA_DIR / "features_68.parquet"
OUTPUT_PARQUET = DATA_DIR / "features_48.parquet"
OUTPUT_CSV = DATA_DIR / "features_48.csv"

DROP_COLS = [
    "Std_H",
    "Std_V",
    "Variance_H",
    "Variance_V",
    "MAV_H",
    "MAV_V",
    "ImpulseFactor_H",
    "ImpulseFactor_V",
    "MarginFactor_H",
    "MarginFactor_V",
    "SpectralFlatness_H",
    "SpectralFlatness_V",
    "SpectralRMS_H",
    "SpectralRMS_V",
    "Peak_H",
    "Peak_V",
    "P2P_H",
    "P2P_V",
    "ShapeFactor_H",
    "ShapeFactor_V",
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

    missing = [c for c in DROP_COLS if c not in df.columns]
    if missing:
        print("❌ Estas columnas a eliminar no están en el archivo:")
        for c in missing:
            print(f"   - {c}")
        print("\nColumnas disponibles:")
        for c in df.columns:
            print(f"   · {c}")
        sys.exit(1)

    df_out = df.drop(columns=DROP_COLS)

    expected = 48  # solo si features_68 tenía exactamente esas 20 + 48
    n_feat_like = len(df_out.columns)
    print(f"   Columnas tras eliminar: {n_feat_like} (se quitaron {len(DROP_COLS)})")

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

    print("Listo. No se modificaron filas ni el resto de columnas.")


if __name__ == "__main__":
    main()