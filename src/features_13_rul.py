"""
Lee features_13.parquet, agrega la columna rul_frac (RUL como fraccion de vida,
de 1 a 0) y guarda features_13_rul.parquet + .csv.

Para cada rodamiento:
    avance   = (time_seconds - tiempo_inicial) / (tiempo_final - tiempo_inicial)
    rul_frac = 1 - avance
Es 1 en la primera rafaga y 0 en la ultima (la falla).

No se modifican filas, orden ni valores de las demas columnas.
"""

from pathlib import Path
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("❌ Falta pandas/numpy. Instala con: pip install pandas numpy pyarrow")
    sys.exit(1)

DATA_DIR = Path(r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed")
INPUT_FILE = DATA_DIR / "features_13.parquet"
OUTPUT_PARQUET = DATA_DIR / "features_13_rul.parquet"
OUTPUT_CSV = DATA_DIR / "features_13_rul.csv"

BEARING_COL = "bearing_id"
TIME_COL = "time_seconds"
RUL_COL = "rul_frac"


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

    for col in (BEARING_COL, TIME_COL):
        if col not in df.columns:
            print(f"❌ Falta la columna '{col}'. Columnas disponibles:")
            for c in df.columns:
                print(f"   · {c}")
            sys.exit(1)

    if RUL_COL in df.columns:
        print(f"❌ La columna '{RUL_COL}' ya existe en el archivo.")
        sys.exit(1)

    tiempo = pd.to_numeric(df[TIME_COL], errors="coerce")
    if tiempo.isna().any():
        print(f"❌ Hay {int(tiempo.isna().sum())} valores no numericos en '{TIME_COL}'.")
        sys.exit(1)

    # Tiempo inicial y final de cada rodamiento (sin reordenar filas)
    t_ini = tiempo.groupby(df[BEARING_COL]).transform("min")
    t_fin = tiempo.groupby(df[BEARING_COL]).transform("max")
    duracion = t_fin - t_ini

    if (duracion <= 0).any():
        malos = sorted(df.loc[duracion <= 0, BEARING_COL].unique())
        print(f"❌ Estos rodamientos tienen duracion 0 o negativa: {malos}")
        sys.exit(1)

    df_out = df.copy()
    df_out[RUL_COL] = 1.0 - (tiempo - t_ini) / duracion

    # Verificaciones
    assert len(df_out) == len(df), "Cambio el numero de filas"
    assert df_out[RUL_COL].between(0.0, 1.0).all(), "rul_frac fuera de [0, 1]"
    assert not df_out[RUL_COL].isna().any(), "rul_frac tiene NaN"

    print()
    print("Resumen por rodamiento:")
    resumen = df_out.groupby(BEARING_COL).agg(
        filas=(RUL_COL, "size"),
        rul_inicio=(RUL_COL, "max"),
        rul_fin=(RUL_COL, "min"),
        horas=(TIME_COL, lambda s: (s.max() - s.min()) / 3600.0),
    )
    print(resumen.round(3).to_string())
    print()

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

    print(f"Listo. Columna '{RUL_COL}' agregada ({len(df_out.columns)} columnas en total).")


if __name__ == "__main__":
    main()