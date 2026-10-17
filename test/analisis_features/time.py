
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# RUTAS
# ============================================================

INPUT_FILE = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed\features_68.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina\analisis_features\time"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

GROUP_1 = [
    "RMS",
    "Peak",
    "P2P",
    "Variance",
    "Std",
    "MAV",
]

GROUP_2 = [
    "Kurtosis",
    "Skewness",
    "CrestFactor",
    "ImpulseFactor",
    "ShapeFactor",
    "MarginFactor",
]

GROUPS = {
    "1": GROUP_1,
    "2": GROUP_2,
}

AXES = ["H", "V"]

N_ROWS = 17
N_COLS = 6
DPI = 200


# ============================================================
# PROCESAMIENTO
# ============================================================

def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_FILE)

    required_columns = {
        "bearing_id",
        "time_seconds",
    }

    for axis in AXES:
        for group in GROUPS.values():
            required_columns.update(
                f"{feature}_{axis}" for feature in group
            )

    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(
            f"Faltan columnas en el Parquet: {sorted(missing)}"
        )

    df["time_hours"] = df["time_seconds"] / 3600.0

    bearings = sorted(df["bearing_id"].dropna().unique())

    if len(bearings) != N_ROWS:
        raise ValueError(
            f"Se esperaban {N_ROWS} rodamientos, "
            f"pero se encontraron {len(bearings)}: {bearings}"
        )

    # ========================================================
    # GENERAR LAS 4 IMÁGENES
    # ========================================================

    for axis in AXES:

        for group_id, features in GROUPS.items():

            fig, axes = plt.subplots(
                nrows=N_ROWS,
                ncols=N_COLS,
                figsize=(24, 42),
                squeeze=False,
            )

            fig.suptitle(
                f"Time-Domain Features - Axis {axis} - Group {group_id}",
                fontsize=20,
                fontweight="bold",
                y=0.999,
            )

            for row, bearing in enumerate(bearings):

                bearing_df = (
                    df[df["bearing_id"] == bearing]
                    .sort_values("time_hours")
                )

                for col, feature in enumerate(features):

                    ax = axes[row, col]
                    column = f"{feature}_{axis}"

                    ax.plot(
                        bearing_df["time_hours"],
                        bearing_df[column],
                        linewidth=0.7,
                    )

                    if row == 0:
                        ax.set_title(feature, fontsize=10)

                    if col == 0:
                        ax.set_ylabel(
                            f"{bearing}\nValor",
                            fontsize=8,
                        )

                    if row == N_ROWS - 1:
                        ax.set_xlabel("Tiempo (horas)", fontsize=8)

                    ax.tick_params(axis="both", labelsize=7)
                    ax.grid(True, alpha=0.3)

            fig.tight_layout(rect=[0, 0, 1, 0.997])

            output_file = (
                OUTPUT_DIR / f"time_{axis}_{group_id}.png"
            )

            fig.savefig(
                output_file,
                dpi=DPI,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(f"Imagen guardada: {output_file}")

    print("\nProceso terminado correctamente.")


if __name__ == "__main__":
    main()
