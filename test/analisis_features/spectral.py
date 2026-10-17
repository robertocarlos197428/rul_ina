
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
    r"C:\Users\a\Downloads\CursoDL\rul_ina\analisis_features\spectrum"
)


# ============================================================
# CONFIGURACIÓN
# ============================================================

SPECTRAL_FEATURES = [
    "SpectralCentroid",
    "SpectralRMS",
    "SpectralStd",
    "SpectralEntropy",
    "SpectralFlatness",
]

AXES = {
    "H": [f"{feature}_H" for feature in SPECTRAL_FEATURES],
    "V": [f"{feature}_V" for feature in SPECTRAL_FEATURES],
}

N_ROWS = 17
N_COLS = 5
DPI = 200


# ============================================================
# LECTURA Y VALIDACIÓN
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
        *AXES["H"],
        *AXES["V"],
    }

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
    # GENERACIÓN DE LAS DOS IMÁGENES
    # ========================================================

    for axis, features in AXES.items():

        fig, axes = plt.subplots(
            nrows=N_ROWS,
            ncols=N_COLS,
            figsize=(22, 42),
            sharex=False,
            squeeze=False,
        )

        fig.suptitle(
            f"Spectral Features - Axis {axis}",
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

                ax.plot(
                    bearing_df["time_hours"],
                    bearing_df[feature],
                    linewidth=0.7,
                )

                if row == 0:
                    ax.set_title(
                        SPECTRAL_FEATURES[col],
                        fontsize=10,
                    )

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

        output_file = OUTPUT_DIR / f"spectrum_{axis}.png"

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
