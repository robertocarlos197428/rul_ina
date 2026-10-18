
from itertools import combinations
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
    r"C:\Users\a\Downloads\CursoDL\rul_ina\analisis_features\redundancy"
)

CSV_FILE = OUTPUT_DIR / "features_similares_V2.csv"
PNG_FILE = OUTPUT_DIR / "features_similares_V2.png"


# ============================================================
# CRITERIOS DE SIMILITUD
# ============================================================

SPEARMAN_THRESHOLD = 0.7

# Proporción mínima de rodamientos que deben superar el umbral.
MIN_BEARING_PROPORTION = 0.80

# Columnas que no son features.
ID_COLUMNS = {
    "bearing_id",
    "burst_id",
    "burst_index",
    "time_seconds",
    "time_hours",
}


# ============================================================
# FUNCIONES
# ============================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo:\n{INPUT_FILE}"
        )

    df = pd.read_parquet(INPUT_FILE)

    if "bearing_id" not in df.columns:
        raise ValueError("Falta la columna bearing_id.")

    # Seleccionar features numéricas, excluyendo identificadores.
    feature_columns = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in ID_COLUMNS
    ]

    if len(feature_columns) < 2:
        raise ValueError("No hay suficientes features numéricas.")

    bearings = sorted(df["bearing_id"].dropna().unique())

    if len(bearings) == 0:
        raise ValueError("No se encontraron rodamientos.")

    print(f"Rodamientos: {len(bearings)}")
    print(f"Features: {len(feature_columns)}")
    print(
        f"Parejas que se evaluarán: "
        f"{len(feature_columns) * (len(feature_columns) - 1) // 2}"
    )

    # Preparar datos por rodamiento.
    data_by_bearing = {}

    for bearing in bearings:
        part = df[df["bearing_id"] == bearing].copy()

        if "time_seconds" in part.columns:
            part = part.sort_values("time_seconds")
        elif "burst_index" in part.columns:
            part = part.sort_values("burst_index")

        data_by_bearing[bearing] = part[feature_columns]

    # ========================================================
    # COMPARAR TODAS LAS PAREJAS
    # ========================================================

    results = []

    for feature_a, feature_b in combinations(feature_columns, 2):

        spearman_values = []
        pearson_values = []

        for bearing in bearings:

            pair = data_by_bearing[bearing][
                [feature_a, feature_b]
            ].dropna()

            # Se necesitan suficientes observaciones.
            if len(pair) < 10:
                continue

            # Evitar correlaciones indefinidas.
            if (
                pair[feature_a].nunique() < 2
                or pair[feature_b].nunique() < 2
            ):
                continue

            spearman = pair[feature_a].corr(
                pair[feature_b], method="spearman"
            )

            pearson = pair[feature_a].corr(
                pair[feature_b], method="pearson"
            )

            if pd.notna(spearman):
                spearman_values.append(spearman)

            if pd.notna(pearson):
                pearson_values.append(pearson)

        if not spearman_values:
            continue

        abs_spearman = pd.Series(spearman_values).abs()

        median_abs_spearman = abs_spearman.median()

        proportion_high = (
            abs_spearman >= SPEARMAN_THRESHOLD
        ).mean()

        # Conservar solo las parejas similares en la mayoría
        # de los rodamientos.
        if (
            median_abs_spearman >= SPEARMAN_THRESHOLD
            and proportion_high >= MIN_BEARING_PROPORTION
        ):
            results.append({
                "feature_1": feature_a,
                "feature_2": feature_b,
                "mediana_abs_spearman": median_abs_spearman,
                "mediana_abs_pearson": (
                    pd.Series(pearson_values).abs().median()
                    if pearson_values else float("nan")
                ),
                "rodamientos_evaluados": len(spearman_values),
                "proporcion_similar": proportion_high,
            })

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    columns = [
        "feature_1",
        "feature_2",
        "mediana_abs_spearman",
        "mediana_abs_pearson",
        "rodamientos_evaluados",
        "proporcion_similar",
    ]

    result_df = pd.DataFrame(results, columns=columns)

    if not result_df.empty:
        result_df = result_df.sort_values(
            ["mediana_abs_spearman", "proporcion_similar"],
            ascending=False,
        )

    result_df.to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    print(f"\nCSV guardado: {CSV_FILE}")
    print(f"Parejas similares encontradas: {len(result_df)}")

    # ========================================================
    # GENERAR TABLA PNG
    # ========================================================

    if result_df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No se encontraron parejas que cumplan los criterios.",
            ha="center",
            va="center",
            fontsize=12,
        )
    else:
        display_df = result_df.copy()

        display_df["mediana_abs_spearman"] = (
            display_df["mediana_abs_spearman"].map("{:.3f}".format)
        )
        display_df["mediana_abs_pearson"] = (
            display_df["mediana_abs_pearson"].map(
                lambda x: f"{x:.3f}" if pd.notna(x) else "N/D"
            )
        )
        display_df["proporcion_similar"] = (
            display_df["proporcion_similar"].map(
                lambda x: f"{x:.0%}"
            )
        )

        display_df = display_df.rename(columns={
            "feature_1": "Feature 1",
            "feature_2": "Feature 2",
            "mediana_abs_spearman": "Spearman med.",
            "mediana_abs_pearson": "Pearson med.",
            "rodamientos_evaluados": "Rodamientos",
            "proporcion_similar": "% similares",
        })

        fig_height = max(2.5, 0.45 * len(display_df) + 1.5)

        fig, ax = plt.subplots(
            figsize=(14, fig_height)
        )
        ax.axis("off")

        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)

    fig.suptitle(
        "Features potencialmente redundantes",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(PNG_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"PNG guardado: {PNG_FILE}")
    print("\nProceso terminado.")


if __name__ == "__main__":
    main()
