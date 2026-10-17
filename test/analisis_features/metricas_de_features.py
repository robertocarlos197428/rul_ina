
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

PARQUET_PATH = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina"
    r"\data\processed\features_68.parquet"
)

OUTPUT_DIR = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina"
    r"\analisis_features\metricas_hi"
)

# Cinco valores para evaluar la sensibilidad de Robustness.
ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.50]

# Ventanas provisionales para Prognosability.
ONSET_FRACTION = 0.10
FAILURE_FRACTION = 0.10

# Columnas que no son features.
METADATA_COLUMNS = {
    "bearing_id",
    "burst_id",
    "burst_index",
    "time_seconds",
    "time_hours",
}

MIN_OBSERVATIONS = 3


# ============================================================
# 2. FUNCIONES DE MÉTRICAS
# ============================================================

def monotonicity_sign(x):
    """
    Monotonicity por diferencias de signo.
    Rango teórico: [0, 1].
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan

    differences = np.diff(x)

    return abs(np.sign(differences).sum()) / len(differences)


def monotonicity_spearman(x, t):
    """
    Valor absoluto de la correlación de Spearman
    entre la feature y el tiempo.
    Rango: [0, 1].
    """
    x = pd.Series(np.asarray(x, dtype=float))
    t = pd.Series(np.asarray(t, dtype=float))

    valid = x.notna() & t.notna()
    x = x[valid]
    t = t[valid]

    if len(x) < MIN_OBSERVATIONS:
        return np.nan

    if x.nunique() < 2 or t.nunique() < 2:
        return np.nan

    return abs(x.corr(t, method="spearman"))


def trendability(x, t):
    """
    Correlación de Pearson absoluta entre feature y tiempo.
    Rango: [0, 1].
    """
    x = pd.Series(np.asarray(x, dtype=float))
    t = pd.Series(np.asarray(t, dtype=float))

    valid = x.notna() & t.notna()
    x = x[valid]
    t = t[valid]

    if len(x) < MIN_OBSERVATIONS:
        return np.nan

    if x.nunique() < 2 or t.nunique() < 2:
        return np.nan

    return abs(x.corr(t, method="pearson"))


def robustness_ewma(x, alpha):
    """
    Robustness respecto de una tendencia EWMA.

    Se utiliza:
        EWMA_t = alpha * X_t + (1-alpha) * EWMA_(t-1)

    Para evitar división por cero, se utiliza un denominador
    mínimo proporcional a la escala de la feature.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < MIN_OBSERVATIONS:
        return np.nan

    series = pd.Series(x)

    smoothed = series.ewm(
        alpha=alpha,
        adjust=False
    ).mean().to_numpy()

    scale = np.median(np.abs(x))
    epsilon = max(
        scale * 1e-12,
        np.finfo(float).eps
    )

    denominator = np.maximum(np.abs(x), epsilon)

    relative_deviation = (
        np.abs(x - smoothed) / denominator
    )

    values = np.exp(-relative_deviation)

    return float(np.mean(values))


def prognosability(initial_values, final_values):
    """
    Prognosability original:
        exp(-std(valores_finales) /
            abs(media_inicial - media_final))

    Se calcula entre rodamientos, no individualmente.
    """
    initial_values = np.asarray(initial_values, dtype=float)
    final_values = np.asarray(final_values, dtype=float)

    valid = (
        np.isfinite(initial_values)
        & np.isfinite(final_values)
    )

    initial_values = initial_values[valid]
    final_values = final_values[valid]

    if len(initial_values) < 2:
        return np.nan

    initial_mean = np.mean(initial_values)
    final_mean = np.mean(final_values)

    denominator = abs(initial_mean - final_mean)

    if np.isclose(denominator, 0.0):
        return np.nan

    final_std = np.std(final_values, ddof=0)

    return float(np.exp(-final_std / denominator))


# ============================================================
# 3. LEER PARQUET EN MODO SOLO LECTURA
# ============================================================

if not PARQUET_PATH.is_file():
    raise FileNotFoundError(
        f"No se encontró el Parquet:\n{PARQUET_PATH}"
    )

# Solo lectura: no se escribe ni se modifica este archivo.
df = pd.read_parquet(PARQUET_PATH)

required_columns = {"bearing_id"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        "El Parquet debe contener la columna bearing_id."
    )

if "time_seconds" in df.columns:
    TIME_COLUMN = "time_seconds"
elif "burst_index" in df.columns:
    TIME_COLUMN = "burst_index"
else:
    raise ValueError(
        "No se encontró time_seconds ni burst_index."
    )

feature_columns = [
    column
    for column in df.select_dtypes(include=[np.number]).columns
    if column not in METADATA_COLUMNS
]

if not feature_columns:
    raise ValueError("No se encontraron features numéricas.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Features detectadas: {len(feature_columns)}")
print(f"Rodamientos: {df['bearing_id'].nunique()}")
print(f"Columna temporal: {TIME_COLUMN}")
print("Parquet abierto exclusivamente para lectura.")


# ============================================================
# 4. EVALUAR CADA FEATURE Y RODAMIENTO
# ============================================================

detail_rows = []
prognosability_by_feature = {}

bearings = sorted(
    df["bearing_id"].dropna().unique(),
    key=str
)

for feature_index, feature in enumerate(feature_columns, start=1):

    initial_values = []
    final_values = []

    feature_rows = []

    for bearing in bearings:

        group = df.loc[
            df["bearing_id"] == bearing,
            [TIME_COLUMN, feature]
        ].copy()

        group = group.replace(
            [np.inf, -np.inf], np.nan
        )

        group = group.dropna(
            subset=[TIME_COLUMN, feature]
        )

        group = group.sort_values(TIME_COLUMN)

        x = group[feature].to_numpy(dtype=float)
        t = group[TIME_COLUMN].to_numpy(dtype=float)

        row = {
            "feature": feature,
            "bearing_id": bearing,
            "n_observations": len(x),
        }

        if len(x) < MIN_OBSERVATIONS:
            row["Mon_sign"] = np.nan
            row["Mon_spearman"] = np.nan
            row["Trend_pearson"] = np.nan

            for alpha in ALPHAS:
                row[f"Rob_alpha_{alpha:.2f}"] = np.nan

            feature_rows.append(row)
            continue

        # Monotonicity: ambas definiciones.
        row["Mon_sign"] = monotonicity_sign(x)
        row["Mon_spearman"] = monotonicity_spearman(x, t)

        # Trendability: correlación con el tiempo.
        row["Trend_pearson"] = trendability(x, t)

        # Robustness: cinco valores de alpha.
        for alpha in ALPHAS:
            row[f"Rob_alpha_{alpha:.2f}"] = (
                robustness_ewma(x, alpha)
            )

        # Ventanas inicial y final para Prognosability.
        window_initial = max(
            1, int(np.ceil(len(x) * ONSET_FRACTION))
        )

        window_final = max(
            1, int(np.ceil(len(x) * FAILURE_FRACTION))
        )

        initial_mean = float(
            np.mean(x[:window_initial])
        )

        final_mean = float(
            np.mean(x[-window_final:])
        )

        initial_values.append(initial_mean)
        final_values.append(final_mean)

        row["initial_window_mean"] = initial_mean
        row["final_window_mean"] = final_mean

        feature_rows.append(row)

    detail_rows.extend(feature_rows)

    prognosability_by_feature[feature] = prognosability(
        initial_values,
        final_values
    )

    print(
        f"[{feature_index}/{len(feature_columns)}] "
        f"Evaluada: {feature}"
    )


# ============================================================
# 5. TABLA DETALLADA POR RODAMIENTO
# ============================================================

detail_df = pd.DataFrame(detail_rows)

detail_df["Prognosability"] = (
    detail_df["feature"]
    .map(prognosability_by_feature)
)

rob_columns = [
    f"Rob_alpha_{alpha:.2f}"
    for alpha in ALPHAS
]

# Diferencias entre los cinco alphas para cada rodamiento.
detail_df["Rob_range_alpha"] = (
    detail_df[rob_columns].max(axis=1)
    - detail_df[rob_columns].min(axis=1)
)


# ============================================================
# 6. TABLA RESUMEN POR FEATURE
# ============================================================

summary_rows = []

metrics_to_summarize = [
    "Mon_sign",
    "Mon_spearman",
    "Trend_pearson",
] + rob_columns

for feature in feature_columns:

    subset = detail_df.loc[
        detail_df["feature"] == feature
    ]

    summary = {
        "feature": feature,
        "Prognosability": prognosability_by_feature[feature],
        "rodamientos_evaluados": int(
            subset["bearing_id"].nunique()
        ),
    }

    # Mediana, media, desviación estándar, mínimo y máximo
    # de los resultados individuales entre rodamientos.
    for metric in metrics_to_summarize:

        values = subset[metric].dropna()

        summary[f"{metric}_mediana"] = (
            values.median() if len(values) else np.nan
        )

        summary[f"{metric}_media"] = (
            values.mean() if len(values) else np.nan
        )

        summary[f"{metric}_sd_rodamientos"] = (
            values.std(ddof=1)
            if len(values) >= 2
            else np.nan
        )

        summary[f"{metric}_min"] = (
            values.min() if len(values) else np.nan
        )

        summary[f"{metric}_max"] = (
            values.max() if len(values) else np.nan
        )

    # Sensibilidad de la mediana de Robustness al alpha.
    robustness_medians = [
        summary[f"Rob_alpha_{alpha:.2f}_mediana"]
        for alpha in ALPHAS
    ]

    valid_medians = [
        value for value in robustness_medians
        if pd.notna(value)
    ]

    summary["Rob_mediana_entre_alphas"] = (
        np.median(valid_medians)
        if valid_medians else np.nan
    )

    summary["Rob_rango_entre_alphas"] = (
        max(valid_medians) - min(valid_medians)
        if valid_medians else np.nan
    )

    summary_rows.append(summary)

summary_df = pd.DataFrame(summary_rows)


# ============================================================
# 7. LEYENDA METODOLÓGICA
# ============================================================

legend_rows = [
    {
        "Metrica": "Mon_sign",
        "Definicion": (
            "Abs(suma(sign(diff(X)))) / (N-1)"
        ),
        "Limite_inferior": 0,
        "Limite_superior": 1,
        "Mejor_direccion": "Mayor",
        "Interpretacion": (
            "1 indica cambios predominantemente "
            "en una dirección."
        ),
        "Precauciones": (
            "Los empates aportan cero al numerador, "
            "pero permanecen en el denominador."
        ),
    },
    {
        "Metrica": "Mon_spearman",
        "Definicion": (
            "Abs(correlacion de Spearman entre X y tiempo)"
        ),
        "Limite_inferior": 0,
        "Limite_superior": 1,
        "Mejor_direccion": "Mayor",
        "Interpretacion": (
            "1 indica relación monótona perfecta "
            "con el tiempo."
        ),
        "Precauciones": (
            "No distingue entre tendencia creciente y decreciente."
        ),
    },
    {
        "Metrica": "Trend_pearson",
        "Definicion": (
            "Abs(correlacion de Pearson entre X y tiempo)"
        ),
        "Limite_inferior": 0,
        "Limite_superior": 1,
        "Mejor_direccion": "Mayor",
        "Interpretacion": (
            "1 indica asociación lineal perfecta con el tiempo."
        ),
        "Precauciones": (
            "Una tendencia no lineal puede obtener un valor menor."
        ),
    },
    {
        "Metrica": "Prognosability",
        "Definicion": (
            "exp(-std(valores finales) / "
            "abs(media inicial - media final))"
        ),
        "Limite_inferior": 0,
        "Limite_superior": 1,
        "Mejor_direccion": "Mayor",
        "Interpretacion": (
            "Un valor alto indica valores finales consistentes "
            "respecto al cambio entre estados inicial y final."
        ),
        "Precauciones": (
            "Se calcula entre rodamientos. Se usan ventanas "
            "inicial y final del 10 % como supuesto configurable."
        ),
    },
    {
        "Metrica": "Robustness",
        "Definicion": (
            "Media de exp(-abs(X-EWMA(X))/abs(X))"
        ),
        "Limite_inferior": 0,
        "Limite_superior": 1,
        "Mejor_direccion": "Mayor, con cautela",
        "Interpretacion": (
            "Mayor cercanía a la tendencia EWMA según "
            "la desviación relativa definida."
        ),
        "Precauciones": (
            "Depende de alpha y de la escala de X. "
            "Se usa un denominador mínimo para evitar división "
            "por cero. Valores altos no garantizan un buen HI."
        ),
    },
    {
        "Metrica": "Rob_rango_entre_alphas",
        "Definicion": (
            "Máxima mediana de Robustness menos mínima mediana "
            "entre los cinco alphas."
        ),
        "Limite_inferior": 0,
        "Limite_superior": 1,
        "Mejor_direccion": "Menor",
        "Interpretacion": (
            "Un rango pequeño indica menor sensibilidad "
            "a la elección de alpha."
        ),
        "Precauciones": (
            "No es una métrica estándar de calidad del HI; "
            "es un indicador de sensibilidad."
        ),
    },
]

legend_df = pd.DataFrame(legend_rows)

legend_extra = pd.DataFrame([
    {
        "Metrica": "Parametros del analisis",
        "Definicion": (
            f"ALPHAS={ALPHAS}; "
            f"ONSET_FRACTION={ONSET_FRACTION}; "
            f"FAILURE_FRACTION={FAILURE_FRACTION}"
        ),
        "Limite_inferior": np.nan,
        "Limite_superior": np.nan,
        "Mejor_direccion": "No aplica",
        "Interpretacion": (
            "Los resultados dependen de estas decisiones."
        ),
        "Precauciones": (
            "Los límites teóricos no constituyen umbrales "
            "universales para declarar una feature buena."
        ),
    }
])

legend_df = pd.concat(
    [legend_df, legend_extra],
    ignore_index=True
)


# ============================================================
# 8. GUARDAR RESULTADOS EN ARCHIVOS NUEVOS
# ============================================================

summary_csv = OUTPUT_DIR / "metricas_hi_resumen.csv"
detail_csv = OUTPUT_DIR / "metricas_hi_por_rodamiento.csv"
legend_csv = OUTPUT_DIR / "metricas_hi_leyenda.csv"
excel_path = OUTPUT_DIR / "metricas_hi_resultados.xlsx"

summary_df.to_csv(
    summary_csv, index=False, encoding="utf-8-sig"
)

detail_df.to_csv(
    detail_csv, index=False, encoding="utf-8-sig"
)

legend_df.to_csv(
    legend_csv, index=False, encoding="utf-8-sig"
)

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    summary_df.to_excel(
        writer, sheet_name="Resumen_features", index=False
    )

    detail_df.to_excel(
        writer, sheet_name="Por_rodamiento", index=False
    )

    legend_df.to_excel(
        writer, sheet_name="Leyenda", index=False
    )

    for sheet_name, worksheet in writer.sheets.items():
        worksheet.freeze_panes = "B2"
        worksheet.auto_filter.ref = worksheet.dimensions

        # Ajuste moderado del ancho de columnas.
        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter

            max_length = max(
                len(str(cell.value))
                if cell.value is not None else 0
                for cell in column_cells[:100]
            )

            worksheet.column_dimensions[column_letter].width = (
                min(max(max_length + 2, 12), 42)
            )


# ============================================================
# 9. RESUMEN DE EJECUCIÓN
# ============================================================

print("\nANÁLISIS TERMINADO")
print(f"Features evaluadas: {len(feature_columns)}")
print(f"Rodamientos: {len(bearings)}")
print(f"Valores de alpha: {ALPHAS}")

print("\nArchivos creados:")
print(summary_csv)
print(detail_csv)
print(legend_csv)
print(excel_path)

print("\nEl Parquet original no fue modificado.")
