
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Carpeta donde están los resultados calculados
CARPETA = Path(
    r"C:\Users\a\Downloads\CursoDL\rul_ina\analisis_features\metricas_hi"
)

ARCHIVO = CARPETA / "metricas_hi_resumen.csv"
SALIDA = CARPETA / "tabla_metricas_HI.png"

# Leer resultados existentes; no recalcular métricas
df = pd.read_csv(ARCHIVO)

columnas = {
    "feature": "Feature",
    "Mon_sign_mediana": "Mon signo",
    "Mon_spearman_mediana": "Mon Spearman",
    "Trend_pearson_mediana": "Trend",
    "Prognosability": "Prog",
    "Rob_alpha_0.05_mediana": "Rob α=0.05",
    "Rob_alpha_0.10_mediana": "Rob α=0.10",
    "Rob_alpha_0.20_mediana": "Rob α=0.20",
    "Rob_alpha_0.30_mediana": "Rob α=0.30",
    "Rob_alpha_0.50_mediana": "Rob α=0.50",
}

faltantes = [c for c in columnas if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas en el CSV: {faltantes}")

tabla = df[list(columnas)].rename(columns=columnas)

# Ordenar y redondear las métricas
tabla = tabla.sort_values("Feature").reset_index(drop=True)

for col in tabla.columns[1:]:
    tabla[col] = pd.to_numeric(tabla[col], errors="coerce").map(
        lambda x: "" if pd.isna(x) else f"{x:.3f}"
    )

# Una imagen: leyenda arriba, tabla debajo
fig, ax = plt.subplots(figsize=(18, 25))
ax.axis("off")

leyenda = (
    "LEYENDA DE MÉTRICAS HI\n"
    "Mon signo: monotonicidad basada en diferencias consecutivas.\n"
    "Mon Spearman: monotonicidad mediante correlación de rangos con el tiempo.\n"
    "Trend: valor absoluto de Pearson entre la feature y el tiempo.\n"
    "Prog: prognosability calculada con la fórmula acordada.\n"
    "Rob α: robustness usando suavizado EWMA con el alpha indicado.\n"
    "Interpretación general: valores más altos indican mejor comportamiento "
    "según cada métrica.\n"
    "Las cinco columnas Rob permiten comparar la sensibilidad al alpha."
)

fig.text(
    0.04, 0.975, leyenda,
    ha="left", va="top",
    fontsize=10,
    linespacing=1.5
)

tabla_plot = ax.table(
    cellText=tabla.values,
    colLabels=tabla.columns,
    cellLoc="center",
    colLoc="center",
    colWidths=[0.24] + [0.084] * 9,
    bbox=[0.02, 0.01, 0.96, 0.79]
)

tabla_plot.auto_set_font_size(False)
tabla_plot.set_fontsize(8)
tabla_plot.scale(1, 1.15)

# Encabezados y bordes
for (fila, columna), celda in tabla_plot.get_celld().items():
    if fila == 0:
        celda.set_text_props(weight="bold")
        celda.set_height(celda.get_height() * 1.4)
    celda.set_linewidth(0.4)

plt.savefig(SALIDA, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Filas de la tabla: {len(tabla)}")
print(f"Imagen guardada en: {SALIDA}")

