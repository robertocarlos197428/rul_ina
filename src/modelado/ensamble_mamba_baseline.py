"""
Ensamble Mamba + baseline lineal (sin reentrenar nada).

Usa las predicciones ya guardadas:
  - Mamba, por experimento y semilla : modelado/02_mamba/resultados/semilla17/<exp>_seed<N>/folds/<Bearing>_pred.csv
  - Baseline lineal (promedio_50)    : modelado/01_baseline_lineal/resultados/promedio_50/predicciones.csv

Para cada experimento de Mamba y cada semilla:
    pred_ens = W * pred_mamba + (1 - W) * pred_baseline
y se calcula el MAE/RMSE por rodamiento. Luego se promedia entre semillas.

Decision a priori: W_PRINCIPAL = 0.5 (promedio simple). El barrido de pesos
(PESOS_BARRIDO) es solo de sensibilidad: NO se elige el mejor peso mirando los
mismos rodamientos de prueba, porque eso seria optimizar sobre el test.

Salida (modelado/03_ensamble/resultados/):
  - errores_por_rodamiento.csv  : MAE de baseline, Mamba y ensamble por rodamiento
  - resumen_pesos.csv           : macro/micro por experimento y peso
  - predicciones_ensamble.csv   : curvas del ensamble (W_PRINCIPAL, media de semillas)
  - ensamble_mae.png            : barras por rodamiento
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


# ============================================================
# CONFIGURACION
# ============================================================

def encontrar_raiz_proyecto(inicio: Path) -> Path:
    inicio = inicio.resolve()
    candidatos = [inicio, *inicio.parents]
    for carpeta in candidatos:
        if (carpeta / "data" / "processed").is_dir() and (carpeta / ".git").exists():
            return carpeta
    for carpeta in candidatos:
        if (carpeta / "data" / "processed").is_dir() and (carpeta / "modelado").is_dir():
            return carpeta
    raise FileNotFoundError(
        "No se pudo detectar la raiz de rul_ina. Coloca este script dentro del "
        "proyecto y verifica que existan 'data/processed' y 'modelado'."
    )


BASE_DIR = encontrar_raiz_proyecto(Path(__file__).parent)
MODELADO = BASE_DIR / "modelado"
SEMILLAS_DIR = MODELADO / "02_mamba" / "resultados" / "semilla17"
BASELINE_PRED = (
    MODELADO / "01_baseline_lineal" / "resultados" / "promedio_50" / "predicciones.csv"
)
OUT_DIR = MODELADO / "03_ensamble" / "resultados"

# nombre en las tablas -> prefijo de las carpetas dentro de semillas/
EXPERIMENTOS = {
    "exp2_mse": "exp2",
    "exp3_smoothl1": "exp3",
}

W_PRINCIPAL = 0.5                           # peso de Mamba en el ensamble principal
PESOS_BARRIDO = [0.0, 0.25, 0.5, 0.75, 1.0]  # 0 = solo baseline, 1 = solo Mamba

BEARING_COL = "bearing_id"
TIME_COL = "time_seconds"
REAL_COL = "rul_real"
PRED_COL = "rul_pred"
DPI = 150


# ============================================================
# CARGA
# ============================================================

def leer_predicciones(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    faltan = [c for c in (BEARING_COL, TIME_COL, REAL_COL, PRED_COL) if c not in df.columns]
    if faltan:
        print(f"❌ En {ruta} faltan columnas: {faltan}")
        print(f"   Columnas encontradas: {list(df.columns)}")
        sys.exit(1)
    return df[[BEARING_COL, TIME_COL, REAL_COL, PRED_COL]]


def cargar_mamba_por_semilla(prefijo: str) -> dict:
    """Devuelve {nombre_carpeta: DataFrame} con las predicciones de cada semilla."""
    carpetas = sorted(
        c for c in SEMILLAS_DIR.iterdir()
        if c.is_dir() and c.name.lower().startswith(prefijo)
        and (c / "folds").is_dir()
    )
    resultado = {}
    for c in carpetas:
        archivos = sorted((c / "folds").glob("*_pred.csv"))
        if not archivos:
            print(f"⚠️  {c.name}: carpeta folds/ sin archivos *_pred.csv, se salta")
            continue
        if len(archivos) != 17:
            print(f"⚠️  {c.name}: tiene {len(archivos)} folds (se esperaban 17)")
        resultado[c.name] = pd.concat(
            [leer_predicciones(a) for a in archivos], ignore_index=True
        )
    return resultado


# ============================================================
# METRICAS
# ============================================================

def metricas(df: pd.DataFrame, col_pred: str):
    """MAE/RMSE por rodamiento, macro (promedio de rodamientos) y micro (global)."""
    e = df[col_pred] - df[REAL_COL]
    por_b = (
        pd.DataFrame({BEARING_COL: df[BEARING_COL], "ae": e.abs(), "se": e ** 2})
        .groupby(BEARING_COL)
        .agg(MAE=("ae", "mean"), MSE=("se", "mean"))
    )
    por_b["RMSE"] = np.sqrt(por_b["MSE"])
    por_b = por_b[["MAE", "RMSE"]]
    macro = por_b["MAE"].mean()
    micro = float(e.abs().mean())
    return por_b, macro, micro


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not BASELINE_PRED.exists():
        print(f"❌ No se encuentra el baseline:\n   {BASELINE_PRED}")
        sys.exit(1)
    if not SEMILLAS_DIR.is_dir():
        print(f"❌ No se encuentra la carpeta de semillas:\n   {SEMILLAS_DIR}")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = leer_predicciones(BASELINE_PRED).rename(columns={PRED_COL: "pred_base"})
    print(f"📂 Baseline: {BASELINE_PRED}  ({base[BEARING_COL].nunique()} rodamientos)")

    filas_resumen = []
    tablas_b = {}          # exp -> DataFrame por rodamiento (media entre semillas)
    curvas_principal = []  # predicciones del ensamble principal

    for exp, prefijo in EXPERIMENTOS.items():
        por_semilla = cargar_mamba_por_semilla(prefijo)
        if not por_semilla:
            print(f"⚠️  {exp}: no hay carpetas '{prefijo}*' con predicciones.csv, se salta")
            continue
        print(f"\n▶ {exp}: {len(por_semilla)} semillas -> {list(por_semilla)}")

        # Unir cada semilla con el baseline sobre (rodamiento, tiempo)
        unidos = {}
        for nombre, dfm in por_semilla.items():
            m = dfm.merge(base[[BEARING_COL, TIME_COL, "pred_base"]],
                          on=[BEARING_COL, TIME_COL], how="inner")
            if len(m) != len(dfm) or len(m) != len(base):
                print(f"   ⚠️  {nombre}: filas Mamba={len(dfm)}, baseline={len(base)}, "
                      f"comunes={len(m)} (se evalua solo sobre las comunes)")
            unidos[nombre] = m

        # Barrido de pesos: metricas por semilla, luego media y desviacion entre semillas
        for w in PESOS_BARRIDO:
            macros, micros = [], []
            for nombre, m in unidos.items():
                m = m.copy()
                m["pred_ens"] = w * m[PRED_COL] + (1 - w) * m["pred_base"]
                _, macro, micro = metricas(m, "pred_ens")
                macros.append(macro)
                micros.append(micro)
            filas_resumen.append({
                "experimento": exp, "peso_mamba": w, "n_semillas": len(unidos),
                "MAE_macro_media": np.mean(macros), "MAE_macro_desv": np.std(macros),
                "MAE_micro_media": np.mean(micros), "MAE_micro_desv": np.std(micros),
            })

        # Detalle por rodamiento: baseline, Mamba solo y ensamble principal
        acum = {"base": [], "mamba": [], "ens": []}
        for nombre, m in unidos.items():
            m = m.copy()
            m["pred_ens"] = W_PRINCIPAL * m[PRED_COL] + (1 - W_PRINCIPAL) * m["pred_base"]
            acum["base"].append(metricas(m, "pred_base")[0]["MAE"])
            acum["mamba"].append(metricas(m, PRED_COL)[0]["MAE"])
            acum["ens"].append(metricas(m, "pred_ens")[0]["MAE"])

        tabla = pd.DataFrame({
            "MAE_baseline": pd.concat(acum["base"], axis=1).mean(axis=1),
            "MAE_mamba": pd.concat(acum["mamba"], axis=1).mean(axis=1),
            "MAE_mamba_desv": pd.concat(acum["mamba"], axis=1).std(axis=1, ddof=0),
            "MAE_ensamble": pd.concat(acum["ens"], axis=1).mean(axis=1),
            "MAE_ensamble_desv": pd.concat(acum["ens"], axis=1).std(axis=1, ddof=0),
        })
        tabla["ens_mejor_que_baseline"] = tabla["MAE_ensamble"] < tabla["MAE_baseline"]
        tabla["ens_mejor_que_mamba"] = tabla["MAE_ensamble"] < tabla["MAE_mamba"]
        tablas_b[exp] = tabla

        # Curva del ensamble principal: media de semillas
        cat = pd.concat(unidos.values(), ignore_index=True)
        curva = (
            cat.groupby([BEARING_COL, TIME_COL], as_index=False)
            .agg(rul_real=(REAL_COL, "first"),
                 rul_mamba=(PRED_COL, "mean"), rul_baseline=("pred_base", "first"))
        )
        curva["rul_ensamble"] = (W_PRINCIPAL * curva["rul_mamba"]
                                 + (1 - W_PRINCIPAL) * curva["rul_baseline"])
        curva.insert(0, "experimento", exp)
        curvas_principal.append(curva)

    if not tablas_b:
        print("❌ No se pudo armar ningun experimento.")
        sys.exit(1)

    # --------------------------------------------------------
    # Guardar
    # --------------------------------------------------------
    resumen = pd.DataFrame(filas_resumen)
    resumen.to_csv(OUT_DIR / "resumen_pesos.csv", index=False)

    por_b = pd.concat(tablas_b, names=["experimento", BEARING_COL]).reset_index()
    por_b.to_csv(OUT_DIR / "errores_por_rodamiento.csv", index=False)
    pd.concat(curvas_principal, ignore_index=True).to_csv(
        OUT_DIR / "predicciones_ensamble.csv", index=False)

    # --------------------------------------------------------
    # Mostrar
    # --------------------------------------------------------
    print("\n=== Resumen por peso (peso_mamba: 0 = baseline, 1 = Mamba) ===")
    print(resumen.round(4).to_string(index=False))

    for exp, tabla in tablas_b.items():
        n = len(tabla)
        print(f"\n=== {exp}, W={W_PRINCIPAL}: ensamble vs baseline y vs Mamba ===")
        print(f"   Mejor que el baseline en {int(tabla['ens_mejor_que_baseline'].sum())}/{n} rodamientos")
        print(f"   Mejor que Mamba solo en  {int(tabla['ens_mejor_que_mamba'].sum())}/{n} rodamientos")

    # --------------------------------------------------------
    # Grafico: barras por rodamiento (baseline, Mamba, ensamble)
    # --------------------------------------------------------
    exps = list(tablas_b)
    fig, axes = plt.subplots(len(exps), 1, figsize=(15, 4.2 * len(exps)),
                             sharey=True, squeeze=False)
    for ax, exp in zip(axes[:, 0], exps):
        t = tablas_b[exp]
        x = np.arange(len(t))
        ancho = 0.27
        ax.bar(x - ancho, t["MAE_baseline"], ancho, label="Baseline lineal", color="gray")
        ax.bar(x, t["MAE_mamba"], ancho, yerr=t["MAE_mamba_desv"], capsize=2,
               label=f"Mamba ({exp})", color="tab:blue")
        ax.bar(x + ancho, t["MAE_ensamble"], ancho, yerr=t["MAE_ensamble_desv"], capsize=2,
               label=f"Ensamble W={W_PRINCIPAL}", color="tab:green")
        ax.set_xticks(x)
        ax.set_xticklabels(t.index, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("MAE (rul_frac)")
        ax.set_title(f"{exp}: MAE por rodamiento (media ± desv entre semillas)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ensamble_mae.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✅ Resultados guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()