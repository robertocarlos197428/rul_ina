
"""
Mamba pequeño (PyTorch puro, CPU) con validación LOBO y múltiples semillas
para comparar Experimento 2 (MSE) y Experimento 3 (Smooth L1).

- Entrada : features_13_rul.parquet (13 features + rul_frac)
- Procesamiento: 13 features + 13 diferencias finitas (delta f) = 26 features.
- Experimentos: exp2 (MSE) y exp3 (Smooth L1 con beta=0.1).
- Semillas: 42, 123, 456.
- Salida  : resultados/semillas/<exp>_seed<semilla>/
"""

from pathlib import Path
import math
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        "No se pudo detectar la raíz de rul_ina. "
        "Coloca este script dentro del proyecto y verifica que existan "
        "'data/processed' y 'modelado'."
    )

BASE_DIR = encontrar_raiz_proyecto(Path(__file__).parent)
INPUT_FILE = BASE_DIR / "data" / "processed" / "features_13_rul.parquet"
RESULTS_DIR = BASE_DIR / "modelado" / "02_mamba" / "resultados" / "semillas"
BASELINE_ERRORES = (
    BASE_DIR / "modelado" / "01_baseline_lineal" / "resultados"
    / "promedio_50" / "errores_por_rodamiento.csv"
)

# Datos
LONG_SECUENCIA = 50
VENTANA_SUAVIZADO = 10

# Modelo
D_IN = 26
D_MODEL = 32
N_CAPAS = 2
D_STATE = 16
EXPAND = 2
D_CONV = 4

# Entrenamiento
EPOCAS = 20
PASO_ENTRENAMIENTO = 10
BATCH = 256
LR = 2e-3
SEMILLAS = [42, 123, 456]
EXPERIMENTOS = {
    "exp2_mse": "mse",
    "exp3_smoothl1": "smooth_l1"
}

LIMITE_RODAMIENTOS = 6  # 6 rodamientos para pruebas rápidas robustas

BEARING_COL = "bearing_id"
TIME_COL = "time_seconds"
TARGET_COL = "rul_frac"

FEATURES = [
    "RMS_H", "RMS_V",
    "Kurtosis_H", "Kurtosis_V",
    "CrestFactor_H", "CrestFactor_V",
    "SpectralStd_H",
    "WPD_Erel_5_H", "WPD_Erel_3_H", "WPD_Erel_6_H",
    "Skewness_H",
    "WPD_Erel_5_V", "WPD_Erel_4_H",
]

assert D_IN == 2 * len(FEATURES), (
    "D_IN debe ser el doble del número de features originales."
)

DPI = 150


# ============================================================
# MODELO
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state, expand, d_conv):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.dt_rank = max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        B, L, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_in = F.silu(x_in)

        dt, Bm, Cm = torch.split(
            self.x_proj(x_in), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        dA = torch.exp(dt.unsqueeze(-1) * A)
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x_in.unsqueeze(-1)

        h = torch.zeros(B, self.d_inner, self.d_state, dtype=x.dtype, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x_in * self.D

        y = y * F.silu(z)
        return self.out_proj(y)


class MambaRUL(nn.Module):
    def __init__(self, d_in, d_model, n_layers, d_state, expand, d_conv):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.bloques = nn.ModuleList(
            [MambaBlock(d_model, d_state, expand, d_conv) for _ in range(n_layers)]
        )
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        h = self.embed(x)
        for norm, bloque in zip(self.norms, self.bloques):
            h = h + bloque(norm(h))
        return self.head(self.norm_f(h[:, -1])).squeeze(-1)


# ============================================================
# DATOS
# ============================================================

def construir_ventanas(X, L):
    n = len(X)
    idx = np.arange(n)[:, None] + np.arange(-(L - 1), 1)[None, :]
    idx = np.clip(idx, 0, n - 1)
    return X[idx]


def preparar_datos(df):
    requeridas = [BEARING_COL, TIME_COL, TARGET_COL, *FEATURES]
    faltantes = [col for col in requeridas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el Parquet: {faltantes}")

    df = df.sort_values([BEARING_COL, TIME_COL]).reset_index(drop=True)
    if df[requeridas].isna().any().any():
        raise ValueError(
            "Hay valores nulos en las columnas requeridas. "
            "Revisa los datos antes de entrenar."
        )

    if VENTANA_SUAVIZADO > 1:
        df = df.copy()
        df[FEATURES] = (
            df.groupby(BEARING_COL)[FEATURES]
            .transform(lambda s: s.rolling(VENTANA_SUAVIZADO, min_periods=1).mean())
        )

    datos = {}
    for b, part in df.groupby(BEARING_COL, sort=True):
        X_smoothed = part[FEATURES].to_numpy(dtype=np.float32)
        X_diff = np.diff(X_smoothed, axis=0, prepend=X_smoothed[:1, :])
        X = np.concatenate([X_smoothed, X_diff], axis=-1)
        
        datos[b] = {
            "X": X,
            "W": construir_ventanas(X, LONG_SECUENCIA),
            "y": part[TARGET_COL].to_numpy(dtype=np.float32),
            "tiempo": part[TIME_COL].to_numpy(dtype=float),
        }
    return datos


# ============================================================
# ENTRENAMIENTO Y PREDICCION
# ============================================================

def entrenar_y_predecir(datos, bearing_test, semilla, tipo_perdida):
    train_ids = [b for b in datos if b != bearing_test]
    if not train_ids:
        raise ValueError(
            "LOBO requiere al menos un rodamiento para entrenamiento "
            "además del de prueba."
        )

    X_rows = np.concatenate([datos[b]["X"] for b in train_ids])
    media = X_rows.mean(axis=0)
    desv = X_rows.std(axis=0)
    desv[desv == 0] = 1.0

    W_tr = np.concatenate([datos[b]["W"] for b in train_ids])
    y_tr = np.concatenate([datos[b]["y"] for b in train_ids])
    W_tr = torch.from_numpy(((W_tr - media) / desv).astype(np.float32))
    y_tr = torch.from_numpy(y_tr)

    W_te = torch.from_numpy(((datos[bearing_test]["W"] - media) / desv).astype(np.float32))

    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)

    model = MambaRUL(
        d_in=D_IN,
        d_model=D_MODEL,
        n_layers=N_CAPAS,
        d_state=D_STATE,
        expand=EXPAND,
        d_conv=D_CONV
    )
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS)

    n_total = len(W_tr)
    perdidas = []
    for epoca in range(1, EPOCAS + 1):
        model.train()
        inicio = (epoca - 1) % PASO_ENTRENAMIENTO
        perm = rng.permutation(np.arange(inicio, n_total, PASO_ENTRENAMIENTO))
        n = len(perm)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = torch.from_numpy(perm[i:i + BATCH])
            pred = model(W_tr[idx])
            
            if tipo_perdida == "mse":
                loss = F.mse_loss(pred, y_tr[idx])
            else:
                loss = F.smooth_l1_loss(pred, y_tr[idx], beta=0.1)
                
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        perdida = total / n
        perdidas.append(perdida)

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(W_te), 1024):
            preds.append(model(W_te[i:i + 1024]))
    pred = torch.cat(preds).numpy()
    return pred, perdidas


# ============================================================
# GRAFICO CON MULTIPLES SEMILLAS (MEDIA ± DESV)
# ============================================================

def graficar_experimento_semillas(exp_name, semillas_data, destino, titulo):
    bearings = sorted(list(semillas_data[SEMILLAS[0]]["pred_df"][BEARING_COL].unique()))
    n = len(bearings)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows),
        sharey=True, squeeze=False,
    )

    for ax, b in zip(axes.flat, bearings):
        # Graficar cada semilla con línea tenue y la media con banda/línea sólida
        for sem in SEMILLAS:
            part = semillas_data[sem]["pred_df"][semillas_data[sem]["pred_df"][BEARING_COL] == b]
            vida = (1.0 - part["rul_real"].to_numpy()) * 100.0
            ax.plot(vida, part["rul_pred"], color="tab:red", linewidth=0.6, alpha=0.3)

        # Calcular promedio entre semillas para este rodamiento
        preds_lista = []
        for sem in SEMILLAS:
            part = semillas_data[sem]["pred_df"][semillas_data[sem]["pred_df"][BEARING_COL] == b]
            preds_lista.append(part["rul_pred"].to_numpy())
        
        preds_mat = np.array(preds_lista)
        media_pred = preds_mat.mean(axis=0)
        std_pred = preds_mat.std(axis=0)
        
        part_ref = semillas_data[SEMILLAS[0]]["pred_df"][semillas_data[SEMILLAS[0]]["pred_df"][BEARING_COL] == b]
        vida_ref = (1.0 - part_ref["rul_real"].to_numpy()) * 100.0
        
        ax.plot(vida_ref, part_ref["rul_real"], color="black", linewidth=1.5, label="Real" if b == bearings[0] else "")
        ax.plot(vida_ref, media_pred, color="tab:blue", linewidth=1.2, label="Media Mamba" if b == bearings[0] else "")
        ax.fill_between(vida_ref, media_pred - std_pred, media_pred + std_pred, color="tab:blue", alpha=0.2, label="±1 Desv" if b == bearings[0] else "")
        
        maes = []
        for sem in SEMILLAS:
            err = semillas_data[sem]["errores"]
            valor = err.loc[err[BEARING_COL] == b, "MAE"]
            if not valor.empty:
                maes.append(float(valor.iloc[0]))
        mae_medio = float(np.mean(maes)) if maes else float("nan")
        ax.set_title(f"{b} (MAE ~ {mae_medio:.3f})", fontsize=10)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Vida útil (%)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("rul_frac", fontsize=9)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(destino, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Gráfico guardado: {destino}")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not INPUT_FILE.exists():
        print("❌ No se encuentra el archivo:")
        print(f"   {INPUT_FILE}")
        sys.exit(1)

    print(f"📂 Leyendo: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    datos = preparar_datos(df)
    if len(datos) < 2:
        raise ValueError("Se necesitan al menos dos rodamientos para validar con LOBO.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bearings = sorted(datos.keys())
    a_probar = bearings if LIMITE_RODAMIENTOS is None else bearings[:LIMITE_RODAMIENTOS]

    for exp_id, tipo_perdida in EXPERIMENTOS.items():
        print("\n==============================================")
        print(f"🚀 INICIANDO EXPERIMENTO: {exp_id} (Pérdida: {tipo_perdida})")
        print(f"==============================================")
        
        semillas_data = {}

        for sem in SEMILLAS:
            print(f"\n--- Semilla: {sem} ---")
            out_dir = RESULTS_DIR / f"{exp_id}_seed{sem}"
            folds_dir = out_dir / "folds"
            folds_dir.mkdir(parents=True, exist_ok=True)

            for k, b in enumerate(a_probar):
                f_pred = folds_dir / f"{b}_pred.csv"
                f_loss = folds_dir / f"{b}_loss.csv"
                if f_pred.exists() and f_loss.exists():
                    print(f"[{k + 1}/{len(a_probar)}] {b}: ya hecho (semilla {sem}), saltando...")
                    continue

                print(f"[{k + 1}/{len(a_probar)}] Entrenando rodamiento: {b} (semilla {sem})")
                t0 = time.time()
                pred, perdidas = entrenar_y_predecir(datos, b, sem + bearings.index(b), tipo_perdida)

                pd.DataFrame({
                    BEARING_COL: b,
                    TIME_COL: datos[b]["tiempo"],
                    "rul_real": datos[b]["y"],
                    "rul_pred": pred,
                }).to_csv(f_pred, index=False)
                pd.DataFrame({
                    BEARING_COL: b,
                    "epoca": range(1, len(perdidas) + 1),
                    "perdida_entrenamiento": perdidas,
                }).to_csv(f_loss, index=False)
                print(f"      ({time.time() - t0:.0f} s)")

            # Consolidar resultados de la semilla
            pred_df = pd.concat([pd.read_csv(folds_dir / f"{b}_pred.csv") for b in a_probar], ignore_index=True)
            filas = []
            for b in a_probar:
                e = pred_df.loc[pred_df[BEARING_COL] == b, "rul_pred"] - pred_df.loc[pred_df[BEARING_COL] == b, "rul_real"]
                filas.append({
                    BEARING_COL: b, "filas": len(e),
                    "MAE": float(np.mean(np.abs(e))),
                    "RMSE": float(np.sqrt(np.mean(e ** 2))),
                })
            errores = pd.DataFrame(filas)
            
            semillas_data[sem] = {
                "pred_df": pred_df,
                "errores": errores
            }

        # Generar gráfico combinado de las 3 semillas para este experimento
        grafico_path = RESULTS_DIR / f"{exp_id}_semillas_comparacion.png"
        graficar_experimento_semillas(
            exp_id, semillas_data, grafico_path,
            f"Experimento: {exp_id} (Promedio de Semillas {SEMILLAS})"
        )

    print("\n🎉 ¡Todas las ejecuciones y gráficos por experimento han finalizado con éxito!")


if __name__ == "__main__":
    main()
