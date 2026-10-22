"""
Mamba pequeno (PyTorch puro, CPU) con LOBO, usando UNA SOLA feature por corrida.

- Base    : mamba_smoothL1_suav50.py (misma arquitectura, Smooth L1, suavizado 50,
            ventanas de 50 rafagas, 20 epocas, semilla 42).
- Que hace: para cada una de las 13 features, entrena Mamba solo con esa feature
            (LOBO con los 17 rodamientos, 16 para entrenar y 1 para probar) y
            guarda una imagen con las 17 curvas (real vs predicha).
- Salida  : modelado/una_feature/
              01_RMS_H.png ... 13_WPD_Erel_4_H.png   (una imagen por feature)
              tabla_mae_por_feature.csv              (MAE por feature y rodamiento)
  No se guardan folds ni otros archivos.

Si se corta a medias, al volver a correr se saltan las features cuya imagen y
fila de la tabla ya estan hechas (la feature que estaba en curso se repite).
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
OUT_DIR = BASE_DIR / "modelado" / "una_feature"

# Datos
LONG_SECUENCIA = 50        # rafagas por ventana
VENTANA_SUAVIZADO = 50     # 1 = sin suavizar; N>1 = media causal de N rafagas

# Modelo (pequeno, pensado para una Raspberry Pi 5)
D_MODEL = 32
N_CAPAS = 2
D_STATE = 8
EXPAND = 2
D_CONV = 4

# Entrenamiento
EPOCAS = 20
PASO_ENTRENAMIENTO = 10     # cada epoca usa 1 de cada N ventanas (cambiando el
                            # punto de partida entre epocas)
BATCH = 256
LR = 2e-3
SEMILLA = 42
N_HILOS = None              # None = los que use PyTorch por defecto

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

TABLA_CSV = OUT_DIR / "tabla_mae_por_feature.csv"
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
    """Bloque Mamba (SSM selectivo) con scan secuencial en PyTorch puro."""

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

        # Inicializacion del paso de tiempo (softplus(bias) entre 1e-3 y 1e-1)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):                       # x: [B, L, d_model]
        B, L, _ = x.shape
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        x_in = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_in = F.silu(x_in)

        dt, Bm, Cm = torch.split(
            self.x_proj(x_in), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))       # [B, L, d_inner]
        A = -torch.exp(self.A_log)              # [d_inner, d_state]

        dA = torch.exp(dt.unsqueeze(-1) * A)                                  # [B,L,Di,N]
        dBx = dt.unsqueeze(-1) * Bm.unsqueeze(2) * x_in.unsqueeze(-1)         # [B,L,Di,N]

        h = torch.zeros(B, self.d_inner, self.d_state, dtype=x.dtype, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dBx[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x_in * self.D

        y = y * F.silu(z)
        return self.out_proj(y)


class MambaRUL(nn.Module):
    def __init__(self, n_feat, d_model, n_capas, d_state, expand, d_conv):
        super().__init__()
        self.embed = nn.Linear(n_feat, d_model)
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_capas)])
        self.bloques = nn.ModuleList(
            [MambaBlock(d_model, d_state, expand, d_conv) for _ in range(n_capas)]
        )
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                       # x: [B, L, n_feat]
        h = self.embed(x)
        for norm, bloque in zip(self.norms, self.bloques):
            h = h + bloque(norm(h))
        return self.head(self.norm_f(h[:, -1])).squeeze(-1)


# ============================================================
# DATOS
# ============================================================

def construir_ventanas(X, L):
    """Para cada fila t, la ventana de las L filas hasta t (inclusive).
    Si no hay historia suficiente, se repite la primera fila."""
    n = len(X)
    idx = np.arange(n)[:, None] + np.arange(-(L - 1), 1)[None, :]
    idx = np.clip(idx, 0, n - 1)
    return X[idx]                               # [n, L, n_feat]


def preparar_datos(df, features):
    df = df.sort_values([BEARING_COL, TIME_COL]).reset_index(drop=True)

    if VENTANA_SUAVIZADO > 1:
        df = df.copy()
        df[features] = (
            df.groupby(BEARING_COL)[features]
            .transform(lambda s: s.rolling(VENTANA_SUAVIZADO, min_periods=1).mean())
        )

    datos = {}
    for b, part in df.groupby(BEARING_COL, sort=True):
        X = part[features].to_numpy(dtype=np.float32)
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

def entrenar_y_predecir(datos, bearing_test, indice_fold, n_feat):
    train_ids = [b for b in datos if b != bearing_test]

    # Normalizacion con los 16 rodamientos de entrenamiento (filas, no ventanas)
    X_rows = np.concatenate([datos[b]["X"] for b in train_ids])
    media = X_rows.mean(axis=0)
    desv = X_rows.std(axis=0)
    desv[desv == 0] = 1.0

    W_tr = np.concatenate([datos[b]["W"] for b in train_ids])
    y_tr = np.concatenate([datos[b]["y"] for b in train_ids])
    W_tr = torch.from_numpy(((W_tr - media) / desv).astype(np.float32))
    y_tr = torch.from_numpy(y_tr)

    W_te = torch.from_numpy(((datos[bearing_test]["W"] - media) / desv).astype(np.float32))

    torch.manual_seed(SEMILLA + indice_fold)
    rng = np.random.default_rng(SEMILLA + indice_fold)

    model = MambaRUL(n_feat, D_MODEL, N_CAPAS, D_STATE, EXPAND, D_CONV)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS)

    n_total = len(W_tr)
    for epoca in range(1, EPOCAS + 1):
        t0 = time.time()
        model.train()
        inicio = (epoca - 1) % PASO_ENTRENAMIENTO
        perm = rng.permutation(np.arange(inicio, n_total, PASO_ENTRENAMIENTO))
        n = len(perm)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = torch.from_numpy(perm[i:i + BATCH])
            pred = model(W_tr[idx])
            loss = F.smooth_l1_loss(pred, y_tr[idx], beta=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        if epoca == 1 or epoca == EPOCAS:
            print(f"        época {epoca:02d}/{EPOCAS}  pérdida {total / n:.4f}  "
                  f"({time.time() - t0:.1f} s/época)")

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(W_te), 1024):
            preds.append(model(W_te[i:i + 1024]))
    return torch.cat(preds).numpy()


# ============================================================
# GRAFICO
# ============================================================

def graficar(pred_df, errores, destino, titulo):
    bearings = list(errores[BEARING_COL])
    n = len(bearings)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows),
        sharey=True, squeeze=False,
    )

    for ax, b in zip(axes.flat, bearings):
        part = pred_df[pred_df[BEARING_COL] == b]
        vida = (1.0 - part["rul_real"].to_numpy()) * 100.0
        mae = float(errores.loc[errores[BEARING_COL] == b, "MAE"].iloc[0])

        ax.plot(vida, part["rul_real"], color="black", linewidth=1.5, label="Real")
        ax.plot(vida, part["rul_pred"], color="tab:red", linewidth=0.9,
                alpha=0.9, label="Mamba")
        ax.set_title(f"{b}  |  MAE {mae:.3f}", fontsize=10)
        ax.set_xlim(0, 100)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Vida util (%)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("rul_frac", fontsize=9)

    axes[0, 0].legend(fontsize=8)
    fig.suptitle(titulo, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(destino, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():
    if not INPUT_FILE.exists():
        print(f"❌ No se encuentra el archivo:\n   {INPUT_FILE}")
        sys.exit(1)

    if N_HILOS is not None:
        torch.set_num_threads(N_HILOS)

    print(f"📂 Leyendo: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Filas: {len(df):,}  |  Columnas: {len(df.columns)}")

    requeridas = [BEARING_COL, TIME_COL, TARGET_COL] + FEATURES
    faltantes = [c for c in requeridas if c not in df.columns]
    if faltantes:
        print("❌ Faltan columnas:")
        for c in faltantes:
            print(f"   - {c}")
        sys.exit(1)
    if df[requeridas].isna().any().any():
        print("❌ Hay NaN en las columnas usadas.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tabla acumulada (para poder retomar): {feature: fila}
    filas_tabla = {}
    if TABLA_CSV.exists():
        previa = pd.read_csv(TABLA_CSV)
        filas_tabla = {r["feature"]: r.to_dict() for _, r in previa.iterrows()}

    print(f"⚙️  Suavizado: {VENTANA_SUAVIZADO}  |  Ventana: {LONG_SECUENCIA}  |  "
          f"Épocas: {EPOCAS}  |  Semilla: {SEMILLA}")
    print(f"   Features a correr: {len(FEATURES)}  |  Salida: {OUT_DIR}\n")

    t_total = time.time()
    for fi, feat in enumerate(FEATURES, start=1):
        png = OUT_DIR / f"{fi:02d}_{feat}.png"
        if feat in filas_tabla and png.exists():
            print(f"[{fi}/{len(FEATURES)}] {feat}: ya estaba hecha, se salta")
            continue

        print(f"[{fi}/{len(FEATURES)}] Feature: {feat}")
        t_feat = time.time()
        datos = preparar_datos(df, [feat])
        bearings = sorted(datos.keys())

        partes = []
        for k, b in enumerate(bearings):
            t0 = time.time()
            print(f"    ({k + 1}/{len(bearings)}) Prueba: {b}")
            pred = entrenar_y_predecir(datos, b, k, n_feat=1)
            partes.append(pd.DataFrame({
                BEARING_COL: b,
                TIME_COL: datos[b]["tiempo"],
                "rul_real": datos[b]["y"],
                "rul_pred": pred,
            }))
            mae = float(np.mean(np.abs(pred - datos[b]["y"])))
            print(f"        MAE {mae:.4f}  ({time.time() - t0:.0f} s)")

        pred_df = pd.concat(partes, ignore_index=True)
        pred_df["abs_err"] = (pred_df["rul_pred"] - pred_df["rul_real"]).abs()
        por_b = pred_df.groupby(BEARING_COL)["abs_err"].mean()
        errores = por_b.rename("MAE").reset_index()

        macro = float(por_b.mean())
        micro = float(pred_df["abs_err"].mean())
        graficar(pred_df, errores, png,
                 f"Mamba con 1 feature: {feat}  |  LOBO 17 rodamientos  |  "
                 f"MAE macro {macro:.3f}  micro {micro:.3f}")

        fila = {"feature": feat}
        fila.update({b: float(por_b[b]) for b in bearings})
        fila["MAE_macro"] = macro
        fila["MAE_micro"] = micro
        filas_tabla[feat] = fila

        # Se guarda la tabla despues de cada feature (en el orden de FEATURES)
        pd.DataFrame([filas_tabla[f] for f in FEATURES if f in filas_tabla]) \
            .to_csv(TABLA_CSV, index=False)
        print(f"    ✅ {png.name}  |  MAE macro {macro:.4f}  micro {micro:.4f}  "
              f"({(time.time() - t_feat) / 60:.1f} min)\n")

    print(f"⏱️  Tiempo total de esta ejecución: {(time.time() - t_total) / 60:.1f} min\n")

    if filas_tabla:
        tabla = pd.DataFrame([filas_tabla[f] for f in FEATURES if f in filas_tabla])
        print("Ranking por MAE macro (menor = mejor):")
        print(tabla[["feature", "MAE_macro", "MAE_micro"]]
              .sort_values("MAE_macro").round(4).to_string(index=False))
        print(f"\nTabla completa: {TABLA_CSV}")

    print("\nListo.")


if __name__ == "__main__":
    main()