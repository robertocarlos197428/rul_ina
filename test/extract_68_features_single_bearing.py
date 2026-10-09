"""
Extracción de 68 features — un solo rodamiento (fácil de cambiar).

Uso:
  1. Cambia BEARING_ID abajo (ej. "Bearing1_1", "Bearing2_3", etc.)
  2. Ejecuta el script.
  3. Revisa los prints de RMS_H / RMS_V de las últimas ráfagas para verificar.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import scipy.stats as stats

# ============================================================
# CONFIGURACIÓN — cambia solo esto
# ============================================================
BEARING_ID = "Bearing1_1"          # <-- cambia aquí (ej. "Bearing1_2", "Bearing2_1", ...)
RAW_BASE   = "rul_ina/data/raw_femto"
OUT_DIR    = "rul_ina/data/processed"

# ============================================================
# Filtro
# ============================================================
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return b, a

def highpass_filter(data, cutoff=10.0, fs=25600.0, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data)

# ============================================================
# Features temporales (12)
# ============================================================
def extract_temporal_features(x):
    N = len(x)
    rms = np.sqrt(np.sum(x**2) / N)
    peak = np.max(x)
    p2p = np.max(x) - np.min(x)
    variance = np.var(x, ddof=0)
    std = np.std(x, ddof=0)
    mav = np.sum(np.abs(x)) / N
    kurt = stats.kurtosis(x, fisher=True)
    skew = stats.skew(x)

    cf = peak / rms if rms != 0 else 0.0
    if_val = peak / mav if mav != 0 else 0.0
    sf = rms / mav if mav != 0 else 0.0

    denom_mf = (np.sum(np.sqrt(np.abs(x))) / N) ** 2
    mf = peak / denom_mf if denom_mf != 0 else 0.0

    return [rms, peak, p2p, variance, std, mav, kurt, skew, cf, if_val, sf, mf]

# ============================================================
# Features espectrales (5)
# ============================================================
def extract_spectral_features(x, fs=25600.0):
    N = len(x)
    window = np.hanning(N)
    x_w = x * window

    X_k = np.fft.rfft(x_w)
    freqs = np.fft.rfftfreq(N, 1 / fs)
    A_k = np.abs(X_k)
    sum_ak = np.sum(A_k)

    sc = np.sum(freqs * A_k) / sum_ak if sum_ak > 0 else 0.0
    K = len(A_k)
    spectral_rms = np.sqrt(np.sum(A_k**2) / K) if K > 0 else 0.0
    spectral_std = (
        np.sqrt(np.sum(A_k * (freqs - sc) ** 2) / sum_ak) if sum_ak > 0 else 0.0
    )

    if sum_ak > 0:
        p_k = A_k / sum_ak
        p_k_pos = p_k[p_k > 0]
        spectral_entropy = -np.sum(p_k_pos * np.log2(p_k_pos))
    else:
        spectral_entropy = 0.0

    epsilon = 1e-10
    A_k_safe = np.where(A_k == 0, epsilon, A_k)
    geom_mean = np.exp(np.mean(np.log(A_k_safe)))
    arith_mean = np.mean(A_k)
    spectral_flatness = geom_mean / arith_mean if arith_mean != 0 else 0.0

    return [sc, spectral_rms, spectral_std, spectral_entropy, spectral_flatness]

# ============================================================
# Features WPD (16)
# ============================================================
def extract_wpd_features(x):
    import pywt

    wavelet = "db4"
    level = 4
    mode = "periodization"

    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode=mode, maxlevel=level)
    # Orden fijo por path para reproducibilidad
    nodes = sorted([node.path for node in wp.get_level(level, "natural")])

    energies = [np.sum(wp[path].data ** 2) for path in nodes]
    total_energy = np.sum(energies)
    if total_energy > 0:
        e_rel = [e / total_energy for e in energies]
    else:
        e_rel = [0.0] * len(energies)

    return e_rel

# ============================================================
# Features cruzadas (2)
# ============================================================
def extract_cross_features(h, v):
    if np.std(h) != 0 and np.std(v) != 0:
        rho_hv = np.corrcoef(h, v)[0, 1]
        if np.isnan(rho_hv):
            rho_hv = 0.0
    else:
        rho_hv = 0.0

    sum_v2 = np.sum(v**2)
    er_hv = np.sum(h**2) / sum_v2 if sum_v2 != 0 else 0.0
    return [rho_hv, er_hv]

# ============================================================
# Proceso principal
# ============================================================
def process_single_bearing(bearing_id, raw_base, out_dir):
    bearing_path = os.path.join(raw_base, bearing_id)

    if not os.path.isdir(bearing_path):
        print(f"ERROR: No se encontró la carpeta: {bearing_path}")
        return

    burst_files = sorted(glob.glob(os.path.join(bearing_path, "*.csv")))
    if not burst_files:
        # fallback por si no tienen extensión .csv
        burst_files = sorted(
            [f for f in glob.glob(os.path.join(bearing_path, "*")) if os.path.isfile(f)]
        )

    if not burst_files:
        print(f"ERROR: No se encontraron archivos de ráfaga en {bearing_path}")
        return

    print(f"Procesando {bearing_id} — {len(burst_files)} ráfagas")
    print("-" * 60)

    all_rows = []
    base_seconds = None
    n_bad_length = 0

    for burst_idx, burst_file in enumerate(burst_files):
        try:
            df_burst = pd.read_csv(burst_file, header=None)

            if df_burst.shape[1] < 6:
                print(f"  [SKIP] {os.path.basename(burst_file)}: menos de 6 columnas")
                continue

            # Tiempo
            hour = float(df_burst.iloc[0, 0])
            minute = float(df_burst.iloc[0, 1])
            second = float(df_burst.iloc[0, 2])
            microsec = float(df_burst.iloc[0, 3])
            current_seconds = hour * 3600 + minute * 60 + second + microsec / 1e6

            if base_seconds is None:
                base_seconds = current_seconds

            time_seconds = current_seconds - base_seconds
            if time_seconds < 0:
                time_seconds += 24 * 3600

            # Señales
            h_raw = df_burst.iloc[:, 4].values.astype(float)
            v_raw = df_burst.iloc[:, 5].values.astype(float)

            # Validación de longitud
            if len(h_raw) != 2560 or len(v_raw) != 2560:
                n_bad_length += 1
                if n_bad_length <= 5:
                    print(
                        f"  [WARN] {os.path.basename(burst_file)}: "
                        f"longitud H={len(h_raw)}, V={len(v_raw)} (esperado 2560)"
                    )

            # Preprocesamiento
            h_dc = h_raw - np.mean(h_raw)
            v_dc = v_raw - np.mean(v_raw)
            h_filt = highpass_filter(h_dc, cutoff=10.0, fs=25600.0)
            v_filt = highpass_filter(v_dc, cutoff=10.0, fs=25600.0)

            # Features
            feat_temp_h = extract_temporal_features(h_filt)
            feat_temp_v = extract_temporal_features(v_filt)
            feat_spec_h = extract_spectral_features(h_filt)
            feat_spec_v = extract_spectral_features(v_filt)
            feat_wpd_h = extract_wpd_features(h_filt)
            feat_wpd_v = extract_wpd_features(v_filt)
            feat_cross = extract_cross_features(h_filt, v_filt)

            # ORDEN CRÍTICO: debe coincidir exactamente con cols_h + cols_v + cols_cross
            # cols_h = Temp_H + Spec_H + WPD_H
            # cols_v = Temp_V + Spec_V + WPD_V
            row_features = (
                feat_temp_h
                + feat_spec_h
                + feat_wpd_h
                + feat_temp_v
                + feat_spec_v
                + feat_wpd_v
                + feat_cross
            )

            if len(row_features) != 68:
                print(
                    f"  [ERROR] {os.path.basename(burst_file)}: "
                    f"se generaron {len(row_features)} features (esperado 68)"
                )
                continue

            # Debug: imprimir RMS de las últimas ráfagas
            if burst_idx >= len(burst_files) - 15:
                print(
                    f"  burst {burst_idx:4d} | t={time_seconds:8.1f}s | "
                    f"RMS_H={feat_temp_h[0]:.4f}  RMS_V={feat_temp_v[0]:.4f}"
                )

            # Armar fila
            row_dict = {
                "bearing_id": bearing_id,
                "burst_id": os.path.basename(burst_file),
                "burst_index": burst_idx,
                "time_seconds": time_seconds,
            }

            temp_names = [
                "RMS", "Peak", "P2P", "Variance", "Std", "MAV",
                "Kurtosis", "Skewness", "CrestFactor", "ImpulseFactor",
                "ShapeFactor", "MarginFactor",
            ]
            spec_names = [
                "SpectralCentroid", "SpectralRMS", "SpectralStd",
                "SpectralEntropy", "SpectralFlatness",
            ]
            wpd_names = [f"WPD_Erel_{i}" for i in range(16)]

            cols_h = (
                [f"{n}_H" for n in temp_names]
                + [f"{n}_H" for n in spec_names]
                + [f"{n}_H" for n in wpd_names]
            )
            cols_v = (
                [f"{n}_V" for n in temp_names]
                + [f"{n}_V" for n in spec_names]
                + [f"{n}_V" for n in wpd_names]
            )
            cols_cross = ["Pearson_HV", "EnergyRatio_HV"]

            for name, val in zip(cols_h + cols_v + cols_cross, row_features):
                row_dict[name] = val

            all_rows.append(row_dict)

        except Exception as e:
            print(f"  [ERROR] {burst_file}: {e}")

    if not all_rows:
        print("No se generaron registros.")
        return

    df_out = pd.DataFrame(all_rows)
    os.makedirs(out_dir, exist_ok=True)

    out_parquet = os.path.join(out_dir, f"features_68_{bearing_id}.parquet")
    out_csv = os.path.join(out_dir, f"features_68_{bearing_id}.csv")

    df_out.to_parquet(out_parquet, index=False)
    df_out.to_csv(out_csv, index=False)

    print("-" * 60)
    print(f"Listo. {len(df_out)} ráfagas procesadas.")
    if n_bad_length:
        print(f"  Aviso: {n_bad_length} ráfagas con longitud ≠ 2560")
    print(f"  → {out_parquet}")
    print(f"  → {out_csv}")

    # Resumen RMS para verificación rápida
    print("\n=== Resumen RMS ===")
    print(df_out[["RMS_H", "RMS_V"]].describe().round(4))
    print("\n=== Últimas 10 ráfagas (RMS) ===")
    print(
        df_out[["burst_index", "time_seconds", "RMS_H", "RMS_V"]]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    process_single_bearing(BEARING_ID, RAW_BASE, OUT_DIR)
