import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
import scipy.stats as stats

pywt = None 

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=10.0, fs=25600.0, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

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
    
    cf = peak / rms if rms != 0 else 0
    if_val = peak / mav if mav != 0 else 0
    sf = rms / mav if mav != 0 else 0
    
    denom_mf = (np.sum(np.sqrt(np.abs(x))) / N) ** 2
    mf = peak / denom_mf if denom_mf != 0 else 0
    
    return [rms, peak, p2p, variance, std, mav, kurt, skew, cf, if_val, sf, mf]

def extract_spectral_features(x, fs=25600.0):
    N = len(x)
    window = np.hanning(N)
    x_w = x * window
    
    X_k = np.fft.rfft(x_w)
    freqs = np.fft.rfftfreq(N, 1/fs)
    A_k = np.abs(X_k)
    sum_ak = np.sum(A_k)
    
    sc = np.sum(freqs * A_k) / sum_ak if sum_ak > 0 else 0.0
    K = len(A_k)
    spectral_rms = np.sqrt(np.sum(A_k**2) / K) if K > 0 else 0.0
    spectral_std = np.sqrt(np.sum(A_k * (freqs - sc)**2) / sum_ak) if sum_ak > 0 else 0.0
    
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

def extract_wpd_features(x):
    global pywt
    if pywt is None:
        import pywt
    
    wavelet = 'db4'
    level = 4
    mode = 'periodization'
    
    wp = pywt.WaveletPacket(data=x, wavelet=wavelet, mode=mode, maxlevel=level)
    nodes = [node.path for node in wp.get_level(level, 'natural')]
    
    energies = [np.sum(wp[path].data**2) for path in nodes]
    total_energy = np.sum(energies)
    e_rel = [e / total_energy for e in energies] if total_energy > 0 else [0.0] * len(energies)
    
    return e_rel

def extract_cross_features(h, v):
    rho_hv = np.corrcoef(h, v)[0, 1] if np.std(h) != 0 and np.std(v) != 0 else 0.0
    sum_v2 = np.sum(v**2)
    er_hv = np.sum(h**2) / sum_v2 if sum_v2 != 0 else 0.0
    return [rho_hv, er_hv]

def process_dataset(raw_dir, output_parquet, output_csv):
    all_rows = []
    bearing_folders = sorted(glob.glob(os.path.join(raw_dir, "*")))
    
    if not bearing_folders:
        print(f"No se encontraron rodamientos en la ruta: {raw_dir}")
        return

    for bearing_path in bearing_folders:
        bearing_id = os.path.basename(bearing_path)
        print(f"Procesando rodamiento: {bearing_id}...")
        
        burst_files = sorted(glob.glob(os.path.join(bearing_path, "*.csv")))
        if not burst_files:
            burst_files = sorted([f for f in glob.glob(os.path.join(bearing_path, "*")) if os.path.isfile(f)])
            
        base_seconds = None
        
        for burst_idx, burst_file in enumerate(burst_files):
            try:
                df_burst = pd.read_csv(burst_file, header=None)
                if df_burst.shape[1] >= 6:
                    hour = df_burst.iloc[0, 0]
                    minute = df_burst.iloc[0, 1]
                    second = df_burst.iloc[0, 2]
                    microsec = df_burst.iloc[0, 3]
                    
                    current_seconds = hour * 3600 + minute * 60 + second + microsec / 1e6
                    
                    if base_seconds is None:
                        base_seconds = current_seconds
                    
                    time_seconds = current_seconds - base_seconds
                    if time_seconds < 0:
                        time_seconds += 24 * 3600
                        
                    h_raw = df_burst.iloc[:, 4].values.astype(float)
                    v_raw = df_burst.iloc[:, 5].values.astype(float)
                else:
                    continue
                
                # 1. Eliminación de DC y 2. Filtro pasa-altos
                h_dc = h_raw - np.mean(h_raw)
                v_dc = v_raw - np.mean(v_raw)
                h_filt = highpass_filter(h_dc, cutoff=10.0, fs=25600.0)
                v_filt = highpass_filter(v_dc, cutoff=10.0, fs=25600.0)
                
                # 3. Features
                row_features = extract_temporal_features(h_filt) + extract_temporal_features(v_filt) + \
                               extract_spectral_features(h_filt) + extract_spectral_features(v_filt) + \
                               extract_wpd_features(h_filt) + extract_wpd_features(v_filt) + \
                               extract_cross_features(h_filt, v_filt)
                               
                row_dict = {
                    'bearing_id': bearing_id,
                    'burst_id': os.path.basename(burst_file),
                    'burst_index': burst_idx,
                    'time_seconds': time_seconds
                }
                
                temp_names = ['RMS', 'Peak', 'P2P', 'Variance', 'Std', 'MAV', 'Kurtosis', 'Skewness', 'CrestFactor', 'ImpulseFactor', 'ShapeFactor', 'MarginFactor']
                spec_names = ['SpectralCentroid', 'SpectralRMS', 'SpectralStd', 'SpectralEntropy', 'SpectralFlatness']
                wpd_names = [f'WPD_Erel_{i}' for i in range(16)]
                
                # Corrección: Se ordenan intercalando por dominio y canal tal cual se calculan en row_features
                cols_h_temp = [f"{n}_H" for n in temp_names]
                cols_v_temp = [f"{n}_V" for n in temp_names]
                cols_h_spec = [f"{n}_H" for n in spec_names]
                cols_v_spec = [f"{n}_V" for n in spec_names]
                cols_h_wpd  = [f"{n}_H" for n in wpd_names]
                cols_v_wpd  = [f"{n}_V" for n in wpd_names]
                cols_cross  = ['Pearson_HV', 'EnergyRatio_HV']
                
                all_column_names = cols_h_temp + cols_v_temp + cols_h_spec + cols_v_spec + cols_h_wpd + cols_v_wpd + cols_cross
                
                for name, val in zip(all_column_names, row_features):
                    row_dict[name] = val
                    
                all_rows.append(row_dict)
                
            except Exception as e:
                print(f"Error procesando ráfaga {burst_file}: {e}")
                
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
        df_out.to_parquet(output_parquet, index=False)
        df_out.to_csv(output_csv, index=False)
        print(f"\n¡Proceso completado con éxito!")
        print(f"- Archivos guardados con la columna `time_seconds`.")
    else:
        print("No se generaron registros.")

if __name__ == "__main__":
    RAW_DIR = "rul_ina/data/raw_femto"
    OUTPUT_PARQUET = "rul_ina/data/processed/features_68.parquet"
    OUTPUT_CSV = "rul_ina/data/processed/features_68.csv"
    process_dataset(RAW_DIR, OUTPUT_PARQUET, OUTPUT_CSV)