import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

def compute_mahalanobis_hi(df_parquet, feature_list):
    bearing_ids = df_parquet['bearing_id'].unique()
    all_hi_results = []
    
    for b_id in bearing_ids:
        df_b = df_parquet[df_parquet['bearing_id'] == b_id].sort_values('burst_index').copy()
        X = df_b[feature_list].select_dtypes(include=[np.number]).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) < 70:
            continue
            
        X_sano = X[:70]
        mu_sano = np.mean(X_sano, axis=0)
        
        try:
            lw = LedoitWolf()
            lw.fit(X_sano)
            sigma_sano_inv = lw.precision_
        except Exception:
            cov = np.cov(X_sano, rowvar=False) + np.eye(X_sano.shape[1]) * 1e-6
            sigma_sano_inv = np.linalg.inv(cov)
            
        md_list = []
        for i in range(len(X)):
            diff = X[i] - mu_sano
            md = np.sqrt(np.dot(np.dot(diff, sigma_sano_inv), diff.T))
            md_list.append(max(0.0, md))
            
        md_array = np.array(md_list)
        md_sano = md_array[:70]
        tau = np.max(md_sano) if np.max(md_sano) > 0 else 1.0
        hi_array = np.exp(-md_array / tau)
        
        df_b['Health_Indicator'] = hi_array
        all_hi_results.append(df_b)
        
    return pd.concat(all_hi_results, ignore_index=True)

if __name__ == "__main__":
    PARQUET_PATH = r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed\features_68.parquet"
    OUTPUT_DIR = r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed\small_features"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Cargando datos desde el archivo parquet...")
    df_parquet = pd.read_parquet(PARQUET_PATH)
    
    # Las 6 características óptimas para K = 6 obtenidas por mRMR
    selected_features = [
        "SpectralStd_H",
        "WPD_Erel_5_H",
        "SpectralCentroid_V",
        "WPD_Erel_8_H",
        "WPD_Erel_6_H",
        "WPD_Erel_7_H"
    ]
    
    print("Calculando Indicador de Salud (HI) para K = 6...")
    df_hi = compute_mahalanobis_hi(df_parquet, selected_features)
    
    common_x = np.linspace(0, 100, 100)
    interpolated_curves = []
    
    for b_id in df_hi['bearing_id'].unique():
        df_b = df_hi[df_hi['bearing_id'] == b_id].sort_values('burst_index')
        x_b = np.linspace(0, 100, len(df_b))
        hi_b = df_b['Health_Indicator'].values
        interp_hi = np.interp(common_x, x_b, hi_b)
        interpolated_curves.append(interp_hi)
        
    mean_hi_curve = np.mean(interpolated_curves, axis=0)
    std_hi_curve = np.std(interpolated_curves, axis=0)
    
    # Generar el gráfico definitivo individual
    plt.figure(figsize=(10, 6))
    plt.plot(common_x, mean_hi_curve, color='#2ca02c', linewidth=2.5, label='Promedio Flota (K = 6)')
    plt.fill_between(common_x, mean_hi_curve - std_hi_curve, 
                              mean_hi_curve + std_hi_curve, 
                              color='#2ca02c', alpha=0.15, label='Dispersión (±1 Std)')
                              
    plt.title("Indicador de Salud Definitivo - Subconjunto Óptimo (K = 6)", fontsize=13, fontweight='bold')
    plt.xlabel("Progreso de Vida del Rodamiento (%) [Sano -> Falla]", fontsize=11)
    plt.ylabel("Health Indicator (HI)", fontsize=11)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    output_img = os.path.join(OUTPUT_DIR, "Health_Indicator_K6_Definitivo.png")
    plt.savefig(output_img, dpi=300)
    print(f"\n¡Gráfico definitivo de K=6 guardado con éxito en: {output_img}!")
    
    plt.show()
    plt.close()