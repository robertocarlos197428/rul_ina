import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def compute_mahalanobis_hi(df_parquet, feature_list):
    bearing_ids = df_parquet['bearing_id'].unique()
    all_hi_results = []
    
    for b_id in bearing_ids:
        df_b = df_parquet[df_parquet['bearing_id'] == b_id].sort_values('burst_index').copy()
        
        # Extraer matriz de features seleccionadas
        X = df_b[feature_list].select_dtypes(include=[np.number]).values
        
        # Limpieza de NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) < 70:
            continue
            
        # 1. Referencia saludable: primeras 70 ráfagas
        X_sano = X[:70]
        
        # 2. Covarianza con Ledoit-Wolf y media del tramo sano
        mu_sano = np.mean(X_sano, axis=0)
        
        try:
            lw = LedoitWolf()
            lw.fit(X_sano)
            sigma_sano_inv = lw.precision_
        except Exception:
            # Fallback a cov convencional con regularización si falla Ledoit-Wolf
            cov = np.cov(X_sano, rowvar=False) + np.eye(X_sano.shape[1]) * 1e-6
            sigma_sano_inv = np.linalg.inv(cov)
            
        # 3. Cálculo de la Distancia de Mahalanobis para todas las ráfagas del rodamiento
        md_list = []
        for i in range(len(X)):
            diff = X[i] - mu_sano
            md = np.sqrt(np.dot(np.dot(diff, sigma_sano_inv), diff.T))
            md_list.append(max(0.0, md))
            
        md_array = np.array(md_list)
        
        # 4. Parámetro tau basado en el máximo del tramo sano
        md_sano = md_array[:70]
        tau = np.max(md_sano) if np.max(md_sano) > 0 else 1.0
        
        # 5. Transformación a HI: HI = exp(-MD / tau)
        hi_array = np.exp(-md_array / tau)
        
        df_b['Mahalanobis_Distance'] = md_array
        df_b['Health_Indicator'] = hi_array
        all_hi_results.append(df_b)
        
    return pd.concat(all_hi_results, ignore_index=True)

def evaluate_hi_quality(df_hi):
    # Evalúa la calidad de la curva de HI (Monotonicidad y Tendencia descendente hacia la falla)
    bearing_ids = df_hi['bearing_id'].unique()
    scores = []
    
    for b_id in bearing_ids:
        df_b = df_hi[df_hi['bearing_id'] == b_id].sort_values('burst_index')
        hi = df_b['Health_Indicator'].values
        t = np.arange(len(hi))
        
        if len(hi) < 10:
            continue
            
        # Tendencia de Spearman con el tiempo (esperamos correlación negativa alta, pues el HI cae)
        corr, _ = spearmanr(hi, t)
        tre_hi = abs(corr) if not np.isnan(corr) else 0.0
        
        # Monotonicidad del HI
        diffs = np.diff(hi)
        n_down = np.sum(diffs < 0)
        mon_hi = n_down / (len(hi) - 1) if len(hi) > 1 else 0.0
        
        scores.append((tre_hi + mon_hi) / 2.0)
        
    return np.mean(scores) if scores else 0.0

if __name__ == "__main__":
    PARQUET_PATH = "rul_ina/data/processed/features_68.parquet"
    MRMR_DIR = "rul_ina/data/processed/top_features"
    
    print("==================================================")
    print("   ETAPA 4: CÁLCULO HI Y SELECCIÓN AUTOMÁTICA     ")
    print("==================================================")
    
    df_parquet = pd.read_parquet(PARQUET_PATH)
    
    evaluation_summary = []
    best_score = -float('inf')
    best_k = None
    best_hi_df = None
    
    # Evaluar los 11 subconjuntos (K = 10 a 20)
    for k in range(10, 21):
        csv_file = os.path.join(MRMR_DIR, f"mrmr_features_k_{k}.csv")
        if not os.path.exists(csv_file):
            print(f"No se encontró el archivo para K={k}. Saltando...")
            continue
            
        df_features = pd.read_csv(csv_file)
        feature_list = df_features['Feature'].tolist()
        
        print(f"Calculando HI para K = {k} ({len(feature_list)} features)...")
        df_hi = compute_mahalanobis_hi(df_parquet, feature_list)
        
        # Puntuar la calidad del HI resultante
        quality_score = evaluate_hi_quality(df_hi)
        print(f"-> Calidad del HI (Score): {quality_score:.4f}")
        
        evaluation_summary.append({'K': k, 'Quality_Score': quality_score})
        
        if quality_score > best_score:
            best_score = quality_score
            best_k = k
            best_hi_df = df_hi
            
    # Resumen de optimización
    df_eval = pd.DataFrame(evaluation_summary)
    df_eval.to_csv(os.path.join(MRMR_DIR, "hi_evaluation_summary.csv"), index=False)
    
    print("\n==================================================")
    print(f" ¡VEREDICTO AUTOMÁTICO ÓPTIMO: K = {best_k} (Score: {best_score:.4f})! ")
    print("==================================================")
    
    # Guardar resultados del mejor modelo
    best_hi_df.to_csv(os.path.join(MRMR_DIR, f"best_health_indicators_k_{best_k}.csv"), index=False)
    
    # 1. Graficar curva de estancamiento (Calidad vs K)
    plt.figure(figsize=(9, 5))
    plt.plot(df_eval['K'], df_eval['Quality_Score'], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.title("Estancamiento y Calidad del HI vs. Número de Features (K)", fontsize=12, fontweight='bold')
    plt.xlabel("Número de Features (K)", fontsize=10)
    plt.ylabel("Puntaje de Calidad del HI", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Óptimo (K={best_k})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MRMR_DIR, "curva_estancamiento_calidad_hi.png"), dpi=300)
    plt.close()
    
    # 2. Graficar las curvas de HI de los 17 rodamientos para el mejor K
    plt.figure(figsize=(12, 6))
    for b_id in best_hi_df['bearing_id'].unique():
        df_b = best_hi_df[best_hi_df['bearing_id'] == b_id].sort_values('burst_index')
        # Usar índice de ráfaga o tiempo normalizado para el eje X
        x_axis = np.linspace(0, 100, len(df_b))
        plt.plot(x_axis, df_b['Health_Indicator'], alpha=0.7, linewidth=1.5)
        
    plt.title(f"Indicadores de Salud (HI) para los 17 Rodamientos (Modelo Óptimo K = {best_k})", fontsize=13, fontweight='bold')
    plt.xlabel("Progreso de Vida del Rodamiento (%) [Sano -> Falla]", fontsize=10)
    plt.ylabel("Health Indicator (HI)", fontsize=10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(MRMR_DIR, f"curvas_hi_optimas_k_{best_k}.png"), dpi=300)
    plt.close()
    
    print("¡Archivos generados con éxito en la carpeta top_features!")
    print(" - hi_evaluation_summary.csv")
    print(" - curva_estancamiento_calidad_hi.png")
    print(f" - curvas_hi_optimas_k_{best_k}.png")
    print(f" - best_health_indicators_k_{best_k}.csv")