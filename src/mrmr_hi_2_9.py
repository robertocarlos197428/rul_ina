import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

def load_top_30(csv_path):
    print(f"Cargando Top 30 features desde: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    df_top = pd.read_csv(csv_path)
    return df_top['Feature'].tolist()

def compute_mrmr(df_parquet, feature_cols, k_features=5):
    print(f"Calculando mRMR para K = {k_features}...")
    bearing_ids = df_parquet['bearing_id'].unique()
    
    target_dict = {}
    for b_id in bearing_ids:
        df_b = df_parquet[df_parquet['bearing_id'] == b_id].sort_values('burst_index')
        n_bursts = len(df_b)
        degradation_proxy = np.linspace(0, 1, n_bursts)
        for idx, row in df_b.iterrows():
            target_dict[idx] = degradation_proxy[df_b.index.get_loc(idx)]
            
    df_work = df_parquet.copy()
    df_work['degradation_target'] = pd.Series(target_dict)
    
    relevance = {}
    for feat in feature_cols:
        mi_list = []
        for b_id in bearing_ids:
            df_b = df_work[df_work['bearing_id'] == b_id]
            x = pd.to_numeric(df_b[feat], errors='coerce').values
            y = df_b['degradation_target'].values
            
            mask = ~np.isnan(x)
            if np.sum(mask) < 10:
                continue
            x_clean = x[mask].reshape(-1, 1)
            y_clean = y[mask]
            
            try:
                mi = mutual_info_regression(x_clean, y_clean, n_neighbors=3, random_state=42)[0]
                mi_list.append(mi)
            except Exception:
                pass
        relevance[feat] = np.mean(mi_list) if mi_list else 0.0
        
    selected = []
    remaining = list(feature_cols)
    
    best_first = max(remaining, key=lambda f: relevance[f])
    selected.append(best_first)
    remaining.remove(best_first)
    
    while len(selected) < k_features and remaining:
        best_score = -float('inf')
        best_candidate = None
        
        for cand in remaining:
            rel = relevance[cand]
            red_list = []
            for sel in selected:
                mi_pair_list = []
                for b_id in bearing_ids:
                    df_b = df_work[df_work['bearing_id'] == b_id]
                    xc = pd.to_numeric(df_b[cand], errors='coerce').values
                    xs = pd.to_numeric(df_b[sel], errors='coerce').values
                    mask = ~np.isnan(xc) & ~np.isnan(xs)
                    if np.sum(mask) < 10:
                        continue
                    try:
                        mi_s = mutual_info_regression(xc[mask].reshape(-1, 1), xs[mask], n_neighbors=3, random_state=42)[0]
                        mi_pair_list.append(mi_s)
                    except Exception:
                        pass
                red_list.append(np.mean(mi_pair_list) if mi_pair_list else 0.0)
                
            redundancy = np.mean(red_list) if red_list else 0.0
            score = rel - redundancy
            if score > best_score:
                best_score = score
                best_candidate = cand
                
        selected.append(best_candidate)
        remaining.remove(best_candidate)
        
    return selected

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
    PARQUET_PATH = "rul_ina/data/processed/features_68.parquet"
    TOP30_PATH = "rul_ina/data/processed/top_30_features.csv"
    OUTPUT_DIR = "rul_ina/data/processed/small_features"
    
    print("==================================================")
    print("   ANÁLISIS DE SUBGRUPOS PEQUEÑOS (K = 2 a 9)     ")
    print("==================================================")
    
    df_parquet = pd.read_parquet(PARQUET_PATH)
    feature_cols = load_top_30(TOP30_PATH)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    k_values = range(2, 10) # De 2 a 9
    
    # Configurar grilla (4 filas x 2 columnas para 8 gráficos: K=2 al 9)
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True, sharey=True)
    axes = axes.flatten()
    common_x = np.linspace(0, 100, 100)
    
    for idx, k in enumerate(k_values):
        ax = axes[idx]
        print(f"\n--- Procesando subconjunto K = {k} ---")
        selected_features = compute_mrmr(df_parquet, feature_cols, k_features=k)
        
        # Guardar CSV del subgrupo
        df_k = pd.DataFrame({'Ranking_mRMR': range(1, len(selected_features)+1), 'Feature': selected_features})
        df_k.to_csv(os.path.join(OUTPUT_DIR, f"mrmr_features_k_{k}.csv"), index=False)
        
        # Calcular HI
        df_hi = compute_mahalanobis_hi(df_parquet, selected_features)
        
        # Promediar curvas para la visualización limpia
        interpolated_curves = []
        for b_id in df_hi['bearing_id'].unique():
            df_b = df_hi[df_hi['bearing_id'] == b_id].sort_values('burst_index')
            x_b = np.linspace(0, 100, len(df_b))
            hi_b = df_b['Health_Indicator'].values
            interp_hi = np.interp(common_x, x_b, hi_b)
            interpolated_curves.append(interp_hi)
            
        mean_hi_curve = np.mean(interpolated_curves, axis=0)
        
        # Graficar en el panel correspondiente
        ax.plot(common_x, mean_hi_curve, color='#2ca02c', linewidth=2.5, label=f'Promedio K={k}')
        ax.fill_between(common_x, mean_hi_curve - np.std(interpolated_curves, axis=0), 
                                  mean_hi_curve + np.std(interpolated_curves, axis=0), 
                                  color='#2ca02c', alpha=0.15)
        
        ax.set_title(f"Subconjunto K = {k}", fontsize=11, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', fontsize=8)
        
        if idx >= 6:  # Fila inferior
            ax.set_xlabel("Progreso de Vida del Rodamiento (%) [Sano -> Falla]", fontsize=9)
        if idx % 2 == 0:  # Columna izquierda
            ax.set_ylabel("Health Indicator (HI)", fontsize=9)
            
    plt.suptitle("Tendencias del Indicador de Salud (HI) para Subgrupos Reducidos (K = 2 a 9)", fontsize=14, fontweight='bold', y=0.92)
    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    
    output_img = os.path.join(OUTPUT_DIR, "comparacion_grid_small_k_2_9.png")
    plt.savefig(output_img, dpi=300)
    plt.close()
    
    print(f"\n¡Proceso finalizado! Grilla guardada en: {output_img}")