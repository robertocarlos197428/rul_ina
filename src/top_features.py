import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt

def load_top_30(csv_path):
    print(f"Cargando Top 30 features desde: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    df_top = pd.read_csv(csv_path)
    return df_top['Feature'].tolist()

def compute_mrmr(df_parquet, feature_cols, k_features=15):
    print(f"\nEjecutando mRMR para seleccionar un subconjunto de {k_features} features...")
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

def save_pretty_summary_png(df_summary, output_filename):
    print(f"Generando imagen PNG de la tabla resumen: {output_filename}...")
    fig, ax = plt.subplots(figsize=(14, len(df_summary) * 0.4 + 2))
    ax.axis('off')
    
    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)
    
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#d0d0d0')
        if key[0] == 0:
            cell.set_facecolor('#1f77b4')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f9f9f9' if key[0] % 2 == 0 else '#ffffff')
            
    plt.title("Resumen de Subconjuntos mRMR (K = 10 a 20 Features)", fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> ¡Tabla resumen guardada en: {output_filename}!")

def save_mrmr_results(parquet_path, top30_path, output_dir):
    df_parquet = pd.read_parquet(parquet_path)
    feature_cols = load_top_30(top30_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    summary_results = []
    for k in range(10, 21):
        print(f"\n--- Evaluando tamaño de subconjunto: K = {k} ---")
        selected_features = compute_mrmr(df_parquet, feature_cols, k_features=k)
        
        df_k = pd.DataFrame({'Ranking_mRMR': range(1, len(selected_features)+1), 'Feature': selected_features})
        df_k.to_csv(os.path.join(output_dir, f"mrmr_features_k_{k}.csv"), index=False)
        
        # Guardar string resumido para la tabla final
        summary_results.append({
            'K': k, 
            'Ultima Feature Agregada': selected_features[-1],
            'Features Seleccionadas': ", ".join(selected_features)
        })
        print(f"-> Subconjunto K={k} guardado con éxito.")
        
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(os.path.join(output_dir, "mrmr_summary_10_20.csv"), index=False)
    
    # Generar la tabla visual en PNG
    save_pretty_summary_png(df_summary, os.path.join(output_dir, "tabla_mrmr_summary.png"))
    
    print(f"\n¡Proceso mRMR finalizado! Resultados guardados en: {output_dir}")

if __name__ == "__main__":
    PARQUET_PATH = "rul_ina/data/processed/features_68.parquet"
    TOP30_PATH = "rul_ina/data/processed/top_30_features.csv"
    OUTPUT_DIR = "rul_ina/data/processed/top_features"
    
    print("==================================================")
    print("   ETAPA 3: SELECCIÓN mRMR (SUBGRUPOS 10 A 20)    ")
    print("==================================================")
    
    save_mrmr_results(PARQUET_PATH, TOP30_PATH, OUTPUT_DIR)