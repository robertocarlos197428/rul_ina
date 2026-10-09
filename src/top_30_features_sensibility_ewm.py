import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def verify_and_load_parquet(parquet_path):
    print("Verificando archivo Parquet...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"No se encontró el archivo en: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    
    # Metadatos reales a excluir para quedarnos solo con las 68 features
    metadata_cols = ['bearing_id', 'burst_id', 'burst_index', 'time_seconds']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    print(f"-> Total de columnas en Parquet: {len(df.columns)}")
    print(f"-> Total de features detectadas: {len(feature_cols)}")
    
    if len(feature_cols) != 68:
        print(f"¡Advertencia! Se esperaban 68 features y se encontraron {len(feature_cols)}.")
    else:
        print("¡Verificación exitosa! Las 68 features exactas están listas.")
        
    return df, feature_cols

def compute_metrics(df, feature_cols):
    print("\nCalculando métricas de sensibilidad (Mon, Tre, Pro, Rob)...")
    bearing_ids = df['bearing_id'].unique()
    
    results = []
    
    for feat in feature_cols:
        mon_list, tre_list, pro_list, rob_list = [], [], [], []
        
        for b_id in bearing_ids:
            df_b = df[df['bearing_id'] == b_id].sort_values('burst_index')
            
            # Forzar conversión a float para evitar errores con texto/strings
            x = pd.to_numeric(df_b[feat], errors='coerce').values
            t = np.arange(len(x))
            
            # Limpiar NaNs o infinitos reemplazándolos por la media o ceros
            mean_val = np.nanmean(x) if not np.all(np.isnan(x)) else 0.0
            x = np.nan_to_num(x, nan=mean_val, posinf=0, neginf=0)
            
            if len(x) < 10 or np.ptp(x) == 0:
                continue
                
            # 1. Monotonicidad (Mon)
            diffs = np.diff(x)
            n_up = np.sum(diffs > 0)
            n_down = np.sum(diffs < 0)
            mon = abs(n_up - n_down) / (len(x) - 1) if len(x) > 1 else 0
            mon_list.append(mon)
            
            # 2. Tendencia (Tre - Correlación de Spearman con el tiempo)
            corr, _ = spearmanr(x, t)
            tre = abs(corr) if not np.isnan(corr) else 0
            tre_list.append(tre)
            
            # 3. Pronosticabilidad (Pro) - Sano: primeras 70 ráfagas
            f_sano = x[:70] if len(x) >= 70 else x[:len(x)//2]
            f_falla = x[-70:] if len(x) >= 70 else x[len(x)//2:]
            
            std_falla = np.std(f_falla)
            mean_diff = abs(np.mean(f_sano) - np.mean(f_falla))
            
            if mean_diff == 0:
                pro = 0.0
            else:
                pro = np.exp(-std_falla / mean_diff)
            pro_list.append(pro)
            
            # 4. Robustez (Rob) - Suavizado por media móvil
            window = max(5, len(x) // 20)
            f_tilde = pd.Series(x).rolling(window=window, center=True, min_periods=1).mean().values
            safe_x = np.where(x == 0, 1e-10, x)
            rob = np.mean(np.exp(-abs((x - f_tilde) / safe_x)))
            rob_list.append(rob)
            
        results.append({
            'Feature': feat,
            'Mon': np.mean(mon_list) if mon_list else 0,
            'Tre': np.mean(tre_list) if tre_list else 0,
            'Pro': np.mean(pro_list) if pro_list else 0,
            'Rob': np.mean(rob_list) if rob_list else 0
        })
        
    return pd.DataFrame(results)

def apply_ewm(df_metrics):
    print("\nAplicando Entropy Weight Method (EWM)...")
    metrics = ['Mon', 'Tre', 'Pro', 'Rob']
    
    z_matrix = pd.DataFrame()
    for m in metrics:
        col_sum = df_metrics[m].sum()
        if col_sum == 0:
            z_matrix[m] = df_metrics[m] * 0
        else:
            z_matrix[m] = df_metrics[m] / col_sum
            
    n = len(df_metrics)
    k = 1.0 / np.log(n) if n > 1 else 1.0
    
    entropy = []
    for m in metrics:
        p = z_matrix[m].values
        p = np.where(p == 0, 1e-12, p)
        e = -k * np.sum(p * np.log(p))
        entropy.append(e)
        
    entropy = np.array(entropy)
    divergence = 1.0 - entropy
    if divergence.sum() == 0:
        weights = np.ones(4) / 4.0
    else:
        weights = divergence / divergence.sum()
        
    w_dict = {f"w_{m}": weights[i] for i, m in enumerate(metrics)}
    print(f"Pesos EWM obtenidos -> {w_dict}")
    
    score = 0
    for i, m in enumerate(metrics):
        score += weights[i] * df_metrics[m]
        
    df_metrics['Score'] = score
    df_metrics = df_metrics.sort_values(by='Score', ascending=False).reset_index(drop=True)
    df_metrics['Ranking'] = df_metrics.index + 1
    
    return df_metrics, w_dict

def save_pretty_table_png(df_table, title, output_filename):
    print(f"Generando imagen PNG de la tabla: {output_filename}...")
    fig, ax = plt.subplots(figsize=(12, len(df_table) * 0.35 + 1.5))
    ax.axis('off')
    
    df_show = df_table.copy()
    for col in df_show.select_dtypes(include=[np.number]).columns:
        if col != 'Ranking':
            df_show[col] = df_show[col].round(4)
            
    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        cellLoc='center',
        loc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.3)
    
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#d0d0d0')
        if key[0] == 0:
            cell.set_facecolor('#1f77b4')
            cell.set_text_props(color='white', weight='bold')
        else:
            cell.set_facecolor('#f9f9f9' if key[0] % 2 == 0 else '#ffffff')
            
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> ¡Tabla guardada en: {output_filename}!")

if __name__ == "__main__":
    PARQUET_PATH = "rul_ina/data/processed/features_68.parquet"
    
    print("==================================================")
    print("   ETAPA 2: SENSIBILIDAD, EWM Y RANKING TOP 30    ")
    print("==================================================")
    
    df_parquet, feature_cols = verify_and_load_parquet(PARQUET_PATH)
    df_metrics = compute_metrics(df_parquet, feature_cols)
    df_ranked, weights = apply_ewm(df_metrics)
    
    os.makedirs("rul_ina/data/processed", exist_ok=True)
    df_ranked.to_csv("rul_ina/data/processed/ranking_68_features.csv", index=False)
    
    save_pretty_table_png(
        df_ranked, 
        "Ranking General de las 68 Features (Sensibilidad + EWM)", 
        "rul_ina/data/processed/tabla_ranking_68_features.png"
    )
    
    df_top30 = df_ranked.head(30)
    df_top30.to_csv("rul_ina/data/processed/top_30_features.csv", index=False)
    save_pretty_table_png(
        df_top30, 
        "Subconjunto Top 30 Features Seleccionadas por EWM", 
        "rul_ina/data/processed/tabla_top_30_features.png"
    )
    
    print("\n¡Proceso completado con éxito!")