import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt

def main():
    # 1. Definir rutas
    base_dir = r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed"
    output_dir = r"C:\Users\emaker\Downloads\RUL\rul_ina\hi_rul"
    os.makedirs(output_dir, exist_ok=True)
    
    parquet_path = os.path.join(base_dir, "features_68.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(base_dir, "features_68.parket")
        
    print(f"Cargando dataset desde: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # 2. Las 6 features óptimas para K = 6
    selected_features = [
        "SpectralStd_H",
        "WPD_Erel_5_H",
        "SpectralCentroid_V",
        "WPD_Erel_8_H",
        "WPD_Erel_6_H",
        "WPD_Erel_7_H"
    ]
    
    bearing_col = 'bearing_id' if 'bearing_id' in df.columns else df.columns[0]
    bearing_ids = df[bearing_col].unique()
    
    all_processed_data = []
    
    print("Calculando Indicador de Salud (HI) y RUL por rodamiento...")
    
    for b_id in bearing_ids:
        # Filtrar y ordenar cronológicamente por ráfaga
        df_b = df[df[bearing_col] == b_id].sort_values('burst_index').copy()
        X = df_b[selected_features].select_dtypes(include=[np.number]).values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if len(X) < 70:
            continue
            
        # Referencia sana (primeras 70 ráfagas)
        X_sano = X[:70]
        mu_sano = np.mean(X_sano, axis=0)
        
        try:
            lw = LedoitWolf()
            lw.fit(X_sano)
            sigma_sano_inv = lw.precision_
        except Exception:
            cov = np.cov(X_sano, rowvar=False) + np.eye(X_sano.shape[1]) * 1e-6
            sigma_sano_inv = np.linalg.inv(cov)
            
        # Distancia de Mahalanobis
        md_list = []
        for i in range(len(X)):
            diff = X[i] - mu_sano
            md = np.sqrt(np.dot(np.dot(diff, sigma_sano_inv), diff.T))
            md_list.append(max(0.0, md))
            
        md_array = np.array(md_list)
        tau = np.max(md_array[:70]) if np.max(md_array[:70]) > 0 else 1.0
        hi_array = np.exp(-md_array / tau)
        
        # Asignar HI al DataFrame del rodamiento
        df_b['Health_Indicator'] = hi_array
        
        # Calcular RUL (en número de ráfagas restantes hasta el final de la vida del rodamiento)
        total_bursts = len(df_b)
        burst_indices = np.arange(total_bursts)
        rul_array = total_bursts - 1 - burst_indices
        df_b['RUL'] = rul_array
        
        all_processed_data.append(df_b)
        
    # Consolidar tabla global de la flota
    df_global = pd.concat(all_processed_data, ignore_index=True)
    
    # 3. Guardar en formato Parquet y CSV en la nueva ruta
    parquet_out = os.path.join(output_dir, "dataset_hi_rul_features.parquet")
    csv_out = os.path.join(output_dir, "dataset_hi_rul_features.csv")
    
    print(f"Guardando tabla consolidada en Parquet: {parquet_out}")
    df_global.to_parquet(parquet_out, index=False)
    
    print(f"Guardando tabla consolidada en CSV: {csv_out}")
    df_global.to_csv(csv_out, index=False)
    
    # 4. Generar la grilla de 18 espacios (6 filas x 3 columnas) para los rodamientos
    print("Generando grilla de gráficos de RUL (6x3)...")
    fig, axes = plt.subplots(6, 3, figsize=(15, 18), sharex=False, sharey=False)
    axes = axes.flatten()
    
    for idx, b_id in enumerate(bearing_ids):
        if idx >= len(axes):
            break
        ax = axes[idx]
        df_b = df_global[df_global[bearing_col] == b_id]
        
        ax.plot(df_b['burst_index'], df_b['RUL'], color='#1f77b4', linewidth=2)
        ax.set_title(f"Rodamiento: {b_id}", fontsize=10, fontweight='bold')
        ax.set_xlabel("Índice de Ráfaga", fontsize=8)
        ax.set_ylabel("RUL (Ráfagas)", fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        
    # Apagar los subplots vacíos si sobran (17 rodamientos < 18 espacios)
    for j in range(len(bearing_ids), len(axes)):
        fig.delaxes(axes[j])
        
    plt.suptitle("Evolución del RUL Real para la Flota de Rodamientos", fontsize=14, fontweight='bold', y=0.92)
    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    
    img_out = os.path.join(output_dir, "grilla_rul_17_rodamientos.png")
    plt.savefig(img_out, dpi=300)
    print(f"¡Grilla de RUL guardada con éxito en: {img_out}!")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()