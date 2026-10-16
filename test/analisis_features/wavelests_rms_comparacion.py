import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo visual
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Cargar el archivo real desde la ruta indicada
input_path = r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed\features_68.parquet"
output_dir = r"rul_ina/analisis_features/wavelets"
os.makedirs(output_dir, exist_ok=True)

print(f"Cargando datos desde: {input_path}")
df = pd.read_parquet(input_path)

# Obtener la lista de todos los rodamientos únicos en el dataset
bearing_ids = df['bearing_id'].unique()
print(f"Se encontraron {len(bearing_ids)} rodamientos para procesar: {bearing_ids}")

# Iterar sobre cada rodamiento
for b_id in bearing_ids:
    print(f"\nProcesando {b_id}...")
    df_bearing = df[df['bearing_id'] == b_id].sort_values('time_seconds')
    
    # Crear la etiqueta de etapa temporal para los boxplots (Columna 3)
    df_bearing['stage'] = pd.qcut(df_bearing['time_seconds'], q=3, labels=['Inicio (Sano)', 'Intermedio (Degradación)', 'Crítico (Fallo)'])
    
    # Procesar ambos canales: Horizontal (_H) y Vertical (_V) ordenados por rodamiento
    canales = [
        ('H', [f'WPD_Erel_{i}_H' for i in range(16)], 'RMS_H'),
        ('V', [f'WPD_Erel_{i}_V' for i in range(16)], 'RMS_V')
    ]
    
    for canal_nombre, features, rms_col in canales:
        print(f"  -> Generando gráfico 16x3 para el canal {canal_nombre}...")
        
        fig, axes = plt.subplots(16, 3, figsize=(18, 40), sharex=False)
        
        for i, col in enumerate(features):
            # --- Columna 1: RMS Global de Referencia ---
            ax1 = axes[i, 0]
            ax1.plot(df_bearing['time_seconds'], df_bearing[rms_col], color='gray', linewidth=1.0, alpha=0.8)
            ax1.set_title(f'Ref: {rms_col} (Fila {i})', fontsize=9, fontweight='bold')
            ax1.set_ylabel('RMS Total', fontsize=8)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # --- Columna 2: Evolución Temporal Wavelet (Escala 0 a 1) ---
            ax2 = axes[i, 1]
            ax2.plot(df_bearing['time_seconds'], df_bearing[col], color='darkblue' if canal_nombre=='H' else 'darkred', linewidth=1.2)
            ax2.set_title(f'Evolución: {col}', fontsize=9, fontweight='bold')
            ax2.set_ylabel('WPD_Erel', fontsize=8)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.set_ylim(0, 1.05)  # Escala fija de 0 a 1

            # --- Columna 3: Boxplots por Etapas (Escala 0 a 1) ---
            ax3 = axes[i, 2]
            sns.boxplot(
                x='stage', y=col, hue='stage', data=df_bearing, palette='Set2' if canal_nombre=='H' else 'Set3', legend=False, ax=ax3
            )
            ax3.set_title(f'Boxplots: {col}', fontsize=9, fontweight='bold')
            ax3.set_xlabel('')
            ax3.set_ylabel('')
            ax3.set_ylim(0, 1.05)  # Escala fija de 0 a 1

        fig.suptitle(f'Comparativa Integral 16x3 - {b_id} (Canal {canal_nombre})', fontsize=16, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Nombre del archivo ordenado por rodamiento primero y luego canal
        output_file = os.path.join(output_dir, f"{b_id}_canal_{canal_nombre}_wavelets_16x3.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"     Guardado: {output_file}")

print("\n¡Proceso finalizado con éxito para todos los rodamientos y canales!")