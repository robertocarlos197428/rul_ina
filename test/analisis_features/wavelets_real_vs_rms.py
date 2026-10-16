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
output_dir = r"rul_ina/analisis_features/wavelets/Wavelets_real"
os.makedirs(output_dir, exist_ok=True)

print(f"Cargando datos desde: {input_path}")
df = pd.read_parquet(input_path)

# Obtener la lista de todos los rodamientos únicos en el dataset
bearing_ids = df['bearing_id'].unique()
print(f"Se encontraron {len(bearing_ids)} rodamientos para procesar: {bearing_ids}")

# Iterar sobre cada rodamiento
for b_id in bearing_ids:
    print(f"\nProcesando rodamiento: {b_id}...")
    df_bearing = df[df['bearing_id'] == b_id].sort_values('time_seconds')
    
    if len(df_bearing) == 0:
        print(f"  -> Advertencia: {b_id} no tiene registros. Saltando...")
        continue

    # Crear la etiqueta de etapa temporal para los boxplots con duplicates='drop'
    df_bearing['stage'] = pd.qcut(
        df_bearing['time_seconds'], q=3, labels=['Inicio (Sano)', 'Intermedio (Degradación)', 'Crítico (Fallo)'], duplicates='drop'
    )
    
    # Procesar ambos canales: Horizontal (_H) y Vertical (_V)
    canales = [
        ('H', [f'WPD_Erel_{i}_H' for i in range(16)], 'RMS_H'),
        ('V', [f'WPD_Erel_{i}_V' for i in range(16)], 'RMS_V')
    ]
    
    for canal_nombre, features, rms_col in canales:
        print(f"  -> Generando gráfico 16x3 con energía real para canal {canal_nombre}...")
        
        fig, axes = plt.subplots(16, 3, figsize=(18, 40), sharex=False)
        
        for i, col in enumerate(features):
            # --- Columna 1: RMS Global de Referencia ---
            ax1 = axes[i, 0]
            ax1.plot(df_bearing['time_seconds'], df_bearing[rms_col], color='gray', linewidth=1.0, alpha=0.8)
            ymin, ymax = ax1.get_ylim() # Capturamos el límite del RMS para unificar la escala visual
            ax1.set_title(f'Ref: {rms_col} (Fila {i})', fontsize=9, fontweight='bold')
            ax1.set_ylabel('RMS Total', fontsize=8)
            ax1.grid(True, linestyle='--', alpha=0.5)

            # --- Columna 2: Evolución Temporal con Energía Real (WPD_Erel * RMS) ---
            ax2 = axes[i, 1]
            energia_real_banda = df_bearing[col] * df_bearing[rms_col]
            ax2.plot(df_bearing['time_seconds'], energia_real_banda, color='darkblue' if canal_nombre=='H' else 'darkred', linewidth=1.2)
            ax2.set_title(f'Energía Real: {col}', fontsize=9, fontweight='bold')
            ax2.set_ylabel('Energía Absoluta', fontsize=8)
            ax2.grid(True, linestyle='--', alpha=0.5)
            ax2.set_ylim(0, ymax)  # Escala unificada al RMS máximo

            # --- Columna 3: Boxplots por Etapas con Energía Real ---
            ax3 = axes[i, 2]
            df_bearing_temp = df_bearing.copy()
            df_bearing_temp[col + '_real'] = df_bearing[col] * df_bearing[rms_col]
            
            sns.boxplot(
                x='stage', y=col + '_real', hue='stage', data=df_bearing_temp, palette='Set2' if canal_nombre=='H' else 'Set3', legend=False, ax=ax3
            )
            ax3.set_title(f'Boxplots Real: {col}', fontsize=9, fontweight='bold')
            ax3.set_xlabel('')
            ax3.set_ylabel('')
            ax3.set_ylim(0, ymax)  # Escala unificada al RMS máximo

        fig.suptitle(f'Comparativa 16x3 Energía Real - {b_id} (Canal {canal_nombre})', fontsize=16, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Nombre del archivo ordenado por rodamiento primero y luego canal
        output_file = os.path.join(output_dir, f"{b_id}_canal_{canal_nombre}_energia_real_16x3.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"     Guardado: {output_file}")

print("\n¡Proceso masivo de energía real finalizado con éxito para todos los rodamientos y canales!")