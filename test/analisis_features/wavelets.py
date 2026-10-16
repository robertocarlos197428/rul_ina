import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo visual
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Cargar el archivo real desde tu ruta
input_path = r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed\features_68.parquet"
output_dir = r"rul_ina/analisis_features/wavelets"
os.makedirs(output_dir, exist_ok=True)

print(f"Cargando datos desde: {input_path}")
df = pd.read_parquet(input_path)

# Seleccionamos un bearing de ejemplo (puedes iterar sobre todos si lo deseas)
bearing_id = df['bearing_id'].iloc[0]
df_bearing = df[df['bearing_id'] == bearing_id].sort_values('time_seconds')

print(f"Analizando bearing_id: {bearing_id} con {len(df_bearing)} registros.")

# Definir las listas de características wavelet para Horizontal y Vertical
features_h = [f'WPD_Erel_{i}_H' for i in range(16)]
features_v = [f'WPD_Erel_{i}_V' for i in range(16)]

# Crear una etiqueta de etapa temporal para los boxplots (Gráfico C)
df_bearing['stage'] = pd.qcut(df_bearing['time_seconds'], q=3, labels=['Inicio (Sano)', 'Intermedio (Degradación)', 'Crítico (Fallo)'])

# ==========================================
# 1. MAPAS DE CALOR (Heatmaps)
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

sns.heatmap(df_bearing[features_h].T, ax=axes[0], cmap='viridis', cbar=True, xticklabels=max(1, len(df_bearing)//20))
axes[0].set_title(f'Mapa de Calor - Wavelets Canal Horizontal ({bearing_id})', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Bandas H (0-15)', fontsize=10)

sns.heatmap(df_bearing[features_v].T, ax=axes[1], cmap='plasma', cbar=True, xticklabels=max(1, len(df_bearing)//20))
axes[1].set_title(f'Mapa de Calor - Wavelets Canal Vertical ({bearing_id})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Índice Temporal / Burst', fontsize=10)
axes[1].set_ylabel('Bandas V (0-15)', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, '01_heatmap_wavelets_HV.png'), dpi=300)
plt.close()

# ==========================================
# 2. GRÁFICOS B: Subplots 4x4 (Canal H y V separados)
# ==========================================
# Canal Horizontal (B)
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
axes = axes.flatten()
for i, col in enumerate(features_h):
    ax = axes[i]
    ax.plot(df_bearing['time_seconds'], df_bearing[col], color='darkblue', linewidth=1.2)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
fig.suptitle(f'Evolución Temporal - Wavelets Canal Horizontal ({bearing_id})', fontsize=14, fontweight='bold', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(output_dir, '02_subplots_wavelets_H.png'), dpi=300)
plt.close()

# Canal Vertical (B)
fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True)
axes = axes.flatten()
for i, col in enumerate(features_v):
    ax = axes[i]
    ax.plot(df_bearing['time_seconds'], df_bearing[col], color='darkred', linewidth=1.2)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
fig.suptitle(f'Evolución Temporal - Wavelets Canal Vertical ({bearing_id})', fontsize=14, fontweight='bold', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(output_dir, '02_subplots_wavelets_V.png'), dpi=300)
plt.close()

# ==========================================
# 3. GRÁFICOS C: Boxplots por Etapas
# ==========================================
# Canal Horizontal (C)
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(features_h):
    ax = axes[i]
    sns.boxplot(x='stage', y=col, data=df_bearing, palette='Set2', ax=ax)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
fig.suptitle(f'Distribución por Etapas - Wavelets Canal Horizontal ({bearing_id})', fontsize=14, fontweight='bold', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(output_dir, '03_boxplots_wavelets_H.png'), dpi=300)
plt.close()

# Canal Vertical (C)
fig, axes = plt.subplots(4, 4, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(features_v):
    ax = axes[i]
    sns.boxplot(x='stage', y=col, data=df_bearing, palette='Set3', ax=ax)
    ax.set_title(col, fontsize=9, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
fig.suptitle(f'Distribución por Etapas - Wavelets Canal Vertical ({bearing_id})', fontsize=14, fontweight='bold', y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(os.path.join(output_dir, '03_boxplots_wavelets_V.png'), dpi=300)
plt.close()

print("¡Proceso finalizado! Todas las imágenes reales han sido guardadas en rul_ina/analisis_features/wavelets/")