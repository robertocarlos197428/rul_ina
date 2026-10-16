import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuración de estilo visual limpio
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# 1. Cargar el archivo procesado desde la ruta original
input_path = r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed\features_68.parquet"
output_dir = r"rul_ina/analisis_features/spectrum"
os.makedirs(output_dir, exist_ok=True)

print(f"Cargando datos desde: {input_path}")
df = pd.read_parquet(input_path)

# Obtener la lista ordenada de los rodamientos
bearing_ids = sorted(df['bearing_id'].unique())
print(f"Se encontraron {len(bearing_ids)} rodamientos.")

# 2. Identificar las características espectrales de forma robusta
# Excluimos rodamientos, tiempo, cycle, RMS y wavelets (WPD)
exclude_terms = ['WPD', 'RMS', 'bearing_id', 'time_seconds', 'cycle']
spectral_cols = [c for c in df.columns if not any(term in c for term in exclude_terms)]

# Separar por canal H y V
spectral_cols_h = sorted([c for c in spectral_cols if '_H' in c])
spectral_cols_v = sorted([c for c in spectral_cols if '_V' in c])

# Tomar los primeros 10 de cada uno (o los que correspondan al espectro, típicamente 5 por canal o 10 por canal)
# Dividimos en bloques de 5 features por grilla
blocks_h = [spectral_cols_h[:5], spectral_cols_h[5:10]] if len(spectral_cols_h) >= 10 else [spectral_cols_h[:5]]
blocks_v = [spectral_cols_v[:5], spectral_cols_v[5:10]] if len(spectral_cols_v) >= 10 else [spectral_cols_v[:5]]

print(f"Features espectrales H detectadas ({len(spectral_cols_h)}): {spectral_cols_h}")
print(f"Features espectrales V detectadas ({len(spectral_cols_v)}): {spectral_cols_v}")

# Función para generar la grilla 17 filas (rodamientos) x 5 columnas (features)
def generar_grilla_espectral(canal_nombre, bloque_features, numero_bloque):
    if not bloque_features:
        print(f"Bloque {numero_bloque} para canal {canal_nombre} está vacío. Saltando...")
        return
        
    fig, axes = plt.subplots(17, len(bloque_features), figsize=(4 * len(bloque_features), 34), sharex=False, sharey=False)
    
    # Asegurar que axes sea bidimensional incluso si len(bloque_features) == 1
    if len(bloque_features) == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, b_id in enumerate(bearing_ids):
        df_b = df[df['bearing_id'] == b_id].sort_values('time_seconds')
        
        for col_idx, feat in enumerate(bloque_features):
            ax = axes[row_idx, col_idx]
            
            # Graficar la feature bruta en el tiempo sin escalar
            if len(df_b) > 0 and feat in df_b.columns:
                ax.plot(df_b['time_seconds'], df_b[feat], color='darkcyan' if canal_nombre=='H' else 'darkorange', linewidth=0.9)
            
            # Títulos y etiquetas de ejes
            if row_idx == 0:
                ax.set_title(f'{feat}', fontsize=9, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f'{b_id}', fontsize=8, fontweight='bold')
                
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f'Espectral - Canal {canal_nombre} (Bloque {numero_bloque}) | 17 Rodamientos x {len(bloque_features)} Features', fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = os.path.join(output_dir, f"spectrum_canal_{canal_nombre}_bloque_{numero_bloque}_17x5.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Guardado exitosamente: {output_file}")

# 3. Generar las imágenes maestras en la carpeta spectrum corregida
for idx, bloque in enumerate(blocks_h, 1):
    generar_grilla_espectral('H', bloque, idx)

for idx, bloque in enumerate(blocks_v, 1):
    generar_grilla_espectral('V', bloque, idx)

print("\n¡Proceso de la familia espectral en la carpeta spectrum finalizado con éxito!")