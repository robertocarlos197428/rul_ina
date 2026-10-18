import os
import pandas as pd
import matplotlib.pyplot as plt

# Estilo visual (igual al script anterior)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

# Rutas
input_path = r"C:\Users\a\Downloads\CursoDL\rul_ina\data\processed\features_68.parquet"
output_dir = r"C:\Users\a\Downloads\CursoDL\rul_ina\analisis_features\wavelets\wavelets_relativa_17"
os.makedirs(output_dir, exist_ok=True)

print(f"Cargando datos desde: {input_path}")
df = pd.read_parquet(input_path).sort_values(['bearing_id', 'time_seconds'])

bearing_ids = sorted(df['bearing_id'].unique())
n_rows = len(bearing_ids)
print(f"Rodamientos encontrados: {n_rows}")
if n_rows != 17:
    print(f"  -> Aviso: se esperaban 17 rodamientos y hay {n_rows}.")

datos = {b: df[df['bearing_id'] == b] for b in bearing_ids}

canales = [
    ('H', 'darkblue'),
    ('V', 'darkred'),
]

# 4 figuras por canal: nodos 0-3, 4-7, 8-11, 12-15
grupos = [list(range(g, g + 4)) for g in range(0, 16, 4)]

for canal, color in canales:
    for nodos in grupos:
        print(f"Generando canal {canal}, nodos {nodos[0]}-{nodos[-1]}...")

        fig, axes = plt.subplots(n_rows, 4, figsize=(18, 2.4 * n_rows), sharex=False, sharey=False)

        for r, b_id in enumerate(bearing_ids):
            d = datos[b_id]
            for c, nodo in enumerate(nodos):
                col = f'WPD_Erel_{nodo}_{canal}'
                ax = axes[r, c]

                ax.plot(d['time_seconds'], d[col], color=color, linewidth=1.2)

                ax.set_title(f'{b_id} | Energía Relativa: {col}', fontsize=9, fontweight='bold')
                ax.set_ylabel('Erel', fontsize=8)
                ax.grid(True, linestyle='--', alpha=0.5)

        fig.suptitle(
            f'Energía Relativa por nodo - Canal {canal} (Nodos {nodos[0]} a {nodos[-1]})',
            fontsize=16, fontweight='bold', y=0.995
        )
        plt.tight_layout(rect=[0, 0, 1, 0.99])

        output_file = os.path.join(
            output_dir, f"canal_{canal}_nodos_{nodos[0]}-{nodos[-1]}_energia_relativa_17x4.png"
        )
        plt.savefig(output_file, dpi=200)
        plt.close()
        print(f"  Guardado: {output_file}")

print("\nListo: 8 gráficos generados.")
