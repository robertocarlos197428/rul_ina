import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_ranking(csv_path):
    print(f"Cargando características desde: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Verificar si las columnas necesarias existen (asumiendo 'Score' o 'Quality_Score')
    score_col = 'Score' if 'Score' in df.columns else 'Quality_Score'
    if score_col not in df.columns:
        raise ValueError(f"El archivo CSV debe contener una columna de puntaje ('Score' o 'Quality_Score'). Columnas encontradas: {df.columns.tolist()}")
    
    # Ordenar de mayor a menor Quality Score
    df_sorted = df.sort_values(by=score_col, ascending=False).reset_index(drop=True)
    
    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(df_sorted) + 1), df_sorted[score_col], marker='o', color='#1f77b4', linestyle='-', linewidth=2, markersize=4)
    
    # Marcar la línea de corte en el Top 30
    if len(df_sorted) >= 30:
        plt.axvline(x=30, color='darksalmon', linestyle='--', linewidth=2, label='Corte Top 30')
    
    plt.title("Curva de Quality Score de las 68 Características (Ranking)", fontsize=14, fontweight='bold')
    plt.xlabel("Posición en el Ranking", fontsize=11)
    plt.ylabel("Quality Score (Aptitud de Pronóstico)", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    output_img = "ranking_quality_score_curve.png"
    plt.savefig(output_img, dpi=300)
    print(f"¡Gráfico de ranking guardado con éxito en: {output_img}!")
    
    # Muestra el gráfico en pantalla
    plt.show()
    plt.close()

if __name__ == "__main__":
    # Ruta exacta que indicaste
    CSV_PATH = r"C:\Users\emaker\Downloads\RUL\rul_ina\data\processed\ranking_68_features.csv"
    plot_ranking(CSV_PATH)