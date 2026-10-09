import pandas as pd

# Cargar el archivo parquet
file_path = "rul_ina/data/processed/features_68.parquet"
df = pd.read_parquet(file_path)

print("=== AUDITORÍA DEL ARCHIVO PARQUET ===")
print(f"1. Dimensiones totales: {df.shape} (Filas: {df.shape[0]} ráfagas, Columnas: {df.shape[1]})")
print(f"   - Esperado: 3 columnas de ID + 68 features = 71 columnas en total.\n")

print(f"2. Rodamientos encontrados en el archivo:")
print(df['bearing_id'].value_counts())

print(f"\n3. Primeras 5 columnas (IDs):")
print(df.columns[:5].tolist())

print(f"\n4. Verificación de valores nulos o infinitos:")
nulls = df.isnull().sum().sum()
infs = (df.isin([float('inf'), float('-inf')])).sum().sum()
print(f"   - Total de valores NaN en el archivo: {nulls}")
print(f"   - Total de valores Infinitos: {infs}")

print(f"\n5. Vista previa de las primeras 2 filas:")
print(df.iloc[:2, :6]) # Muestra los primeros identificadores y las primeras features