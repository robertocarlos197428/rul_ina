# Documentación Técnica — Script de Extracción de Features (`extract_features.py`)

## 1. Objetivo del Script
Este script procesa los archivos de vibración en formato bruto (RAW) del dataset **PRONOSTIA/FEMTO** correspondientes a los 17 rodamientos completos hasta la falla. Su propósito es aplicar el preprocesamiento acordado, extraer exactamente **68 features** (temporales, espectrales, de descomposición wavelet y cruzadas) por cada ráfaga de vibración, y consolidar los resultados en un archivo optimizado (`.parquet`) y un archivo de respaldo (`.csv`)[cite: 1, 2].

---

## 2. Entradas y Salidas
* **Directorio de entrada (RAW):** `rul_ina/data/raw_femto/` (contiene las carpetas de los 17 rodamientos con sus archivos de ráfagas `.csv`)[cite: 1, 2].
* **Archivos de salida (Procesados):**
  * `rul_ina/data/processed/features_68.parquet` (Archivo principal optimizado)[cite: 2].
  * `rul_ina/data/processed/features_68.csv` (Archivo secundario para auditoría visual)[cite: 2].

---

## 3. Estructura de las Filas (Columnas del Dataset Resultante)
Cada fila del archivo procesado representa **una ráfaga de vibración** e incluye:
1. **Identificadores y Tiempo:**
   * `bearing_id`: Identificador del rodamiento (ej. `Bearing1_1`)[cite: 1].
   * `burst_id`: Nombre del archivo de ráfaga original (ej. `acc_00001.csv`)[cite: 1].
   * `burst_index`: Índice secuencial de la ráfaga (iniciando en `0`)[cite: 1].
   * `time_seconds`: Tiempo transcurrido acumulado en segundos desde el inicio del rodamiento (partiendo en `0.0`, avanzando de forma continua según las marcas de tiempo del dataset)[cite: 1].
2. **Las 68 Features:**
   * **24 Features Temporales** (12 para el canal H y 12 para el canal V)[cite: 2].
   * **10 Features Espectrales** (5 para H y 5 para V)[cite: 2].
   * **32 Features WPD** (16 para H y 16 para V)[cite: 2].
   * **2 Features Cruzadas** (Correlación de Pearson y Ratio de Energía H-V)[cite: 2].

---

## 4. Pipeline de Procesamiento por Ráfaga

### A. Lectura de Datos RAW
* Cada archivo CSV se lee sin cabecera[cite: 1].
* Se seleccionan específicamente la **Columna 4 (Aceleración H)** y la **Columna 5 (Aceleración V)**, omitiendo las primeras 4 columnas correspondientes a la marca de tiempo del registrador (hora, minuto, segundo y microsegundo)[cite: 1].

### B. Preprocesamiento (Etapa 4)
1. **Eliminación de Componente Continua (DC):**
   * Se resta la media aritmética a cada señal de forma independiente: $x_{DC} = x - \text{mean}(x)$[cite: 2].
2. **Filtro Pasa-Altos:**
   * Se aplica un filtro Butterworth de orden 5 con un **corte inferior de 10 Hz** y una frecuencia de muestreo de **25,6 kHz**, sin corte superior[cite: 2].
3. **Ausencia de Normalización:**
   * Las señales conservan su escala física original[cite: 2].

### C. Extracción de Features (Etapa 5)
* **Temporales (12 por canal):** RMS, Peak (sin módulo), Peak-to-Peak (P2P), Variance (poblacional, `ddof=0`), Standard Deviation (poblacional, `ddof=0`), MAV, Kurtosis (Fisher, vía `scipy.stats`), Skewness (vía `scipy.stats`), Crest Factor, Impulse Factor, Shape Factor y Margin Factor[cite: 1, 2].
* **Espectrales (5 por canal):** Spectral Centroid, Spectral RMS, Spectral Std, Spectral Entropy (Shannon) y Spectral Flatness (con protección numérica $\epsilon = 10^{-10}$ para evitar logaritmos de cero)[cite: 2].
* **Wavelet Packet Decomposition - WPD (16 por canal):** Descomposición nivel 4 usando la wavelet `db4` con modo `periodization` para obtener las energías relativas de las 16 subbandas[cite: 2].
* **Cruzadas (2 en total):** Coeficiente de correlación de Pearson entre H y V, y el ratio de energía entre ambos canales ($ER_{HV}$)[cite: 2].

---

## 5. Criterios de Robustez y Manejo Numérico Implementados
* **Población en Varianza y Std:** Se utiliza `ddof=0` para mantener coherencia matemática con los momentos de `scipy.stats`[cite: 1].
* **Protección de divisiones por cero:** Se validan denominadores antes de calcular factores de cresta, impulso, forma, margen y entropías espectrales para evitar excepciones de tipo `ZeroDivisionError`.