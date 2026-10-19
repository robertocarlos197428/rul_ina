# Resumen de la tarde — 18 de octubre de 2026

De 17 features candidatas a un primer modelo Mamba con validación dejando un rodamiento fuera (LOBO). Continúa el resumen de la mañana (`resumen_sesion_2026-10-18.md`).

---

## 1. De 17 a 13 features

- Las 17 candidatas (corte de área 0,31) se agruparon por forma de la mediana: **suben**, **bajan** y **suben y bajan**.
- Se dejaron fuera las 4 que cambian de sentido: `Erel_3_V`, `Erel_4_V`, `Erel_11_V` y `Erel_12_V`.
- **Quedan 13 features:**

| Grupo | Features |
|---|---|
| Suben (7) | `RMS_H`, `RMS_V`, `Kurtosis_H`, `Kurtosis_V`, `CrestFactor_H`, `CrestFactor_V`, `SpectralStd_H` |
| Bajan (6) | `Erel_5_H`, `Erel_3_H`, `Erel_6_H`, `Erel_4_H`, `Erel_5_V`, `Skewness_H` |

- Por familia: Time 7, Spectral 1, Wavelets 5, HV 0.
- **SpectralStd:** mide qué tan repartida está la energía en frecuencia (el ancho del espectro). No depende de dónde está centrada.

## 2. Datos preparados

- `features_13.py` → `features_13.parquet` y `.csv` (las 13 features + `bearing_id`, `burst_id`, `burst_index`, `time_seconds`).
- `agregar_rul.py` → `features_13_rul.parquet` y `.csv`, con la columna `rul_frac`.
- **Etiqueta elegida: fracción de vida.** Para cada rodamiento, `rul_frac = 1 − (t − t_inicial) / (t_final − t_inicial)`, de 1 a 0. El RUL por tramos queda para más adelante.

## 3. Cuidados sobre fuga de información

- El **mín-máx por rodamiento** y el **suavizado centrado** (cuyo tamaño depende de la vida total) usan información del futuro.
- Para lo que hicimos (análisis y selección) no es grave, y el mín-máx solo estaba en los gráficos, no en los datos.
- **Para el modelo:** normalizar con estadísticas solo del entrenamiento y usar suavizado causal (solo hacia atrás). Los scripts de modelado ya lo hacen así.
- Queda un sesgo leve: las features se eligieron mirando los 17 rodamientos.

## 4. Plan de modelado (decidido)

- Mamba pequeño, solo CPU (el modelo irá a una Raspberry Pi 5), con las 13 features y `rul_frac`.
- Ventanas de 50 ráfagas, predicción de la última ráfaga.
- Validación **LOBO**: 17 entrenamientos, cada uno con 16 rodamientos y prueba en el que queda fuera.
- Normalización con estadísticas del entrenamiento, 20 épocas fijas.
- Primero un **baseline lineal** como punto de comparación.
- Carpetas: `modelado\01_baseline_lineal` y `modelado\02_mamba`.

## 5. Baseline lineal (`baseline_lineal.py`)

- Regresión lineal sobre el promedio causal de las últimas 50 ráfagas, con LOBO. Corre en menos de 1 segundo.
- **MAE promedio ≈ 0,195** (en fracción de vida). Un modelo que predijera siempre 0,5 tendría 0,25.
- Mejores: `Bearing1_3` (0,046), `1_1` (0,102), `1_7` (0,119). Peores: `1_2` (0,352), `2_7` (0,308), `1_4` (0,278).
- Se vio una caída temprana de la predicción en `1_2`, `2_1` y `2_2`, mesetas en varios y un disparo al final en `1_4`.

## 6. Mamba (`mamba_lobo.py`)

- Mamba en PyTorch puro (sin CUDA), 17.345 parámetros: 2 capas, d_model 32, d_state 8. El scan es un bucle simple, fácil de exportar después.
- **Las ventanas** avanzan de a una ráfaga (1-50, 2-51, 3-52…), así que son casi iguales. Eso lo debí advertir antes de fijarlo.
- Se agregó `PASO_ENTRENAMIENTO`: cada época usa 1 de cada N ventanas, cambiando el punto de partida entre épocas.
- **Tiempos (2 hilos):** ~30 min por rodamiento con todas las ventanas, ~8 min con paso 4 y ~3 a 6 min con paso 10.

**Resultados de los primeros 6 rodamientos (MAE):**

| Rodamiento | Baseline | Mamba paso 1 | Mamba paso 10 |
|---|---|---|---|
| 1_1 | 0,102 | 0,074 | 0,099 |
| 1_2 | 0,352 | 0,412 | 0,367 |
| 1_3 | 0,046 | 0,113 | 0,124 |
| 1_4 | 0,278 | 0,089 | 0,114 |
| 1_5 | 0,163 | 0,151 | 0,133 |
| 1_6 | 0,239 | 0,203 | 0,155 |
| **Promedio** | **0,195** | **0,174** | **0,165** |

- **Lectura:** Mamba mejora al baseline en 4 de 6 y por unos 0,03 de MAE (15%). La mejora grande está en `1_4`.
- El **paso 10 no salió peor** que usar todas las ventanas (diferencias de ±0,05 por rodamiento, ruido de entrenamiento), así que se usa para los 17.
- Con 6 de 17 rodamientos y una sola semilla todavía no se puede concluir.
- `Bearing1_2` sigue con error alto (~0,37) en ambos modelos.

## 7. Dónde estamos

- La corrida con paso 10 tiene los 6 primeros rodamientos. Faltan los otros 11 (unas 1 a 2 horas).
- La corrida larga (paso 1) se puede cortar: los rodamientos terminados quedan guardados en `folds` y se retoma sin repetirlos.

## 8. Caminos posibles

1. **Terminar los 17 con paso 10** y comparar con el baseline rodamiento por rodamiento.
2. **Revisar `Bearing1_2`** (y los de caída temprana, `2_1` y `2_2`).
3. **Probar el baseline en modo `"actual"`** (sin promedio) para ver cuánto aporta el suavizado causal.
4. **Repetir con otra semilla** para ver cuánto varían los resultados.
5. **RUL por tramos** (con un máximo en la etapa sana), si la fracción de vida resulta difícil de predecir al inicio.
6. **Modelo final** entrenado con los 17 rodamientos y pruebas de exportación y velocidad para la Raspberry Pi 5.
7. **Git:** commitear los scripts ahora y los resultados cuando termine la corrida (sin los parquet grandes de `data\processed`).

## 9. Archivos de la tarde

| Archivo | Para qué |
|---|---|
| `grupos_17_features.py` | Las 17 features agrupadas por forma (una fila por grupo) |
| `features_13.py` | Deja las 13 features finales |
| `agregar_rul.py` | Agrega `rul_frac` |
| `baseline_lineal.py` | Baseline lineal con LOBO |
| `mamba_lobo.py` | Mamba pequeño con LOBO (paso de entrenamiento configurable) |