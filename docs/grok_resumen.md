# Resumen del trabajo — Features y suavizado (rul_ina)

## Contexto
Proyecto de predicción de RUL en rodamientos (INACAP).  
Datos de features extraídos de vibración (canales H y V) a partir de datasets tipo FEMTO / XJTU-SY.

**Carpeta de trabajo principal:**  
`C:\Users\a\Downloads\CursoDL\rul_ina\data\processed`

---

## 1. Reducción de features: 68 → 48

Se partió de `features_68.parquet` y se eliminaron **20 features** redundantes o poco útiles, sin tocar filas ni columnas de identificación.

### Features eliminadas (20)
| Familia | Features |
|--------|----------|
| Tiempo | `Std_H/V`, `Variance_H/V`, `MAV_H/V`, `ImpulseFactor_H/V`, `MarginFactor_H/V`, `Peak_H/V`, `P2P_H/V`, `ShapeFactor_H/V` |
| Espectral | `SpectralFlatness_H/V`, `SpectralRMS_H/V` |

### Features que quedaron (48)
| Familia | Cantidad | Detalle |
|--------|----------|---------|
| Tiempo | 8 | `RMS`, `Kurtosis`, `Skewness`, `CrestFactor` (H y V) |
| Espectral | 6 | `SpectralCentroid`, `SpectralStd`, `SpectralEntropy` (H y V) |
| WPD (energía relativa) | 32 | `WPD_Erel_0` … `WPD_Erel_15` (H y V) |
| Cruzadas H–V | 2 | `Pearson_HV`, `EnergyRatio_HV` |

### Archivos generados
- `features_48.parquet`
- `features_48.csv`

**Columnas de ID (sin cambios):**  
`bearing_id`, `burst_id`, `burst_index`, `time_seconds`

---

## 2. Visualización de features (crudas)

Script para graficar en estilo grilla (filas = rodamientos, columnas = features), eje X en **horas**.

### Tiempo + espectral (7 por eje)
- `time_spectral_features_axis_H.png`
- `time_spectral_features_axis_V.png`

Features por eje:  
`RMS`, `Kurtosis`, `Skewness`, `CrestFactor`, `SpectralCentroid`, `SpectralStd`, `SpectralEntropy`

---

## 3. Suavizado moderado

Se definió un pipeline de suavizado **por rodamiento** (no global) para reducir ruido sin deformar la tendencia de degradación.

### Pipeline
1. **Hampel** (ventana 11, k=4.0) → quita picos aislados  
2. **Mediana móvil** (2 % de la vida del bearing)  
3. **Media móvil** (1 % de la vida del bearing)

> Nota: un suavizado más agresivo (mediana 10 % + media 5 %) deformaba mucho las curvas; se descartó.

### Gráficos suavizados generados
Carpeta: `rul_ina/analisis_features/suavizado_moderado/`

| # | Archivo | Contenido |
|---|---------|-----------|
| 1–2 | Eje H / V tiempo+espectral | 7 features c/u |
| 3 | `wpd_H_0_7` | WPD Erel 0–7 canal H |
| 4 | `wpd_H_8_15` | WPD Erel 8–15 canal H |
| 5 | `wpd_V_0_7` | WPD Erel 0–7 canal V |
| 6 | `wpd_V_8_15` | WPD Erel 8–15 canal V |
| 7 | `cruzadas_HV` | Pearson + EnergyRatio |

En los gráficos se muestra la curva **cruda en gris** y la **suavizada en azul**.

---

## 4. Dataset suavizado exportado

Se aplicó el mismo suavizado a **todas las features** y se guardó un nuevo dataset.

### Archivos
- `features_48_suavizado.parquet`
- `features_48_suavizado.csv`

### Características
- Mismas columnas de identificación y tiempo  
- Mismo número de filas (el suavizado **no reduce** ráfagas)  
- Solo cambian los valores de las 48 features  
- Suavizado independiente por cada `bearing_id`

---

## 5. Otros temas tratados (contexto previo)

- Extracción de subset de 6 features (`features_6`) para análisis de distancia de Mahalanobis (MD²) y aportes por característica  
- Referencia sana para MD: primeras **70 ráfagas** por rodamiento  
- Descomposición de MD² en aportes porcentuales (pueden ser negativos por correlación)  
- Arquitectura de redes y despliegue en Raspberry Pi (línea de trabajo del proyecto RUL)

---

## Estado actual (datos)
