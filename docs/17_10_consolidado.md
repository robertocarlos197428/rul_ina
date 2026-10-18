# Resumen unificado — Pipeline RUL: features, métricas y suavizado

**Fecha:** 17 de octubre de 2026  
**Proyecto:** Predicción de RUL en rodamientos (INACAP)  
**Datos:** FEMTO / PRONOSTIA — 17 rodamientos, canales H y V  
**Carpeta principal:** `rul_ina/data/processed`

---

## 1. Contexto y objetivo del día

Análisis cuantitativo y visual de características de vibración, depuración de métricas de calidad, reducción de redundancias (68 → 48 features), definición de un suavizado usable y exportación del dataset depurado/suavizado para HI, MD y modelos posteriores.

---

## 2. Dominio espectral y hallazgos físicos

### Evaluación de features espectrales
Se revisaron `SpectralCentroid`, `SpectralRMS`, `SpectralStd`, `SpectralEntropy` y `SpectralFlatness` en los 17 rodamientos.

- **`SpectralStd_H` — líder:** mejor calidad global (p. ej. Score EWM ~0,64; robustez ~0,99). Curva suave, continua y monótona en la mayoría de casos.
- **`SpectralRMS` ≈ `RMS` temporal:** se explica por Parseval/Rayleigh (conservación de energía tiempo–frecuencia); aporta poca información nueva → candidata a eliminación por redundancia.
- **Canal H vs V:** H suele verse más limpio y monótono (carga radial del banco actúa sobre todo en horizontal).

### Caso `Bearing1_3` (entropía vs std espectral)
| Fase | Qué ocurre |
|------|------------|
| Falla localizada (~2–5 h) | Picos tonales ordenan el espectro → **baja** `SpectralEntropy`; `SpectralStd` se mantiene alta |
| Colapso (>~5,5 h) | Spalling / ruido de banda ancha → **sube** la entropía y cae la std espectral |

---

## 3. Métricas de calidad de features

| Métrica | Significado | Decisión |
|---------|------------|----------|
| **Mon_sign** | Signos locales ráfaga a ráfaga | **Descartada** (hipersensible al ruido, no discrimina) |
| **Mon_spearman** | Tendencia global (Spearman vs tiempo) | **Adoptada** como monotonía principal |
| **Trend (Pearson)** | Correlación lineal feature–tiempo | Se mantiene |
| **Prognosability** | Dispersión poblacional del valor en EOL | Se mantiene (un valor por feature) |
| **Robustez (EWMA)** | Tolerancia al ruido | Se fija **α = 0,10** (se descartan 0,05 / 0,20 / 0,30 / 0,50) |

**Fuente consolidada:** `metricas_hi_por_rodamiento.csv` (luego reducida a columnas útiles).

**Hallazgos globales**
- Pocas features superan 0,7 de monotonía mediana; las mejores no funcionan en *todos* los rodamientos (`SpectralStd_H` cae fuerte en Bearing1_3).
- Por familia, **WPD (Erel)** destaca; tiempo puro es lo más débil.
- Robustez alta en general; prognosabilidad baja (mediana ~0,19).
- Rodamientos difíciles: Bearing3_2, Bearing1_2. Más fácil: Bearing1_7.

**Nota:** `Mon_spearman` y `Trend` parecen en valor absoluto en la tabla → no indican dirección (sube/baja).

---

## 4. WPD: Erel vs “energía real”

- **`WPD_Erel_i`:** fracción de energía del nodo i (0–1); 16 nodos por canal (nivel 4) deberían sumar ~1.
- **Erel × RMS:** reparto proporcional del RMS, no es energía física por banda. Se parece más al RMS global en muchos nodos y **oculta** la redistribución temprana entre bandas.
- **Decisión:** trabajar con **Erel** (no Erel × RMS).
- Ningún nodo Erel se comporta igual en los 17 rodamientos.

---

## 5. Reducción 68 → 48 features (por redundancia)

Criterio principal: correlación de Spearman alta de forma consistente entre rodamientos (no fue un filtro solo por monotonía).

### Eliminadas (20) — H y V
`Std`, `Variance`, `MAV`, `ImpulseFactor`, `MarginFactor`, `Peak`, `P2P`, `ShapeFactor`, `SpectralFlatness`, `SpectralRMS`

### Quedan (48)
| Familia | n | Features |
|---------|---|----------|
| Tiempo | 8 | `RMS`, `Kurtosis`, `Skewness`, `CrestFactor` (H, V) |
| Espectral | 6 | `SpectralCentroid`, `SpectralStd`, `SpectralEntropy` (H, V) |
| WPD Erel | 32 | `WPD_Erel_0` … `WPD_Erel_15` (H, V) |
| Cruzadas | 2 | `Pearson_HV`, `EnergyRatio_HV` |

### Archivos
- `features_48.parquet` / `.csv`  
- Verificado: ~24.889 filas, 17 bearings, 52 columnas (48 + 4 ID), sin NaN/inf.

**ID:** `bearing_id`, `burst_id`, `burst_index`, `time_seconds`

---

## 6. Top features (mediana sobre 17 rodamientos)

| Rank | Feature | Mediana Spearman | Trend | Robustez α=0,10 | Prognosability |
|-----:|---------|------------------:|------:|----------------:|---------------:|
| 1 | **SpectralStd_H** | 0,831 | 0,820 | 0,987 | 0,757 |
| 2 | **WPD_Erel_6_H** | 0,757 | 0,761 | 0,877 | 0,642 |
| 3 | WPD_Erel_14_H | 0,729 | 0,601 | 0,871 | 0,393 |
| 4 | WPD_Erel_13_H | 0,696 | 0,527 | 0,874 | 0,397 |
| 5 | WPD_Erel_4_H | 0,692 | 0,701 | 0,884 | 0,707 |
| 6 | WPD_Erel_12_V | 0,669 | 0,545 | 0,906 | 0,241 |
| 7 | WPD_Erel_15_H | 0,669 | 0,627 | 0,874 | 0,287 |
| 8 | WPD_Erel_7_H | 0,668 | 0,642 | 0,878 | 0,443 |
| 9 | SpectralCentroid_V | 0,634 | 0,608 | 0,987 | 0,156 |
| 10 | WPD_Erel_3_H | 0,633 | 0,621 | 0,881 | 0,710 |
| … | … | … | … | … | … |

Dominan **canal H** y nodos **WPD**, con `SpectralStd_H` como referencia clara.

---

## 7. Visualización y suavizado

### Observaciones de figuras 17×7 (tiempo + espectral)
- `RMS`: plana y sube al final en la mayoría.
- `Kurtosis` / `Skewness` / `CrestFactor`: mucho ruido y picos aislados.
- Espectrales: cambian antes que el RMS, con subidas/bajadas o escalones.
- Posible asentamiento inicial en `SpectralStd` / `SpectralCentroid` (primera media hora) — hipótesis a validar.
- Cambios de régimen notables en algunos bearings (p. ej. 1_3, 2_1, 2_2).

### Evolución del suavizado
| Versión | Config | Resultado |
|---------|--------|-----------|
| Solo Hampel | picos aislados | No alisa ruido continuo |
| Agresivo | mediana 10 % + media 5 % vida | **Deforma** las curvas |
| **Moderado (adoptado)** | Hampel w=11, k=4; mediana **2 %**; media **1 %** | Tendencia clara, sin aplastar forma |

Suavizado **por `bearing_id`**, no global. En gráficos: cruda en gris + suavizada en azul.

### Imágenes generadas (`analisis_features/suavizado_moderado/`)
1. Tiempo+espectral eje H (7 feats)  
2. Tiempo+espectral eje V (7 feats)  
3. WPD H 0–7  
4. WPD H 8–15  
5. WPD V 0–7  
6. WPD V 8–15  
7. Cruzadas H–V (`Pearson`, `EnergyRatio`)

### Dataset suavizado exportado
- `features_48_suavizado.parquet` / `.csv`  
- Mismas filas e IDs/tiempo; **no reduce ráfagas**; solo cambian valores de las 48 features.

---

## 8. Criterio de selección: replanteo

- La monotonía sola penaliza eventos tardíos, cambios de régimen y curvas no monótonas.
- Umbral duro de monotonía mediana 0,5 eliminaría muchas features que sí son útiles en algunos bearings.
- Las 20 eliminadas salieron por **redundancia**, no por mal score de monotonía.
- **Objetivo real:** utilidad para RUL, no solo “curva bonita monótona”.

### Caminos propuestos (pendientes de elegir)
1. Información mutua con RUL / fracción de vida  
2. Modelo simple 1-feature + validación leave-one-bearing-out  
3. Detección temprana (desvío respecto a estado sano vs falsas alarmas)  
4. Ablación con el modelo final (p. ej. TCN + Transformer)

### Etiqueta RUL (pendiente)
Los 17 son run-to-failure; opciones: RUL lineal en tiempo, fracción de vida [0,1], o RUL por tramos. Sugerencia: construir las dos primeras para rankear features.

---

## 9. Estado de archivos clave

| Archivo | Rol |
|---------|-----|
| `features_68.parquet` | Original completo |
| `features_48.parquet` / `.csv` | Tras quitar 20 redundantes |
| `features_48_suavizado.parquet` / `.csv` | 48 features suavizadas |
| `metricas_hi_por_rodamiento.csv` | Métricas por feature × bearing |
| Scripts de correlación / suavizado / WPD | Análisis de la sesión |

---

## 10. Pendientes

1. Elegir método de evaluación feature ↔ RUL y definir etiqueta de RUL.  
2. Gráfico de barras de las 48 features por mediana Spearman (presentación).  
3. **mRMR** (o similar) sobre las 48 → cesta final ~12–15 no redundantes.  
4. HI por distancia de Mahalanobis con la cesta optimizada (referencia sana a definir, p. ej. primeras 70 ráfagas).  
5. Verificar Σ Erel ≈ 1 y orden frecuencial de nodos WPD.  
6. Si hace falta dirección (sube/baja), recalcular Spearman/Trend **con signo**.  
7. Valorar recorte del transitorio inicial y análisis en vida normalizada [0,1].

---

## 11. Línea de trabajo del proyecto (contexto amplio)

- Subsets pequeños (`features_6`) y descomposición de aportes de MD².  
- Arquitectura de redes y despliegue en Raspberry Pi 4B (validación real posterior).  
- Objetivo de cadena: **features depuradas → HI/MD → modelo RUL → edge**.