# Resumen de avance y caminos posibles — 18 de octubre de 2026

Trabajo sobre las features de vibración de FEMTO/PRONOSTIA (17 rodamientos, canales H y V): de 68 features a 17 candidatas, qué criterio se usó y qué falta decidir.

---

## 1. Qué hemos hecho

### 1.1 De 68 a 48 features (sesión del 17 de octubre)
- **Erel** (energía relativa por nodo WPD) como representación de los nodos, en vez de Erel × RMS.
- **Columnas de métricas eliminadas** (`metricas_hi_por_rodamiento.csv`): alfas de robustez (se deja `Rob_alpha_0.10`), `Rob_range_alpha`, `Mon_sign`, medias de ventana inicial y final, `n_observations`.
- **Redundancia** (Spearman, 80% de los rodamientos): umbral 0,9 y luego 0,7. Salieron 20 features (10 por canal): `Std`, `Variance`, `MAV`, `ImpulseFactor`, `MarginFactor`, `Peak`, `P2P`, `ShapeFactor`, `SpectralFlatness` y `SpectralRMS`.
- **Resultado:** `features_48.parquet` (24.889 filas, 17 rodamientos).
- **Replanteamiento:** la monotonía es solo un criterio más. El objetivo es el comportamiento que lleva al RUL, y filtrar por monotonía descartaría features útiles en algunos rodamientos.

### 1.2 Suavizado y vista colectiva
- Suavizado moderado (según tu script: Hampel ventana 11 con k=4, mediana 2% y media 1% de la vida), guardado en `features_48_suavizado.parquet`. Las pruebas agresivas deformaban las formas.
- **Gráficos colectivos** (`promedio_total_suavizado.py`): cada rodamiento con Min-Max propio, remuestreado a 101 puntos de vida útil (0 a 100%). Línea = mediana de los 17 y sombra = percentiles 25 a 75.
- **Lo que mostraron:**
  - Los nodos Erel 3 a 9 bajan y los 10 a 15 suben. Varios invierten al final.
  - RMS, Kurtosis y CrestFactor despegan solo sobre el 90%.
  - `SpectralStd_H` sube de forma continua y consistente.
  - `EnergyRatio_HV` cae en el primer 15% (posible asentamiento).

### 1.3 Criterio nuevo: área de la banda P25–P75
- **Script 1** (`areas_ordenadas.py`): área de la sombra de cada feature, normalizada entre 0 y 1. Genera `areas_ordenadas.csv` y una imagen de 8 × 6 ordenada de menor a mayor.
- **Script 2** (`percentiles_areas.py`): percentiles de las 48 áreas, percentil y cuartil por feature, e histograma. Las áreas suben de forma continua de 0,10 a 0,50, sin saltos naturales (salvo uno tras las 5 primeras).
- **Qué mide el área:** cuánto coinciden los 17 rodamientos. No mide si la feature sirve. La utilidad se decide después contra el RUL.
- **Paralelismo de los bordes P25/P75:** lo descartaste por ahora (da más problemas que ayuda).

### 1.4 Corte y resultado
- **Corte en área 0,31:** se conservan las features con área ≤ 0,31. Se subió desde 0,299 para no perder `Erel_4_H` (0,307).
- **Resultado: 17 de 48 features** (un poco más de un tercio).

| Familia | 68 originales | 48 | 17 |
|---|---|---|---|
| Time | 24 | 8 | 7 |
| Spectral | 10 | 6 | 1 |
| Wavelets | 32 | 32 | 9 |
| HV | 2 | 2 | 0 |
| **Total** | **68** | **48** | **17** |

Familias asignadas por nombre de feature (`Std`, `Variance`, `MAV`, `Peak`, `P2P`, `ShapeFactor`, `ImpulseFactor`, `MarginFactor` como Time; `SpectralFlatness` y `SpectralRMS` como Spectral).

**Las 17, en pares H–V** (ordenadas por el promedio de las dos áreas):

| # | Feature | Área H | Área V | Promedio | Quedan |
|---|---|---|---|---|---|
| 1 | RMS | 0,100 | 0,109 | 0,104 | H y V |
| 2 | Kurtosis | 0,143 | 0,153 | 0,148 | H y V |
| 3 | CrestFactor | 0,214 | 0,156 | 0,185 | H y V |
| 4 | Erel_3 | 0,263 | 0,243 | 0,253 | H y V |
| 5 | Erel_5 | 0,224 | 0,295 | 0,260 | H y V |
| 6 | Erel_4 | 0,307 | 0,268 | 0,287 | H y V |
| 7 | Erel_6 | 0,263 | 0,346 | 0,305 | solo H |
| 8 | SpectralStd | 0,186 | 0,455 | 0,321 | solo H |
| 9 | Skewness | 0,274 | 0,416 | 0,345 | solo H |
| 10 | Erel_12 | 0,490 | 0,246 | 0,368 | solo V |
| 11 | Erel_11 | 0,458 | 0,299 | 0,378 | solo V |

### 1.5 Comparación con la tabla original de 6 features
La tabla original eran `SpectralStd` y `Erel` 2, 4, 5, 6 y 7, todas en H. Con el corte en 0,31 entran 4 de 6 (`SpectralStd_H`, `Erel_4_H`, `Erel_5_H`, `Erel_6_H`). Quedan fuera `Erel_2_H` (0,474) y `Erel_7_H` (0,424), con bandas anchas.

### 1.6 Observaciones sobre ejes
No hay un eje claramente mejor: por área, H es más angosto en 10 pares y V en 12 (promedios 0,344 y 0,336). En las temporales y espectrales gana H en `SpectralStd`, `Skewness`, `RMS` y `Kurtosis`, y V en `CrestFactor` y `SpectralCentroid`.

---

## 2. Dónde estamos

- Tenemos **17 candidatas** de las 48, elegidas por consistencia entre rodamientos, no por utilidad para el RUL.
- Familias que desaparecen del conjunto: **HV** completa, y en Spectral solo sobrevive `SpectralStd_H`.
- No se han evaluado todavía contra el RUL, y la etiqueta de RUL aún no está definida.
- **Riesgo:** el área puede estar dejando fuera features útiles en algunos rodamientos (el bloque de nodos Erel 10 a 15, `SpectralCentroid`, `Erel_2_H` y `Erel_7_H`).

---

## 3. Caminos posibles

1. **Definir el RUL y evaluar las features contra él** (el paso que falta para saber qué sirve).
   - Etiquetas: RUL lineal en tiempo y fracción de vida (0 a 1); RUL por tramos más adelante.
   - Métodos: información mutua, predicción simple por feature dejando un rodamiento fuera, detección temprana (cuánto antes de la falla se desvía del comportamiento sano) o ablación con el modelo real (TCN + Transformer).
2. **Comprobar si el corte de área perdió algo útil:** evaluar las 31 descartadas con el mismo método del camino 1 y ver cuáles se recuperan.
3. **Recortar el transitorio inicial** (primer 10 a 15%) y recalcular las áreas. Afecta a `Erel_7_H`, `8_H`, `8_V` y `9_H`, que parten con banda muy ancha.
4. **Analizar por condición de operación** (grupos 1, 2 y 3 de FEMTO). Algunas bandas anchas pueden deberse a que mezclan condiciones.
5. **Criterio H–V:** decidir qué hacer con los pares donde los ejes se comportan distinto (descartar el par o quedarse con el mejor eje). Hoy 6 features pasan en un solo eje.
6. **Nodos Erel como datos composicionales** (log-ratios) o PCA, para reducir los 32 nodos.
7. **Verificaciones pendientes:** que ΣErel = 1, y si los nodos están ordenados por frecuencia.

**Sugerencia de orden:** 1, luego 2, y dejar 3 a 6 como refinamientos según lo que muestre la evaluación.

---

## 4. Archivos

| Archivo | Para qué |
|---|---|
| `promedio_total_suavizado.py` | Gráficos colectivos (mediana y P25–P75 por vida útil) |
| `areas_ordenadas.py` | Área de banda por feature, CSV e imagen ordenada |
| `percentiles_areas.py` | Percentiles de las 48 áreas e histograma |
| `features_48_suavizado.parquet` | Datos suavizados con las 48 features |