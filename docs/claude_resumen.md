# Resumen de la sesión — 17 de octubre de 2026

Trabajo sobre las features de vibración de FEMTO/PRONOSTIA (17 rodamientos, canales H y V): qué significan los nodos WPD, cómo evaluar las features, qué se eliminó y qué falta definir.

---

## 1. Consultas generales

- **NotebookLM (Gemini):** usa modelos Gemini, pero es un asistente de recuperación sobre las fuentes subidas (tipo RAG), no un agente. Sirve como pieza "lector" de un sistema multiagente, pero no tiene API abierta, así que se usaría como herramienta aparte y no integrada en CrewAI.
- **API:** forma estandarizada de que un programa le pida cosas a otro. En CrewAI cada agente llama al LLM a través de la API con una clave que cobra por uso.
- **Archivos borrados que estaban en `.gitignore`:** nunca estuvieron en GitHub, así que no se recuperan desde ahí. Opciones: copia en el computador del laboratorio, papelera, OneDrive (papelera de 30 días y "Restaurar OneDrive"), versiones anteriores de Windows y, como último recurso, Recuva. Los raw de FEMTO son públicos y se pueden volver a descargar. Recomendación: respaldarlos en OneDrive, fuera del repositorio.

---

## 2. Energía de los nodos WPD: Erel vs Erel × RMS

- **WPD_Erel_i:** fracción de la energía total que cae en el nodo i (banda de frecuencia). Va de 0 a 1 y los 16 nodos de un canal suman 1. Con 16 nodos, la descomposición es de nivel 4.
- **Erel × RMS** (la "energía real" de los gráficos): la suma de los 16 nodos da el RMS, siempre que ΣErel = 1 (el script no lo verifica). No es energía ni RMS por banda, sino un reparto proporcional del RMS.
  - Alternativas físicas: Erel × RMS² (suma RMS²) y RMS × √Erel (se combinan en cuadratura).
- **Comparación (Bearing1_1, canal H):** no son la misma curva a otra escala. Erel × RMS se parece al RMS (en los nodos 3 a 11 solo muestra el pico final), mientras que la Erel muestra cómo se redistribuye la energía entre bandas desde etapas tempranas.
- **Decisión: quedarse con la Erel.**
- **Observación:** ningún nodo de Erel se comporta igual en los 17 rodamientos.

---

## 3. Métricas de calidad (`metricas_hi_por_rodamiento.csv`)

68 features × 17 rodamientos.

**Hallazgos**
- Solo 3 features superan 0,7 de monotonía mediana: `SpectralStd_H` (0,83), `WPD_Erel_6_H` (0,76) y `WPD_Erel_14_H` (0,73). Ni siquiera la mejor funciona en todos: `SpectralStd_H` cae a 0,11 en Bearing1_3.
- Por familia, WPD (Erel) es la mejor y las features de tiempo las peores.
- La robustez es alta en casi todas (el problema no es el ruido) y la prognosabilidad es baja en general (mediana 0,19).
- Rodamientos más difíciles: Bearing3_2 y Bearing1_2. El más fácil: Bearing1_7.

**Columnas eliminadas (9)**
| Columna | Motivo |
|---|---|
| `Rob_alpha_0.05`, `0.20`, `0.30`, `0.50` | Casi idénticas a las otras (correlación 0,996 entre 0,1 y 0,2). Se deja `Rob_alpha_0.10`. |
| `Rob_range_alpha` | Depende de los cinco alfa. |
| `Mon_sign` | No discrimina (máximo 0,09, correlación 0,02 con `Mon_spearman`). |
| `initial_window_mean`, `final_window_mean` | Escalas distintas entre features y dirección poco confiable si la feature sube y baja. |
| `n_observations` | Constante por rodamiento. |

**Tabla resultante:** `feature`, `bearing_id`, `Mon_spearman`, `Trend_pearson`, `Rob_alpha_0.10`, `Prognosability`.

**Notas:** `Mon_spearman` y `Trend_pearson` parecen venir en valor absoluto (no hay negativos), así que no indican si la feature sube o baja. `Prognosability` tiene un solo valor por feature.

---

## 4. Redundancia entre features

Script `spearman_correlation.py`: |Spearman| mayor o igual al umbral en al menos el 80% de los rodamientos, con las curvas completas dentro de cada rodamiento.

**Ronda 1 (umbral 0,9, 20 parejas):** se conservó una por grupo.
- RMS (eliminadas `Std`, `Variance`, `MAV`)
- CrestFactor (eliminadas `ImpulseFactor`, `MarginFactor`)
- SpectralEntropy (eliminada `SpectralFlatness`)

**Ronda 2 (umbral 0,7, 38 parejas, 28 features involucradas):**
- Se eliminan `SpectralRMS`, `Peak`, `P2P` y `ShapeFactor`; se conserva `Kurtosis`.

**Las 20 features eliminadas** (cada una en H y V): `Std`, `Variance`, `MAV`, `ImpulseFactor`, `MarginFactor`, `SpectralFlatness`, `SpectralRMS`, `Peak`, `P2P`, `ShapeFactor`.

**Quedan 48 features:**
- `RMS`, `CrestFactor`, `SpectralEntropy`, `Kurtosis` (8)
- `Skewness`, `SpectralCentroid`, `SpectralStd` (6)
- Los 32 nodos `WPD_Erel`
- `Pearson_HV` y `EnergyRatio_HV`

**`features_48.parquet`:** verificado. 24.889 filas, 17 rodamientos, 52 columnas (48 features + `bearing_id`, `burst_id`, `burst_index`, `time_seconds`), sin NaN ni infinitos. No se pudo comparar con el original los valores, solo la estructura.

**Observación:** 40 features no tienen ninguna pareja redundante a 0,7 (los 32 nodos de Erel, `SpectralCentroid`, `SpectralStd`, `Skewness` y las cruzadas H-V). La correlación entre parejas ya dio casi todo lo que podía.

---

## 5. Replanteamiento del criterio de selección

- La monotonía mide solo una tendencia sostenida. Deja mal parados los eventos tardíos, los cambios de régimen y las curvas que suben y bajan.
- Con un umbral de monotonía mediana de 0,5 se perderían 24 de las 48 features, y 16 features con mediana menor a 0,5 superan 0,8 en al menos un rodamiento.
- **Ninguna de las 20 eliminadas salió por esas métricas**, solo por redundancia.
- **Objetivo:** el comportamiento que lleva al RUL. La monotonía pasa a ser un criterio más.

**Formas propuestas de evaluar cada feature contra el RUL** (sin elegir aún):
1. Información mutua con el RUL o la fracción de vida.
2. Predicción simple por feature con validación dejando un rodamiento fuera.
3. Detección temprana (cuánto antes de la falla se desvía de su comportamiento sano, y con cuántas falsas alarmas).
4. Ablación con el modelo real (TCN + Transformer).

**Etiqueta de RUL:** los 17 rodamientos son los originales run-to-failure, y el RUL aún no está calculado. Opciones: RUL lineal en tiempo, fracción de vida (0 a 1) y RUL por tramos. Sugerencia: construir las dos primeras para evaluar features y decidir la tercera más adelante (**pendiente de tu confirmación**).

**Observaciones de las figuras 17 × 7** (canales H y V):
- `RMS` es plana y sube de golpe al final en la mayoría de los rodamientos.
- `Kurtosis`, `Skewness` y `CrestFactor` son casi todo ruido y picos aislados.
- Las espectrales cambian antes que el RMS, pero con subidas y bajadas o escalones.
- Hipótesis (sin confirmar): `SpectralStd` y `SpectralCentroid` en H suben fuerte en la primera media hora en casi todos los rodamientos, lo que podría ser un periodo de asentamiento y no desgaste.
- Bearing2_1 y 2_2 tienen un salto del centroide cerca de 0,3 a 0,5 h.
- Bearing1_3 en V cambia de régimen hacia las 4,5 h en varias features a la vez. Corrección: ese eje era tiempo en segundos y no en muestras.

---

## 6. Suavizado para visualizar

- `hampel_features.py`: filtro de Hampel. Solo quita picos aislados y no suaviza el ruido continuo.
- `suavizado_agresivo.py`: Hampel + mediana móvil (10% de la vida) + media móvil (5%). Dio líneas continuas, pero **deformó demasiado las formas**.
- **Siguiente prueba:** bajar `MEDIAN_FRAC` a 0,03 y `MEAN_FRAC` a 0,015, o usar Hampel + Savitzky-Golay.

---

## 7. Archivos de la sesión

| Archivo | Para qué |
|---|---|
| `wavelets_real_17.py` | 8 imágenes (17 filas × 4 nodos) de Erel × RMS |
| `wavelets_relativa_17.py` | 8 imágenes (17 filas × 4 nodos) de Erel |
| `comparar_erel.py` | Comparación Erel vs Erel × RMS en Bearing1_1, canal H |
| `comparacion_erel_vs_real.py` | Versión con tabla, mapas de calor y gráficos (más de lo pedido) |
| `spearman_correlation.py` | Parejas redundantes (tuyo) |
| `hampel_features.py` | Gráficos 17 × 7 con filtro de Hampel |
| `suavizado_agresivo.py` | Gráficos 17 × 7 con suavizado agresivo |
| `features_48.parquet` | Datos con las 48 features |

---

## 8. Pendientes

1. Elegir el método para evaluar features contra el RUL (puntos 1 a 4 de la sección 5) y definir la etiqueta de RUL.
2. Ajustar el suavizado (0,03 / 0,015 o Savitzky-Golay).
3. Propuestas no ejecutadas: nodos de Erel como datos composicionales (log-ratios), PCA, análisis por condición de operación y en escala de vida de 0 a 1, y recorte del transitorio inicial.
4. Verificar que ΣErel = 1 y si los nodos están ordenados por frecuencia.
5. Si hace falta la dirección de cada feature (sube o baja), recalcularla con signo desde las features originales.