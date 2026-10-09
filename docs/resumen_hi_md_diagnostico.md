# Resumen: de las 6 features al diagnóstico de la distancia de Mahalanobis (MD)

Documento de trabajo. Solo PRONOSTIA (17 rodamientos). Describe lo que se hizo desde que quedaron seleccionadas las 6 features, qué se observó y por qué se hizo el análisis de la MD.

---

## 1. Punto de partida: las 6 features

De las 68 features calculadas por ráfaga (33 por canal más 2 cruzadas), el filtro de sensibilidad (4 métricas + EWM) y el mRMR dejaron **K = 6**:

| # | Feature | Canal |
|---|---|---|
| 1 | `SpectralStd_H` | H |
| 2 | `WPD_Erel_5_H` | H |
| 3 | `SpectralCentroid_V` | V |
| 4 | `WPD_Erel_8_H` | H |
| 5 | `WPD_Erel_6_H` | H |
| 6 | `WPD_Erel_7_H` | H |

- Cinco de las seis son del canal H; cuatro son energías relativas de wavelet (nodos 5 a 8).
- Archivo: `rul_ina\data\processed\features_6.parquet` (y `.csv`), generado por `extract_features_6.py`.
- Contenido: 24,889 filas (una por ráfaga) y 10 columnas: `bearing_id`, `burst_id`, `burst_index`, `time_seconds` y las 6 features.

Cada ráfaga original (2,560 puntos de vibración) queda resumida en **una fila de 6 números**.

---

## 2. Cálculo del Indicador de Salud (HI)

Script: `hi_rul.py`. Se aplica **por rodamiento**, con estos pasos:

1. **Referencia sana:** las primeras 70 ráfagas del rodamiento (tabla de 70 filas × 6 columnas). De ahí se calcula el promedio de cada feature (`mu_sano`) y la covarianza (6×6, estimada con Ledoit-Wolf).
2. **Distancia de Mahalanobis (MD)** para **cada** ráfaga del rodamiento: qué tan lejos está del promedio sano, usando la covarianza como regla. MD = 0 significa igual al estado sano; mientras mayor, más lejos.
3. **τ** = MD máxima dentro de las 70 ráfagas sanas.
4. **HI = exp(−MD / τ).** HI = 1 es sano; HI cerca de 0 es dañado.
5. **RUL real** = número de ráfagas restantes hasta el final de la vida del rodamiento. Es la etiqueta de entrenamiento; **no se calcula a partir del HI**.

Salida: `rul_ina\hi_rul\dataset_hi_rul_features.parquet` (y `.csv`).

---

## 3. Qué se observó en las curvas del HI

Grilla de los 17 rodamientos con el promedio global (`grilla_hi_17_rodamientos_con_promedio.png`, K = 6):

- Curvas **muy ruidosas** y no monótonas, con picos (Bearing1_6, Bearing1_4, Bearing2_7).
- Muchos rodamientos **parten en ~0.4 a 0.6 con saltos**, en vez de partir cerca de 1 y estables.
- Varios **caen muy pronto**: Bearing3_2 y Bearing2_5 colapsan antes del 10 % de vida; Bearing1_1, 1_4, 1_6, 1_7, 2_4 y 3_1 bajan de forma continua desde el inicio.
- **Niveles finales muy distintos** entre rodamientos (Bearing1_2 termina en ~0.3; Bearing2_1 y 2_5 se estabilizan y suben al final; Bearing1_3 llega a 0 al ~68 %).
- El **promedio global sí baja** de forma clara, pero las curvas individuales son poco comparables.

---

## 4. Por qué se hizo el análisis de la MD

Se quería usar el HI para el paso siguiente (suavizado y detección del inicio de degradación con PELT, y luego el RUL). Pero con curvas así, cualquier detección de cambio iba a marcar la caída temprana aunque no fuera degradación real.

Había dos explicaciones posibles para el comportamiento del HI:

- **A. Problema en la referencia sana:** las 70 primeras ráfagas no representan un estado estable (por ejemplo, un transitorio de arranque).
- **B. Problema en la conversión de MD a HI:** la MD sí aporta información, pero `exp(−MD/τ)` la aplasta.

Para distinguirlas se graficó la **MD cruda**, antes de la exponencial. Si la MD ya fuera inestable dentro de las 70 ráfagas, el problema sería A; si fuera estable ahí y creciera con el desgaste, sería B.

---

## 5. El diagnóstico

Script: `diagnostico_md.py`. Replica exactamente el cálculo de `hi_rul.py` (misma referencia sana, mismo Ledoit-Wolf sobre covarianza cruda) y grafica la MD de 4 rodamientos elegidos por sus comportamientos distintos, con una línea en la ráfaga 70 y otra en τ.

Salida: `rul_ina\hi_rul\diagnostico_md_4_rodamientos.png`.

### Resultados

| Rodamiento | Ráfagas | τ | Comportamiento de la MD |
|---|---|---|---|
| Bearing3_2 | 1,637 | 2.64 | Baja y plana en el tramo sano (~1 a 2); salta a ~10 poco después de la ráfaga 70; sube lento hasta ~14; salta a ~22 al ~87 % de la vida. |
| Bearing2_5 | 2,311 | 4.35 | Baja al inicio; sube fuerte hasta ~11 al ~12 %; luego se estanca en ~8 y baja suave a ~7. No crece con la degradación. |
| Bearing1_1 | 2,803 | 3.22 | Sube gradual hasta ~13 al ~60 %; baja a ~5 al ~78 %; vuelve a subir. Pico de ~45 al ~47 %. |
| Bearing2_7 | 230 | 3.28 | Baja hasta ~60 % de la vida (las 70 ráfagas son el 30 %); después sube con picos hasta ~16. Es el único con el comportamiento esperado. |

### Conclusiones

1. **La referencia sana funciona** (hipótesis A descartada en lo que muestra este gráfico): en los cuatro rodamientos la MD es baja y estable dentro de las 70 ráfagas.
2. **La MD sí trae información y el HI la pierde** (hipótesis B): en 3_2 y 1_1 la MD sigue creciendo, pero con τ entre 2.6 y 4.4 una MD de 8 o más deja el HI cerca de 0.
3. **En Bearing2_5 las 6 features parecen saturar:** la MD se estanca y no distingue etapas avanzadas.
4. **Los niveles de MD no son comparables entre rodamientos** (~8 en 2_5, ~14 en 3_2, ~12 en 1_1), porque cada uno usa su propia referencia sana.
5. **Queda sin explicar** el salto temprano justo después de la ráfaga 70 en Bearing3_2 y 2_5: puede ser degradación temprana o un transitorio. Para decidirlo hay que ver las features crudas en esa zona.

---

## 6. Diferencias entre `hi_rul.py` y lo decidido antes

| Punto | Decisión previa | Lo que hace `hi_rul.py` |
|---|---|---|
| τ | Múltiplo ajustable del MD promedio del tramo sano (ejemplo 3) | MD **máxima** del tramo sano |
| Covarianza | Ledoit-Wolf sobre la matriz de correlación, estandarizando con el tramo sano (33.6) | Ledoit-Wolf sobre la covarianza **cruda**, sin estandarizar |
| NaN / infinitos | Interpolación lineal entre ráfagas vecinas (33.5) | Se reemplazan por 0.0 |

---

## 7. Decisión pendiente

Cómo convertir la MD en un indicador que no se aplaste:

1. **Usar la MD directamente (o log MD).** Simple y sin pérdida de información, pero no queda entre 0 y 1.
2. **Mantener `exp(−MD/τ)` con un τ mucho mayor** (múltiplo grande de la MD sana). Mantiene el rango 0 a 1, pero hay que afinar el múltiplo.
3. **Otra transformación acotada,** como `1 / (1 + MD/τ)`, que baja más lento que la exponencial.

Además, queda por decidir si se alinea el cálculo con lo decidido antes (τ, estandarización, imputación) y revisar las features crudas en el salto temprano de 3_2 y 2_5.

---

## 8. Archivos relacionados

- `extract_features_6.py` → `data\processed\features_6.parquet` / `.csv`
- `hi_rul.py` → `hi_rul\dataset_hi_rul_features.parquet` / `.csv`
- `plot_hi_17.py` → `grilla_hi_17_rodamientos_con_promedio.png`
- `diagnostico_md.py` → `hi_rul\diagnostico_md_4_rodamientos.png`
