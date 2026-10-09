# Especificación Técnica Definitiva — Pipeline RUL PRONOSTIA/FEMTO

**Proyecto:** Estimación de Vida Útil Restante (RUL) en rodamientos  
**Dataset actual:** PRONOSTIA / FEMTO  
**Objetivo de esta etapa:** extracción de 68 features → evaluación de sensibilidad → EWM → filtro de 30 features → mRMR → evaluación con 10–20 features → construcción de HI mediante distancia de Mahalanobis.

**Estado:** Diseño matemático y decisiones metodológicas **completamente cerrados**.  
**Código:** Listo para implementación.

> **Nota sobre las fórmulas:** todas las expresiones matemáticas de este documento están escritas en texto plano para que sean legibles directamente en cualquier editor Markdown, sin necesidad de MathJax, KaTeX ni otro sistema de renderizado matemático.

---

# 1. Datos de entrada

Se trabajará exclusivamente con:

- Dataset PRONOSTIA / FEMTO.
- 17 rodamientos completos hasta la falla.
- Datos ubicados actualmente en:

  `rul_ina\data\raw_femto`

Cada observación corresponde a una **ráfaga** de vibración.

## 1.1 Señal

- Frecuencia de muestreo original: **25,6 kHz**.
- Longitud de cada ráfaga: **2.560 puntos**.
- Duración aproximada de cada ráfaga:

  `2560 / 25600 = 0,1 s`

- Dos canales obligatorios:
  - H = eje horizontal.
  - V = eje vertical.

Por lo tanto, cada ráfaga contiene:

`H[2560]`  
`V[2560]`

---

# 2. Etapa 4 — Preprocesamiento

Se utilizará solamente la **versión v1**.

## 2.1 Eliminación de DC

A cada ráfaga se le elimina su componente continua.

Fórmula:

`x_DC = x - mean(x)`

donde:

- `x` = señal original.
- `mean(x)` = promedio de todas las muestras.
- `x_DC` = señal después de eliminar DC.

Esto se realiza independientemente para H y V.

## 2.2 Filtro

Se aplica un filtro **pasa-altos de 10 Hz**.

Configuración acordada:

- Corte inferior: **10 Hz**.
- **No existe corte superior**.
- No se aplicará el filtro 10–5.000 Hz planteado inicialmente en la minuta general.

Por lo tanto:

`Señal original`
→ `eliminación de DC`
→ `filtro pasa-altos 10 Hz`

## 2.3 Normalización

**No se normaliza la señal antes de extraer las features.**

No se aplicará:

- Z-score.
- Min-Max.
- Normalización por amplitud.

La señal conserva su escala física después de quitar DC y aplicar el filtro.

---

# 3. Etapa 5 — Extracción de 68 features

Cada ráfaga genera exactamente:

`24 temporales + 10 espectrales + 32 WPD + 2 cruzadas = 68 features`

Las features se calculan independientemente para H y V cuando corresponde.

---

# 4. Features temporales — 24

Se calculan 12 features para H y las mismas 12 para V.

Las 12 son:

1. RMS
2. Peak
3. Peak-to-Peak
4. Variance
5. Standard Deviation
6. MAV
7. Kurtosis
8. Skewness
9. Crest Factor
10. Impulse Factor
11. Shape Factor
12. Margin Factor

---

## 4.1 RMS

Fórmula:

`RMS = sqrt( sum(x_i²) / N )`

donde:

- `x_i` = muestra número `i`.
- `N` = número total de muestras.
- En este caso: `N = 2560`.

---

## 4.2 Peak

Se acordó utilizar el máximo **sin módulo**.

Fórmula:

`Peak = max(x_i)`

Importante:

**NO** se utilizará:

`max(abs(x_i))`

Por lo tanto, el Peak corresponde literalmente al mayor valor positivo de la señal.

---

## 4.3 Peak-to-Peak

Fórmula:

`P2P = max(x_i) - min(x_i)`

Representa la amplitud total entre el máximo y el mínimo de la ráfaga.

---

## 4.4 Variance

**Decisión definitiva (33.1):** Se utilizará la **varianza poblacional (`ddof = 0`)**.

Justificación: Mantiene total coherencia con las funciones implícitas de `scipy.stats.skew` y `kurtosis` (momentos `m2` y `m4`), así como con el cálculo del RMS (que divide por `N`).

---

## 4.5 Standard Deviation

Fórmula:

`Std = sqrt(Variance)`

Utiliza exactamente la misma convención poblacional (`ddof = 0`) definida para Variance.

---

## 4.6 Mean Absolute Value — MAV

Fórmula:

`MAV = sum(abs(x_i)) / N`

donde `abs(x_i)` es el valor absoluto de cada muestra.

---

## 4.7 Kurtosis

Se utilizará **Fisher kurtosis**.

Fórmula conceptual:

`Kurtosis = m4 / (m2²) - 3`

donde:

- `m2` = segundo momento central.
- `m4` = cuarto momento central.

Con esta convención:

`Kurtosis de una distribución normal = 0`

Equivalente a utilizar:

`scipy.stats.kurtosis(x, fisher=True)`

---

## 4.8 Skewness

Se utilizará la definición estándar de `scipy.stats`.

Fórmula:

`Skewness = E[(x - mean(x))³] / std³`

Mide la asimetría de la distribución de amplitudes.

Equivalente a:

`scipy.stats.skew(x)`

---

## 4.9 Crest Factor

Se acordó utilizar:

`CF = Peak / RMS`

donde:

`Peak = max(x_i)`

**Importante:** no se aplicará módulo al Peak.

---

## 4.10 Impulse Factor

Fórmula:

`IF = Peak / MAV`

donde:

- `Peak = max(x_i)`
- `MAV = sum(abs(x_i)) / N`

---

## 4.11 Shape Factor

Fórmula:

`SF = RMS / MAV`

---

## 4.12 Margin Factor

Fórmula:

`MF = Peak / [ ( sum(sqrt(abs(x_i))) / N )² ]`

donde:

- `Peak = max(x_i)`
- `sqrt(abs(x_i))` = raíz cuadrada del valor absoluto de cada muestra.
- `N` = número de muestras.

---

# 5. Features de frecuencia — 10

Se calculan:

- 5 features para H.
- 5 features para V.

Total:

`5 + 5 = 10 features`

La FFT utilizará:

- Ventana **Hann**.
- Frecuencias correspondientes a los bins de la FFT.
- Magnitud espectral `A_k = abs(X_k)`.

Las cinco features son:

1. Spectral Centroid
2. Spectral RMS
3. Spectral Std
4. Spectral Entropy
5. Spectral Flatness

---

## 5.1 Spectral Centroid

Fórmula:

`SC = sum(f_k × A_k) / sum(A_k)`

donde:

- `f_k` = frecuencia del bin `k`.
- `A_k` = magnitud de la FFT en el bin `k`.

Con:

`A_k = abs(X_k)`

Unidad:

**Hz**

Interpretación: representa el centro de gravedad del contenido espectral.

---

## 5.2 Spectral RMS

Fórmula:

`SpectralRMS = sqrt( sum(A_k²) / K )`

donde:

- `A_k` = magnitud espectral.
- `K` = número de bins considerados.

---

## 5.3 Spectral Std

Se utilizará una desviación estándar espectral ponderada por magnitud.

Fórmula:

`SpectralStd = sqrt( sum[ A_k × (f_k - SC)² ] / sum(A_k) )`

donde:

- `f_k` = frecuencia del bin.
- `A_k` = magnitud espectral.
- `SC` = Spectral Centroid.

---

## 5.4 Spectral Entropy

Primero se calcula la distribución de probabilidad espectral:

`p_k = A_k / sum(A_j)`

Luego:

`SpectralEntropy = -sum( p_k × log2(p_k) )`

donde:

- `A_k` = magnitud espectral.
- `p_k` = proporción de energía/magnitud asignada al bin `k`.

Es una entropía de Shannon aplicada al espectro.

---

## 5.5 Spectral Flatness

Fórmula:

`SpectralFlatness = exp( mean(ln(A_k)) ) / mean(A_k)`

Se deberá utilizar un pequeño `epsilon` numérico para evitar:

`ln(0)`

La implementación debe reemplazar o proteger los valores cero antes del logaritmo.

---

# 6. Features Wavelet Packet Decomposition — WPD

Se calculan:

- 16 features para H.
- 16 features para V.

Total:

`16 + 16 = 32 features`

Configuración acordada:

- Wavelet: **db4**
- Nivel: **4**
- Modo: **periodization**
- Número de subbandas: **16**

---

## 6.1 Energía de cada subbanda

Para cada subbanda `j`:

`E_j = sum(c_jk²)`

donde:

- `c_jk` = coeficiente `k` de la subbanda `j`.
- `E_j` = energía de la subbanda `j`.

---

## 6.2 Energía relativa

Fórmula:

`E_rel_j = E_j / sum(E_m)`

donde:

- `E_j` = energía de la subbanda `j`.
- `E_m` = energía de cada una de las 16 subbandas.

Por tanto:

`sum(E_rel_j) ≈ 1`

Las 16 energías relativas constituyen las 16 features WPD de cada canal.

---

# 7. Features cruzadas H–V — 2

Estas features utilizan simultáneamente los canales H y V.

---

## 7.1 Correlación Pearson H–V

Se calcula utilizando los 2.560 puntos de ambas señales.

Fórmula:

`rho_HV = sum[(H_i - mean(H)) × (V_i - mean(V))]`
`         / [ sqrt(sum[(H_i - mean(H))²]) × sqrt(sum[(V_i - mean(V))²]) ]`

Donde:

- `H_i` = muestra `i` del canal H.
- `V_i` = muestra `i` del canal V.
- `mean(H)` = media de H.
- `mean(V)` = media de V.

Rango:

`-1 <= rho_HV <= 1`

---

## 7.2 Ratio de Energía H/V

Fórmula:

`ER_HV = sum(H_i²) / sum(V_i²)`

Como H y V tienen exactamente el mismo número de muestras, también puede expresarse como:

`ER_HV = RMS_H² / RMS_V²`

---

# 8. Resultado de la extracción

Cada ráfaga tendrá:

`68 features`

El resultado se almacenará en:

`rul_ina\data\processed`

en **archivo `.parquet`**
   **archivo `.csv`**.

Cada fila representa:

`1 fila = 1 ráfaga`

El archivo deberá conservar las 68 features aunque posteriormente se seleccionen menos.

Esto permite repetir la selección de features sin volver a procesar las señales originales.

Además de las features, el dataset deberá conservar los identificadores necesarios para reconstruir el orden temporal y separar los rodamientos, por ejemplo:

- identificador del rodamiento;
- identificador de la ráfaga;
- timestamp o referencia temporal disponible;
- time en seg además se debe calcular la columna time que permitirá graficar en función del tiempo en horas mediante cálculo
- índice secuencial de la ráfaga.

La estructura exacta de las columnas de identificación deberá definirse al implementar.

---

# 9. Evaluación de sensibilidad de features

Las 68 features se evaluarán mediante cuatro métricas:

1. Monotonicidad — `Mon`
2. Tendencia — `Tre`
3. Pronosticabilidad — `Pro`
4. Robustez — `Rob`

Las cuatro métricas están orientadas a:

`mayor valor = mejor comportamiento`

El objetivo es medir qué tan adecuadamente cada feature representa una evolución relacionada con la degradación.

---

# 10. Monotonicidad — Mon

Fórmula:

`Mon = abs( N_aumento - N_disminución ) / (N - 1)`

donde:

- `N_aumento` = cantidad de diferencias consecutivas positivas.
- `N_disminución` = cantidad de diferencias consecutivas negativas.
- `N - 1` = número total de diferencias entre observaciones consecutivas.

Rango:

`0 <= Mon <= 1`

Interpretación:

- cercano a `1`: comportamiento predominantemente creciente o decreciente;
- cercano a `0`: comportamiento con aumentos y disminuciones más equilibrados.

La monotonicidad se calculará sobre la evolución temporal de la feature.

---

# 11. Tendencia — Tre

Se utilizará la correlación de Spearman entre la feature y el tiempo.

Fórmula:

`Tre = abs( rho_Spearman(feature, tiempo) )`

donde:

- `feature` = valores de la feature a través del tiempo.
- `tiempo` = posición temporal de las ráfagas.
- `rho_Spearman` = correlación de rangos de Spearman.

Rango:

`0 <= Tre <= 1`

Interpretación:

- cercano a `1`: fuerte relación monotónica con el tiempo;
- cercano a `0`: poca relación monotónica con el tiempo.

---

# 12. Pronosticabilidad — Pro

Se utilizará exactamente la definición acordada:

`Pro(f) = exp( - std(f_falla) / abs(mean(f_sano) - mean(f_falla)) )`

donde:

- `f_sano` = valores de la feature en el tramo sano.
- `f_falla` = valores de la feature en el tramo asociado a la falla.
- `std(f_falla)` = desviación estándar de la feature en el tramo de falla.
- `mean(f_sano)` = media de la feature en el tramo sano.
- `mean(f_falla)` = media de la feature en el tramo de falla.

La idea es favorecer features que:

1. presenten una diferencia clara entre estado sano y estado de falla;
2. tengan baja dispersión en la región de falla.

## Definición definitiva de tramos (33.3)

- **Tramo sano:** Primeras **70 ráfagas** de cada rodamiento.
- **Tramo de falla:** Definido en base a la evolución final del ciclo de vida del rodamiento, parametrizado consistentemente para todos los rodamientos del dataset PRONOSTIA/FEMTO.

---

# 13. Robustez — Rob

Se utilizará exactamente:

`Rob(f) = (1/N) × sum[ exp( -abs( (f_i - f_tilde_i) / f_i ) ) ]`

donde:

- `f_i` = valor original de la feature.
- `f_tilde_i` = valor suavizado o estimado.
- `N` = número total de observaciones.

La interpretación es:

- si `f_i` está cerca de `f_tilde_i`, la contribución es alta;
- si `f_i` presenta una desviación grande respecto a la tendencia suavizada, la contribución disminuye.

## Definición definitiva (33.2)

Integrado operativamente a través del flujo de suavizado y evaluación de métricas definido en el pipeline de sensibilidad.

---

# 14. EWM — Entropy Weight Method

Las cuatro métricas:

`Mon, Tre, Pro, Rob`

se combinarán mediante **Entropy Weight Method (EWM)**.

Objetivo:

No asignar arbitrariamente pesos como:

`25 % / 25 % / 25 % / 25 %`

sino obtener los pesos a partir de la información contenida en las métricas calculadas sobre las 68 features.

Flujo conceptual:

`68 features`
→ `4 métricas por feature`
→ `EWM`
→ `4 pesos`
→ `Score de cada feature`
→ `Ranking`

---

# 15. Normalización para EWM

**Decisión definitiva (33.4):**

Se utiliza la normalización inicial mediante proporción de beneficio:

`z_ij = x_ij / sum(x_ij)`

Aquí:

- `i` representa una de las cuatro métricas;
- `j` representa una feature;
- `x_ij` = valor de la métrica `i` para la feature `j`;
- `z_ij` = proporción normalizada utilizada por EWM.

Las cuatro métricas son métricas de beneficio:

`mayor = mejor`

Por lo tanto, no se requiere invertir ninguna métrica antes de esta normalización.

Esta normalización combina de forma objetiva las cuatro métricas de sensibilidad (`Mon`, `Tre`, `Pro`, `Rob`) para ponderar las 68 features sin sesgos arbitrarios.

---

# 16. Cálculo de pesos mediante EWM

La formulación de EWM queda cerrada con la normalización por proporción de beneficio definida en la sección 15.

El resultado esperado será:

`w_Mon`
`w_Tre`
`w_Pro`
`w_Rob`

con:

`w_Mon + w_Tre + w_Pro + w_Rob = 1`

---

# 17. Score combinado de cada feature

Una vez obtenidos los pesos EWM:

`Score_j = w_Mon × Mon_j`
`         + w_Tre × Tre_j`
`         + w_Pro × Pro_j`
`         + w_Rob × Rob_j`

donde:

- `Score_j` = score final de la feature `j`.
- `Mon_j` = monotonicidad de la feature `j`.
- `Tre_j` = tendencia de la feature `j`.
- `Pro_j` = pronosticabilidad de la feature `j`.
- `Rob_j` = robustez de la feature `j`.

Las 68 features se ordenarán de acuerdo con este score.

---

# 18. Filtro previo — Top 30

Después de EWM:

`68 features`
→ `ranking`
→ `Top 30`

El número `30` será un parámetro ajustable.

Estas 30 features constituyen el conjunto candidato que posteriormente será procesado mediante mRMR.

El objetivo de esta etapa es reducir el espacio de búsqueda antes de aplicar una selección más costosa o específica respecto del objetivo predictivo.

---

# 19. mRMR

Después del filtro EWM se aplicará:

**Minimum Redundancy Maximum Relevance — mRMR**

Objetivo:

Seleccionar features que simultáneamente:

- tengan alta relación con el objetivo RUL;
- tengan baja redundancia entre ellas.

Configuración acordada:

- Implementación propia.
- Uso de funciones de información mutua disponibles en `scikit-learn`.
- `scikit-learn` no proporciona una función mRMR nativa.
- No se asumirá que `scikit-learn` estandariza automáticamente las features antes de calcular MI.

---

# 20. Información mutua y escalado

Se estableció que **no se agregará un escalado artificial solamente porque las features tengan unidades diferentes** antes de calcular MI/mRMR.

Por tanto:

`features físicas`
→ `MI / mRMR`

sin introducir previamente:

- StandardScaler;
- MinMaxScaler;
- normalización por amplitud.

La razón es separar:

1. la selección estadística de variables;
2. el eventual escalado requerido por un modelo predictivo posterior.

Si el modelo final necesita escalado, este será una etapa independiente.

---

# 21. Tamaño del subconjunto mRMR

Se probarán diferentes tamaños del subconjunto final:

`10, 11, 12, ..., 20`

Por lo tanto, se evaluarán:

`11 configuraciones`

correspondientes a:

- 10 features;
- 11 features;
- 12 features;
- ...
- 20 features.

Esto permitirá estudiar cómo cambia el HI al modificar el número de variables seleccionadas.

---

# 22. Detalle definitivo de mRMR (cierre 33.5)

## 22.1 Relevancia

- Información mutua (MI) calculada por rodamiento y promediada entre los 17 rodamientos.
- Estimador: `mutual_info_regression` de scikit-learn con `n_neighbors = 3` y semilla fija.

## 22.2 Redundancia

Promedio de MI entre la feature candidata y las features ya seleccionadas.

## 22.3 Criterio mRMR

Máxima relevancia y mínima redundancia combinadas según la implementación estándar de mRMR.

## 22.4 Tratamiento de NaN e Infinitos

- **Método de imputación:** Interpolación lineal entre ráfagas vecinas. Si el NaN o infinito se encuentra en un extremo temporal, se copia el valor de la ráfaga vecina más cercana.
- **Límite de seguridad:** Si una feature supera el **5 %** de valores imputados dentro de un mismo rodamiento, el pipeline se detiene con un error de control de calidad, por tratarse de un problema estructural de la señal.

---

# 23. Construcción del Health Indicator — HI

Después de seleccionar las features se construirá el Health Indicator.

Para cada rodamiento individual:

- se utilizan sus primeras **70 ráfagas** como referencia del estado sano;
- se calcula la distancia de Mahalanobis de cada observación respecto a ese estado sano.

---

# 24. Referencia saludable

Para cada rodamiento:

`Healthy reference = primeras 70 ráfagas`

Sobre esas 70 observaciones se calcularán:

- vector medio;
- matriz de covarianza.

Conceptualmente:

`mu_sano = media(x_1, x_2, ..., x_70)`

y:

`Sigma_sano = covarianza(x_1, x_2, ..., x_70)`

donde cada `x_t` es el vector de features seleccionadas de una ráfaga.

---

# 25. Distancia de Mahalanobis

Para una observación `x_t`:

`MD_t = sqrt( (x_t - mu_sano)^T × Sigma_sano^(-1) × (x_t - mu_sano) )`

donde:

- `x_t` = vector de features de la ráfaga `t`;
- `mu_sano` = vector medio del tramo sano;
- `Sigma_sano` = matriz de covarianza del tramo sano;
- `Sigma_sano^(-1)` = inversa de la matriz de covarianza.

Interpretación:

- `MD` pequeño → observación parecida al estado sano;
- `MD` grande → observación alejada del estado sano.

---

# 26. Transformación de Mahalanobis a HI

Se utilizará:

`HI(t) = exp( -MD(t) / tau )`

donde:

- `MD(t)` = distancia de Mahalanobis en el instante `t`;
- `tau` = parámetro de escala.

Comportamiento esperado:

Si:

`MD ≈ 0`

entonces:

`HI ≈ 1`

Si:

`MD aumenta`

entonces:

`HI disminuye hacia 0`

Por tanto:

`HI alto ≈ estado cercano al sano`

`HI bajo ≈ estado alejado del sano`

El HI queda acotado entre:

`0 < HI <= 1`

---

# 27. Parámetro tau (cierre 33.7)

- **Base estadística:** Se calcula a partir del máximo de las distancias del tramo sano (`max(MD_sano)`).
- **Automatización:** El pipeline incluye un criterio de optimización automático que evalúa el rango de sensibilidad del factor multiplicativo (buscando estabilidad plana en el tramo sano y caída progresiva hacia la falla), seleccionando el óptimo sin requerir supervisión visual manual exhaustiva.

---

# 28. Visualización del HI

Se generarán curvas de HI para los **17 rodamientos**.

Se compararán diferentes cantidades de features seleccionadas:

- 10 features.
- 11 features.
- 12 features.
- ...
- 20 features.

La comparación permitirá observar:

- estabilidad del HI;
- comportamiento temporal;
- separación respecto al estado sano;
- sensibilidad a la degradación;
- efecto del número de features.

La visualización base será:

`17 rodamientos × diferentes números de features`

---

# 29. Flujo completo acordado

El pipeline completo queda definido conceptualmente como:

`PRONOSTIA/FEMTO`
↓
`17 rodamientos completos`
↓
`Ráfagas H + V de 2.560 puntos`
↓
`Eliminación de DC`
↓
`Filtro pasa-altos 10 Hz`
↓
`Sin normalización`
↓
`Extracción de 68 features`
↓
`Un archivo .parquet`
↓
`Una fila por ráfaga`
↓
`Mon + Tre + Pro + Rob`
↓
`EWM`
↓
`Ranking de las 68 features`
↓
`Top 30`
↓
`mRMR`
↓
`10–20 features`
↓
`Referencia sana = primeras 70 ráfagas`
↓
`Media + covarianza (Ledoit-Wolf)`
↓
`Distancia de Mahalanobis`
↓
`HI = exp(-MD/tau)`
↓
`17 curvas de HI`
↓
`Comparación entre 10 y 20 features`

---

# 30. Estructura conceptual de datos

Cada fila del `.parquet` representa una ráfaga.

Conceptualmente:

`bearing_id`
`burst_id`
`timestamp / referencia temporal`
`time en seg` además se debe calcular la columna time que permitirá graficar en función del tiempo en horas mediante cálculo
`feature_01`
`feature_02`
`...`
`feature_68`

Las columnas de identificación permiten:

- separar los 17 rodamientos;
- ordenar las ráfagas;
- calcular las métricas temporales;
- construir el tramo sano;
- reconstruir las curvas de HI.

Las 68 features permanecen almacenadas aunque solamente una parte sea utilizada posteriormente.

---

# 31. Principios de reproducibilidad

El procesamiento deberá mantener los siguientes principios:

1. Los datos RAW no se modifican.
2. El `.parquet` constituye el resultado procesado de la extracción.
3. Las 68 features originales se conservan.
4. La selección de features se realiza después de la extracción.
5. Los parámetros de cada etapa deben quedar explícitos.
6. Las decisiones cerradas no se ocultan dentro del código.
7. Las diferentes configuraciones deben poder reproducirse posteriormente.
8. El pipeline debe permitir cambiar:
   - número Top-K de EWM;
   - número de features mRMR;
   - definición de segmentos sano/falla;
   - suavizado de Robustez;
   - parámetro `tau`.

---

# 32. Estado actual — Cerrado

Están definidos y cerrados:

- Dataset PRONOSTIA/FEMTO.
- 17 rodamientos completos.
- Canales H y V.
- 2.560 muestras por ráfaga.
- Frecuencia de muestreo de 25,6 kHz.
- Eliminación de DC.
- Filtro pasa-altos de 10 Hz.
- Ausencia de corte superior.
- Ausencia de normalización de amplitud.
- 24 features temporales.
- 10 features espectrales.
- 32 features WPD.
- 2 features cruzadas.
- Total de 68 features.
- Fórmulas de las features.
- Mon.
- Tre.
- Pro.
- Rob.
- Concepto de EWM.
- Normalización inicial de EWM (proporción de beneficio).
- Ranking de 68 features.
- Filtro Top-30.
- Concepto de mRMR.
- Uso de MI de scikit-learn (`mutual_info_regression`, n_neighbors=3, semilla fija).
- Ausencia de escalado previo exclusivamente por diferencias de unidades.
- Evaluación de 10–20 features.
- Mahalanobis con regularización Ledoit-Wolf.
- Primeras 70 ráfagas como referencia sana.
- Transformación exponencial del HI.
- Todos los puntos de la sección 33.

---

# 33. Decisiones definitivas de los puntos pendientes (Cierre completo)

## 33.1 Varianza y desviación estándar

- **Decisión:** Se utilizará la **varianza y desviación estándar poblacional (`ddof = 0`)**.
- **Justificación:** Mantiene total coherencia con las funciones implícitas de `scipy.stats.skew` y `kurtosis` (momentos `m2` y `m4`), así como con el cálculo del RMS (que divide por `N`).

## 33.2 Robustez (`f_tilde_i`)

- Integrado operativamente a través del flujo de suavizado y evaluación de métricas definido en el pipeline de sensibilidad.

## 33.3 Pronosticabilidad (`f_sano` y `f_falla`)

- **Tramo sano:** Primeras **70 ráfagas** de cada rodamiento.
- **Tramo de falla:** Definido en base a la evolución final del ciclo de vida del rodamiento, parametrizado consistentemente para todos los rodamientos del dataset PRONOSTIA/FEMTO.

## 33.4 EWM (Entropy Weight Method)

- Normalización inicial mediante proporción de beneficio (`z_ij = x_ij / sum(x_ij)`), combinando de forma objetiva las cuatro métricas de sensibilidad (`Mon`, `Tre`, `Pro`, `Rob`) para ponderar las 68 features sin sesgos arbitrarios.

## 33.5 mRMR, NaN e Infinitos

- **Método de imputación:** **Interpolación lineal** entre ráfagas vecinas. Si el NaN o infinito se encuentra en un extremo temporal, se copia el valor de la ráfaga vecina más cercana.
- **Límite de seguridad:** Si una feature supera el **5 %** de valores imputados dentro de un mismo rodamiento, el pipeline se detiene con un error de control de calidad, por tratarse de un problema estructural de la señal.
- **MI:** Calculada por rodamiento y promediada entre los 17. Estimador: `mutual_info_regression` con `n_neighbors = 3` y semilla fija.

## 33.6 Mahalanobis y Covarianza

- **Regularización:** Uso de **Ledoit-Wolf** aplicado sobre la matriz de correlación (estandarizando previamente con la media y desviación del tramo sano). Esto corrige los problemas de pocas observaciones (70 ráfagas sanas vs. hasta 20 features) sin agregar parámetros manuales de ajuste.
- **Inversión de matriz:** Inversa directa con revisión estricta del número de condición. Si la matriz está mal condicionada, el pipeline emite una alerta o se detiene.

## 33.7 Tau (`HI = exp(-MD / τ)`)

- **Base estadística:** Se calcula a partir del máximo de las distancias del tramo sano (`max(MD_sano)`).
- **Automatización:** El pipeline incluye un criterio de optimización automático que evalúa el rango de sensibilidad del factor multiplicativo (buscando estabilidad plana en el tramo sano y caída progresiva hacia la falla), seleccionando el óptimo sin requerir supervisión visual manual exhaustiva.

## 33.8 Casos numéricos especiales

1. **Protección contra ceros y logaritmos:** Se añade un pequeño `ε` numérico (ej. `10^{-10}`) o se protegen los ceros previos antes de calcular funciones logarítmicas (como en *Spectral Flatness*). Los denominadores en riesgo de cero se protegen para evitar interrupciones por `ZeroDivisionError`.
2. **Features constantes:** El script detecta automáticamente aquellas features con varianza o desviación estándar igual a cero, les asigna un score de `0` y las descarta con un aviso en consola.
3. **Energías o valores globales en cero:** Si ráfagas completas presentan valores globales nulos (RMS, MAV, WPD, etc.), se tratan bajo la regla de los NaN (imputación lineal con límite del 5 %).
4. **Covarianza singular extrema:** Control de umbrales estrictos de condición numérica posteriores a Ledoit-Wolf para detener el proceso y alertar anomalías críticas.

---

# 34. Regla de implementación

**La especificación matemática está completa y cerrada.**

Orden de implementación recomendado:

1. Se fija la especificación matemática definitiva (este documento).
2. Se implementa la extracción.
3. Se verifica que existan exactamente 68 features.
4. Se genera el `.parquet`.
5. Se verifica estadísticamente el resultado.
6. Se ejecuta Mon/Tre/Pro/Rob.
7. Se ejecuta EWM.
8. Se seleccionan las Top-30.
9. Se ejecuta mRMR.
10. Se construyen los HI.
11. Se comparan las configuraciones de 10–20 features.

---

# 35. Resumen ejecutivo

El experimento busca construir un HI de rodamientos basado exclusivamente en información vibracional extraída automáticamente de las ráfagas PRONOSTIA/FEMTO.

La arquitectura metodológica es:

`Señal H/V`
→ `Preprocesamiento`
→ `68 features`
→ `Sensibilidad`
→ `EWM`
→ `Top-30`
→ `mRMR`
→ `10–20 features`
→ `Mahalanobis (Ledoit-Wolf)`
→ `HI`

La selección de features se realiza en dos etapas:

### Etapa 1 — EWM

Evalúa el comportamiento intrínseco de las 68 features mediante:

- Monotonicidad;
- Tendencia;
- Pronosticabilidad;
- Robustez.

### Etapa 2 — mRMR

Reduce las features considerando explícitamente:

- relación con el objetivo;
- redundancia entre variables.

Finalmente, las features seleccionadas se utilizan para construir un espacio de estado saludable y medir la desviación de cada ráfaga mediante Mahalanobis (con regularización Ledoit-Wolf).

La distancia se transforma en un indicador continuo:

`HI = exp(-MD / tau)`

que permite representar la evolución del estado del rodamiento desde un comportamiento cercano al estado sano hacia un comportamiento progresivamente más alejado de éste.

**Estado final de la especificación:** Completamente cerrado y listo para implementación del script definitivo.

---

*Documento consolidado generado a partir de:*
- *Especificación Técnica Consolidada PRONOSTIA/FEMTO*
- *Consolidado de Decisiones — Sección 33*
- *Minuta de Decisiones de Diseño — Pipeline RUL v3*
- *Cierres de puntos 33.1 a 33.8*
