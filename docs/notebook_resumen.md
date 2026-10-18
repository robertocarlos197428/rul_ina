# Resumen de Trabajo del Día — Pipeline RUL (Análisis y Depuración de Features)

**Fecha:** 17 de Octubre de 2026  
**Contexto:** Análisis cuantitativo y visual de características espectrales, depuración de métricas y consolidación de la selección de features para la estimación de RUL.

---

## 📍 1. Análisis de Imágenes del Dominio Frecuencial (`spectrum_H.png` y `spectrum_V.png`)

Se realizó la evaluación cualitativa y cuantitativa de las **5 características del dominio espectral/frecuencial** (`SpectralCentroid`, `SpectralRMS`, `SpectralStd`, `SpectralEntropy` y `SpectralFlatness`) a lo largo de la vida útil de los **17 rodamientos** del dataset PRONOSTIA (FEMTO):

* **`SpectralStd_H` (Líder Absoluto):** Destacó como la variable de mayor calidad del dataset (Score EWM: **0.6412**), mostrando una curva suave, continua y monótona de degradación con altísima robustez ($0.986$) y sin saltos erráticos.
* **Coincidencia Matemática entre `SpectralRMS` y `RMS` Bruto:** Se explicó matemáticamente por qué ambas métricas se comportan de forma idéntica (planas hasta el colapso final), fundamentado en el **Teorema de Parseval / Rayleigh** (conservación de la energía entre el dominio del tiempo y la frecuencia).
* **Comparación Eje Horizontal ($H$) vs. Vertical ($V$):** El canal horizontal presenta señales significativamente más limpias y monótonas debido a que la carga radial del banco de ensayos actúa predominantemente en dicha dirección.

---

## 🔍 2. Diagnóstico Físico del Comportamiento Anómalo en `Bearing1_3`

Se analizó la divergencia opuesta observada en el rodamiento `Bearing1_3` entre la **Entropía Espectral (`SpectralEntropy`)** y la **Desviación Estándar Espectral (`SpectralStd`)**:
* **Fase de Falla Localizada ($t = 2.0\text{ h}$ a $5.0\text{ h}$):** La aparición de picos armónicos tonales limpios concentra la energía, ordenando el espectro y haciendo **caer la Entropía** (de $9.8$ a $8.8$). Al mismo tiempo, como estos picos abarcan frecuencias bajas y altas de resonancia, la **Desviación Estándar se mantiene elevada** ($\sim 4,000$).
* **Fase de Colapso ($t > 5.5\text{ h}$):** El desprendimiento masivo (*spalling*) destruye la estructura tonal, generando ruido de banda ancha que **hace rebotar la Entropía hacia arriba** y desploma la Desviación Estándar.

---

## 🧮 3. Profundización Teórica en las Métricas de Evaluación

Se definieron las fórmulas y significados físicos de las métricas de evaluación de características:
1. **Monotonicidad ($\text{Mon}$):**
   * **Conteo de Signos ($\text{Mon\_sign}$):** Mide diferencias locales ráfaga a ráfaga. Se demostró que es inútil para la selección de features crudas por ser hiper-sensible al ruido estocástico ($\approx 0.01$).
   * **Correlación de Rangos de Spearman ($\text{Mon\_spearman}$):** Mide la tendencia macroscópica global a lo largo del tiempo, filtrando fluctuaciones locales.
2. **Tendibilidad ($\text{Trend}$):** Correlación de Pearson entre la característica y el tiempo.
3. **Prognosticabilidad ($\text{Prog}$):** Métrica **poblacional/global** que evalúa la dispersión ($\sigma$) del valor alcanzado en el momento del colapso ($EOL$) entre todos los rodamientos.
4. **Robustez ($\text{Rob}$):** Tolerancia al ruido evaluada mediante suavizado EWMA.

---

## ✂️ 4. Decisiones de Simplificación y Filtrado del Dataset

Se tomaron las siguientes decisiones clave para limpiar la matriz de análisis:

1. **Fijación de Parámetro Único de Suavizado ($\alpha = 0.10$):**
   * Se descartaron los valores de $\alpha \in \{0.05, 0.20, 0.30, 0.50\}$, fijando **$\alpha = 0.10$** como el filtro EWMA óptimo (equilibrio entre reducción de ruido y cero desfase).
2. **Descarte Definitivo de `Mon_sign`:**
   * Se elimina del criterio de selección de features, adoptando exclusivamente **`Mon_spearman`**.
3. **Desduplicación y Reducción de Features (de 68 a 48 variables):**
   * Se eliminaron **20 características redundantes o inestables**:
     * **Dominio Temporal (16 eliminadas):** `Std`, `Variance`, `MAV`, `Peak`, `P2P`, `ShapeFactor`, `MarginFactor`, `ImpulseFactor` (en ambos canales $H$ y $V$).
     * **Dominio Espectral (4 eliminadas):** `SpectralFlatness` y `SpectralRMS` (en ambos canales $H$ y $V$).
   * **Estructura final de 48 features restantes:** 32 Wavelet (`WPD_Erel`), 8 Temporales (`RMS`, `CrestFactor`, `Kurtosis`, `Skewness` en $H$ y $V$), 6 Espectrales (`SpectralStd`, `SpectralCentroid`, `SpectralEntropy` en $H$ y $V$) y 2 Inter-canal (`Pearson_HV`, `EnergyRatio_HV`).

---

## 📊 5. Consolidación de Resultados y Top 15 de Features

A partir de las 1,156 evaluaciones desglosadas en `metricas_hi_por_rodamiento.csv`, se consolidó la tabla a nivel macro mediante la **Mediana de los 17 rodamientos** (caso típico) con control del peor caso (**Mínimo**):

| Ranking | Feature | Mediana Spearman | Mediana Trend | Robustez ($\alpha=0.10$) | Prognosability |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **#1** | **`SpectralStd_H`** | **0.8309** | 0.8196 | 0.9865 | 0.7566 |
| **#2** | **`WPD_Erel_6_H`** | **0.7570** | 0.7613 | 0.8770 | 0.6424 |
| **#3** | **`WPD_Erel_14_H`** | **0.7289** | 0.6007 | 0.8709 | 0.3934 |
| **#4** | **`WPD_Erel_13_H`** | **0.6963** | 0.5266 | 0.8744 | 0.3971 |
| **#5** | **`WPD_Erel_4_H`** | **0.6915** | 0.7009 | 0.8836 | 0.7067 |
| **#6** | **`WPD_Erel_12_V`** | **0.6693** | 0.5447 | 0.9064 | 0.2408 |
| **#7** | **`WPD_Erel_15_H`** | **0.6685** | 0.6267 | 0.8737 | 0.2874 |
| **#8** | **`WPD_Erel_7_H`** | **0.6680** | 0.6416 | 0.8781 | 0.4431 |
| **#9** | **`SpectralCentroid_V`** | **0.6335** | 0.6076 | 0.9865 | 0.1560 |
| **#10** | **`WPD_Erel_3_H`** | **0.6330** | 0.6213 | 0.8812 | 0.7104 |
| **#11** | **`WPD_Erel_4_V`** | **0.6145** | 0.5710 | 0.8884 | 0.3406 |
| **#12** | **`WPD_Erel_12_H`** | **0.6129** | 0.5309 | 0.8783 | 0.2591 |
| **#13** | **`Pearson_HV`** | **0.5999** | 0.5016 | 0.8811 | 0.0986 |
| **#14** | **`WPD_Erel_11_V`** | **0.5971** | 0.5796 | 0.9046 | 0.2535 |
| **#15** | **`SpectralCentroid_H`** | **0.5890** | 0.5830 | 0.9746 | 0.2947 |

---

## 🚀 Próximos Pasos Recomendados

1. **[ ] Generar Gráfico de Barras Consolidado:** Graficar las 48 features ordenadas por su Mediana de Spearman para la presentación técnica.
2. **[ ] Aplicación de mRMR sobre las 48 Features:** Ejecutar el filtro de Redundancia Mínima y Relevancia Máxima para seleccionar la sub-cesta final de 12 a 15 variables no correlacionadas.
3. **[ ] Fusión y Construcción del Nuevo $HI$:** Calcular el $HI$ basado en Distancia de Mahalanobis usando la cesta optimizada.
