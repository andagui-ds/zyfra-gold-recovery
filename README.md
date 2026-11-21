[🇺🇸 Read in English](#-zyfra--gold-recovery-prediction-in-industrial-processes) | [🇲🇽 Leer en Español](#-zyfra--predicción-de-recuperación-de-oro-en-procesos-industriales)

---

# ⛏️ Zyfra — Gold Recovery Prediction in Industrial Processes

> Machine Learning model to predict gold recovery efficiency across two flotation and purification stages, evaluated using a custom weighted sMAPE metric while preventing Data Leakage.

🚀 **[Launch Interactive Streamlit App](https://zyfra-gold-recovery.streamlit.app/)** | 🌐 **[View Quarto Web Report](https://andagui-ds.github.io/zyfra-gold-recovery/)**

---

## 🎯 Business Problem

**Zyfra** (a heavy-industry digital solutions provider) needs to optimize gold production and eliminate unprofitable operational parameters. The extraction and purification process is continuous and highly complex.

The objective is to build a prototype predicting gold recovery across two critical stages:
1. **Rougher Recovery (Primary Flotation):** `rougher.output.recovery`
2. **Final Recovery (Purification):** `final.output.recovery`

The business success metric is the **weighted sMAPE** (Symmetric Mean Absolute Percentage Error), assigning a 25% weight to the Rougher stage and a 75% weight to the Final stage.

## 📊 Data and Preprocessing

* **Sources:** Training (`train`), test (`test`), and full (`full`) datasets.
* **Data Leakage Prevention:** Identified 34 features present in the training set that were absent in the test set. These corresponded to post-process `output` and `calculation` parameters. They were properly removed, as they represent "future" readings unavailable at real-time inference.
* **Imputation:** Since this is a continuous industrial process, temporal forward-fill propagation (`ffill`) was applied to handle missing values.
* **Data Cleaning:** Detected and removed 2,806 anomalous records where total metal concentration equaled zero (physically impossible in the extraction process).

## 🧠 Exploratory Data Analysis (EDA)

* Plotted metal concentrations (Au, Ag, Pb) across processing stages, confirming that Gold (Au) concentration increases from ~8% in raw feed to ~45% in the final concentrate, validating process efficiency.
* Compared raw feed particle size distributions (`feed_size`) between train and test sets, confirming stability and ensuring strong model generalization.

## 🏆 Modeling and Results

Two independent models were trained (one for the Rougher stage and one for the Final stage) to capture the specific dynamics of each phase. Linear Regression (baseline), Decision Trees, and Random Forest were evaluated.

**Winning Model: Random Forest (Trained with 5-Fold Cross-Validation)**
* **Rougher sMAPE:** 7.34%
* **Final sMAPE:** 9.77%
* **Total Weighted sMAPE (Test): 7.72%** *(Successfully surpassing the company's quality benchmark).*

## ✅ Business Recommendation

The Random Forest pipeline is ready for integration into Zyfra's plant monitoring system. It enables operators to forecast in real time how changes in reagents or raw feed affect final gold recovery. By keeping weighted sMAPE below 8%, plant engineers can proactively tune operational parameters, preventing efficiency drops and maximizing mineral yield.

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit
* **Techniques:** Feature Engineering, 5-Fold Cross-Validation, Data Leakage Prevention, Random Forest, Custom sMAPE Metric.

## 🔗 Full Analysis & Production App

* 📓 [Original Jupyter Notebook with full code and charts](notebooks/analisis-zyfra.ipynb)
* 🚀 [Interactive Gold Recovery Simulator (Streamlit)](https://zyfra-gold-recovery.streamlit.app/)

## ▶️ How to Run Locally

    pip install -r requirements.txt
    streamlit run app.py

---

# ⛏️ Zyfra — Predicción de Recuperación de Oro en Procesos Industriales

> Modelo de Machine Learning para predecir la eficiencia de recuperación de oro en dos etapas del proceso de flotación y purificación, utilizando la métrica sMAPE ponderada y previniendo la fuga de datos (Data Leakage).

🚀 **[Abrir App Interactiva en Streamlit](https://zyfra-gold-recovery.streamlit.app/)** | 🌐 **[Ver Reporte Web en Quarto](https://andagui-ds.github.io/zyfra-gold-recovery/)**

---

## 🎯 Problema de Negocio

**Zyfra** (empresa de soluciones para la industria pesada) necesita optimizar la producción de oro y eliminar parámetros no rentables. El proceso de extracción y purificación es continuo y complejo.

El objetivo es construir un prototipo que prediga la recuperación de oro en dos etapas críticas:
1. **Recuperación Rougher (Flotación primaria):** `rougher.output.recovery`
2. **Recuperación Final (Purificación):** `final.output.recovery`

La métrica de éxito definida por el negocio es el **sMAPE** (Error Medio Absoluto Porcentual Simétrico) ponderado, otorgando un 25% de peso a la etapa Rougher y un 75% a la etapa Final.

## 📊 Datos y Preprocesamiento

* **Fuentes:** Conjuntos de entrenamiento (`train`), prueba (`test`) y completo (`full`).
* **Fuga de datos (Data Leakage):** Se identificó que 34 características presentes en el entrenamiento no estaban en el conjunto de prueba. Estas correspondían a parámetros de salida (`output`) y cálculos (`calculation`). Se descartaron correctamente, ya que representan datos del "futuro" que no estarían disponibles en el momento de hacer la predicción real.
* **Imputación:** Al tratarse de un proceso industrial continuo, se utilizó propagación temporal (`ffill` / forward fill) para rellenar valores nulos.
* **Limpieza:** Se detectaron y eliminaron 2,806 registros anómalos donde la concentración total de metales era cero (físicamente imposible en el proceso).

## 🧠 Análisis Exploratorio (EDA)

* Se graficó la concentración de metales (Au, Ag, Pb) a lo largo de las etapas. Se confirmó que el Oro (Au) aumenta su concentración del ~8% en la materia prima hasta el ~45% en el producto final, validando la eficiencia del proceso.
* Se compararon las distribuciones del tamaño de partícula de la materia prima (`feed_size`) entre train y test, encontrando que son perfectamente estables, lo que garantiza la capacidad de generalización del modelo.

## 🏆 Resultados y Modelado

Se entrenaron dos modelos independientes (uno para la etapa Rougher y otro para la etapa Final) para capturar la dinámica específica de cada fase. Se probaron Regresión Lineal (base), Árboles de Decisión y Random Forest.

**Modelo Ganador: Random Forest (Entrenado con validación cruzada de 5 folds)**
* **sMAPE Rougher:** 7.34%
* **sMAPE Final:** 9.77%
* **sMAPE Total Ponderado (Test): 7.72%** *(Sobrepasando con éxito el umbral de calidad establecido por la empresa).*

## ✅ Recomendación de Negocio

El modelo de Random Forest está listo para ser integrado en el sistema de monitoreo de Zyfra. Permite a los operadores predecir en tiempo real cómo variaciones en los reactivos o en la alimentación afectarán la recuperación final. Al mantener el error sMAPE por debajo del 8%, la planta puede ajustar los parámetros operativos de manera proactiva, evitando caídas de eficiencia y maximizando el rendimiento del mineral extraído.

## 🛠️ Stack Técnico

* **Lenguajes:** Python
* **Librerías:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit
* **Técnicas:** Feature Engineering, Validación Cruzada (5-fold), Prevención de Data Leakage, Random Forest, Métrica sMAPE personalizada.

## 🔗 Ver el Análisis Completo y App en Vivo

* 📓 [Notebook original con todo el código y las gráficas](notebooks/analisis-zyfra.ipynb)
* 🚀 [Simulador Interactivo de Recuperación de Oro (Streamlit)](https://zyfra-gold-recovery.streamlit.app/)

## ▶️ Cómo ejecutar localmente

    pip install -r requirements.txt
    streamlit run app.py
