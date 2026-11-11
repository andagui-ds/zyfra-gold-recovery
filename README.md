# ⛏️ Zyfra — Predicción de Recuperación de Oro en Procesos Industriales

> Modelo de Machine Learning para predecir la eficiencia de recuperación de oro en dos etapas del proceso de flotación y purificación, utilizando la métrica sMAPE ponderada y previniendo la fuga de datos (Data Leakage).

## 🎯 Problema de Negocio

**Zyfra** (empresa de soluciones para la industria pesada) necesita optimizar la producción de oro y eliminar parámetros no rentables. El proceso de extracción y purificación es continuo y complejo. 

El objetivo es construir un prototipo que prediga la recuperación de oro en dos etapas críticas:
1. **Recuperación Rougher (Flotación primaria):** `rougher.output.recovery`
2. **Recuperación Final (Purificación):** `final.output.recovery`

La métrica de éxito definida por el negocio es el **sMAPE** (Error Medio Absoluto Porcentual Simétrico) ponderado, otorgando un 25% de peso a la etapa Rougher y un 75% a la etapa Final.

## 📊 Datos y Preprocesamiento

- **Fuentes:** Conjuntos de entrenamiento (`train`), prueba (`test`) y completo (`full`).
- **Fuga de datos (Data Leakage):** Se identificó que 34 características presentes en el entrenamiento no estaban en el conjunto de prueba. Estas correspondían a parámetros de salida (`output`) y cálculos (`calculation`). Se descartaron correctamente, ya que representan datos del "futuro" que no estarían disponibles en el momento de hacer la predicción real.
- **Imputación:** Al tratarse de un proceso industrial continuo, se utilizó propagación temporal (`ffill` / forward fill) para rellenar valores nulos.
- **Limpieza:** Se detectaron y eliminaron 2,806 registros anómalos donde la concentración total de metales era cero (físicamente imposible en el proceso).

## 🧠 Análisis Exploratorio (EDA)

- Se graficó la concentración de metales (Au, Ag, Pb) a lo largo de las etapas. Se confirmó que el Oro (Au) aumenta su concentración del ~8% en la materia prima hasta el ~45% en el producto final, validando la eficiencia del proceso.
- Se compararon las distribuciones del tamaño de partícula de la materia prima (`feed_size`) entre train y test, encontrando que son perfectamente estables, lo que garantiza la capacidad de generalización del modelo.

## 🏆 Resultados y Modelado

Se entrenaron dos modelos independientes (uno para la etapa Rougher y otro para la etapa Final) para capturar la dinámica específica de cada fase. Se probaron Regresión Lineal (base), Árboles de Decisión y Random Forest.

**Modelo Ganador: Random Forest (Entrenado con validación cruzada de 5 folds)**
- **sMAPE Rougher:** 7.34%
- **sMAPE Final:** 9.77%
- **sMAPE Total Ponderado (Test): 7.72%** *(Sobrepasando con éxito el umbral de calidad establecido por la empresa).*

## ✅ Recomendación de Negocio

El modelo de Random Forest está listo para ser integrado en el sistema de monitoreo de Zyfra. Permite a los operadores predecir en tiempo real cómo variaciones en los reactivos o en la alimentación afectarán la recuperación final. Al mantener el error sMAPE por debajo del 8%, la planta puede ajustar los parámetros operativos de manera proactiva, evitando caídas de eficiencia y maximizando el rendimiento del mineral extraído.

## 🛠️ Stack Técnico
- **Lenguajes:** Python
- **Librerías:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Técnicas:** Feature Engineering, Validación Cruzada (5-fold), Prevención de Data Leakage, Random Forest, Métrica sMAPE personalizada.

## 🔗 Ver el Análisis Completo
- 📓 [Notebook original con todo el código y las gráficas](notebooks/analisis-zyfra.ipynb)
