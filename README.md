---
title: "Zyfra: Optimización del Coeficiente de Recuperación de Oro"
subtitle: "Modelado Predictivo Industrial y Métrica sMAPE Personalizada"
description: "Predicción de la eficiencia de extracción de oro en procesos de refinación industrial utilizando algoritmos de regresión evaluados con la métrica sMAPE."
categories: [Machine Learning, Regresión, Proyectos Industriales, Python]
---

# ⛏️ Optimización de recuperación de oro con métrica asimétrica (sMAPE)

**Problema de negocio:** Zyfra, empresa de eficiencia para la industria pesada, necesita optimizar el proceso de extracción y purificación de oro. El objetivo es construir un modelo predictivo que estime la recuperación de oro (concentrado rougher y concentrado final) para eliminar parámetros operativos no rentables.

**El reto técnico:** La competencia evalúa el modelo usando el **sMAPE ponderado** (Symmetric Mean Absolute Percentage Error), no MAE ni RMSE. Esta métrica castiga por igual las sobreestimaciones y subestimaciones, y el score final da un peso del 25% al concentrado rougher y un 75% al concentrado final.

---

## 📊 Resultados y Metodología

- **Modelos evaluados:** Regresión Lineal, Árbol de Decisión y Random Forest.
- **Métrica de evaluación:** Implementación explícita de `sMAPE = 100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))`.
- **Feature Engineering:** Tratamiento de nulos mediante propagación temporal (`ffill`) dado que los procesos físico-químicos son secuenciales.

---

## 📁 Contenido del repositorio

- `notebooks/01-prediccion-recuperacion.ipynb`: EDA, ingeniería de características, entrenamiento y evaluación con sMAPE.
- `requirements.txt`: Dependencias para reproducir el modelo.

---

## ▶️ Cómo ejecutar

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01-prediccion-recuperacion.ipynb