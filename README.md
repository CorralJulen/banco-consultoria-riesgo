# 🏦 Análisis Integral de Riesgo y Rentabilidad Bancaria

> Proyecto de consultoría de datos aplicado al sector bancario, desarrollado como parte del Máster en Big Data, Business Analytics e Inteligencia Artificial.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://banco-consultoria-riesgo.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📋 Descripción

Este repositorio simula un encargo real de consultoría de datos para una entidad financiera. El objetivo es demostrar cómo el análisis de datos y el machine learning apoyan la **toma de decisiones estratégicas** en tres áreas críticas de la banca moderna.

El proyecto incluye análisis exploratorio, modelos predictivos, interpretabilidad con SHAP y un dashboard interactivo desplegado en producción — todo documentado con orientación de negocio, no solo técnica.

🔗 **[Ver dashboard en vivo](https://banco-consultoria-riesgo.streamlit.app)**

---

## 🗂️ Estructura del proyecto

```
banco-consultoria-riesgo/
│
├── datos/
│   ├── raw/                    # Datasets originales sin modificar
│   └── procesados/             # Datos limpios listos para modelar
│
├── notebooks/
│   ├── 01_riesgo_credito/
│   │   ├── 01_exploracion.ipynb
│   │   ├── 02_preprocesamiento.ipynb
│   │   └── 03_modelado.ipynb
│   ├── 02_deteccion_fraude/
│   │   ├── 01_exploracion.ipynb
│   │   ├── 02_preprocesamiento.ipynb
│   │   └── 03_modelado.ipynb
│   └── 03_rentabilidad/
│       ├── 01_exploracion.ipynb
│       └── 02_analisis.ipynb
│
├── src/
│   └── utils.py
│
├── dashboard/
│   └── app.py                  # Dashboard Streamlit desplegado en producción
│
├── requirements.txt
└── README.md
```

---

## 📦 Módulos

### Módulo 1 — Riesgo de Crédito

**Pregunta de negocio:** ¿Qué variables predicen mejor el impago de un cliente?

- **Dataset:** [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (UCI) — 1.000 clientes, 20 variables
- **Técnicas:** Regresión Logística · Random Forest · XGBoost · SMOTE · SHAP
- **Resultado:** La Regresión Logística obtiene el mejor AUC-ROC (**0.7977**), superando a modelos más complejos. El análisis SHAP revela que el estado de la cuenta corriente y la duración del crédito son los predictores más potentes.
- **Entregable:** Simulador interactivo de scoring disponible en el dashboard

### Módulo 2 — Detección de Fraude

**Pregunta de negocio:** ¿Cómo minimizar las pérdidas por fraude sin penalizar a clientes legítimos?

- **Dataset:** [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle) — 284.807 transacciones, 0.17% fraude
- **Técnicas:** Isolation Forest · XGBoost · SMOTE (sampling_strategy=0.1) · Curva Precision-Recall
- **Resultado:** XGBoost alcanza un Average Precision de **0.852** — 500x mejor que el baseline aleatorio. Detecta 86 de 98 fraudes reales con 131 falsas alarmas (umbral óptimo F1 = 0.85).
- **Entregable:** Análisis del umbral de decisión con implicaciones de negocio

### Módulo 3 — Rentabilidad y Eficiencia Bancaria

**Pregunta de negocio:** ¿Qué factores determinan la rentabilidad y eficiencia de una entidad bancaria?

- **Dataset:** [FDIC BankFind Suite API](https://banks.data.fdic.gov/docs/) — 5.000+ bancos americanos, datos 2019-2023
- **Técnicas:** Análisis de ratios (ROA, ROE, NIM, CIR) · K-Means clustering · PCA · Benchmarking sectorial
- **Resultado:** Se identifican 4 perfiles bancarios distintos. La correlación CIR/ROA de **-0.75** confirma que la eficiencia operativa es la palanca más directa hacia la rentabilidad.
- **Entregable:** Mapa de posicionamiento sectorial + radar chart por perfil

---

## 📊 Resultados clave

| Módulo | Modelo ganador | Métrica principal | Resultado |
|--------|---------------|-------------------|-----------|
| Riesgo de Crédito | Regresión Logística | AUC-ROC | **0.7977** |
| Detección de Fraude | XGBoost | Average Precision | **0.852** |
| Rentabilidad | K-Means (k=4) | Silhouette Score | **0.33** |

---

## 🛠️ Tecnologías utilizadas

| Categoría | Herramientas |
|-----------|-------------|
| Lenguaje | Python 3.10+ |
| Análisis de datos | pandas · numpy |
| Machine Learning | scikit-learn · xgboost · imbalanced-learn |
| Interpretabilidad | shap |
| Visualización | matplotlib · seaborn · plotly |
| Dashboard | Streamlit |
| Datos externos | FDIC BankFind Suite API |
| Entorno | Jupyter Notebook |

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/corraljulen/banco-consultoria-riesgo.git
cd banco-consultoria-riesgo
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar los notebooks
Abrir Jupyter y navegar a la carpeta `notebooks/`. Seguir el orden numérico de los módulos.

### 4. Ver el dashboard en local
```bash
streamlit run dashboard/app.py
```

---

## 🤖 Nota sobre el proceso de desarrollo

Este proyecto fue desarrollado mediante **vibe coding** — una metodología de desarrollo asistido por IA en la que el programador describe los objetivos en lenguaje natural y construye el proyecto de forma iterativa con un modelo de lenguaje (en este caso, Claude de Anthropic).


---

## ⚠️ Aviso

Este proyecto es de carácter **académico y divulgativo**. Los datasets utilizados son públicos y anonimizados. Las conclusiones y recomendaciones son ilustrativas y no constituyen asesoramiento financiero.

---

## 👤 Autor

**Julen Corral** · Máster en Big Data, Business Analytics e IA
📍 Bilbao, País Vasco
