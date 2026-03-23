# 🏦 Análisis Integral de Riesgo y Rentabilidad Bancaria

> Proyecto de consultoría de datos aplicado al sector bancario, desarrollado como parte del Máster en Big Data, Business Analytics e Inteligencia Artificial.

---

## 📋 Descripción

Este repositorio simula un encargo de consultoría de datos para una entidad financiera ficticia. El objetivo es demostrar cómo el análisis de datos y el machine learning pueden apoyar la **toma de decisiones estratégicas** en tres áreas críticas de la banca moderna:

- **Riesgo de crédito**: ¿A quién conceder un préstamo y en qué condiciones?
- **Detección de fraude**: ¿Cómo identificar transacciones fraudulentas en tiempo real?
- **Rentabilidad y eficiencia**: ¿Qué factores determinan la salud financiera de una entidad?

El enfoque combina rigor técnico con orientación al negocio: cada módulo incluye no solo el modelo, sino también las **conclusiones accionables** que un consultor presentaría a dirección.

---

## 🗂️ Estructura del proyecto

```
banco-consultoria-riesgo/
│
├── datos/
│   ├── raw/              # Datasets originales sin modificar
│   └── procesados/       # Datos limpios listos para modelar
│
├── notebooks/
│   ├── 01_riesgo_credito/
│   ├── 02_deteccion_fraude/
│   └── 03_rentabilidad/
│
├── src/
│   └── utils.py          # Funciones auxiliares compartidas
│
├── dashboard/
│   └── app.py            # Dashboard interactivo (Streamlit)
│
└── informe/
    └── executive_summary.pdf
```

---

## 📦 Módulos

### Módulo 1 — Riesgo de Crédito

**Pregunta de negocio:** ¿Qué variables predicen mejor el impago de un cliente?

- Dataset: [German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (UCI Machine Learning Repository)
- Técnicas: Análisis exploratorio · Regresión logística · Random Forest · XGBoost
- Interpretabilidad: SHAP values para explicar las predicciones en lenguaje de negocio
- **Entregable clave:** Perfil de cliente de alto riesgo + recomendaciones de política crediticia

### Módulo 2 — Detección de Fraude

**Pregunta de negocio:** ¿Cómo minimizar las pérdidas por fraude sin penalizar a clientes legítimos?

- Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle)
- Técnicas: Tratamiento de clases desbalanceadas (SMOTE) · Isolation Forest · XGBoost
- Métricas: Precision, Recall, F1 y AUC-ROC 
- **Entregable clave:** Umbral de decisión óptimo según coste de falso positivo vs. falso negativo

### Módulo 3 — Rentabilidad y Eficiencia Bancaria

**Pregunta de negocio:** ¿Qué diferencia a los bancos más rentables del resto?

- Dataset: Datos públicos del Banco Central Europeo (BCE) / Banco de España
- Técnicas: Análisis de ratios financieros (ROE, ROA, CIR) · Clustering · Visualización
- **Entregable clave:** Benchmarking del sector y palancas de mejora de la rentabilidad

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
| Entorno | Jupyter Notebook |

---

## 🚀 Cómo ejecutar el proyecto

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/banco-consultoria-riesgo.git
cd banco-consultoria-riesgo
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar los notebooks
Abrir Jupyter y navegar a la carpeta `notebooks/`. Se recomienda seguir el orden numérico de los módulos.

### 4. Ver el dashboard
```bash
streamlit run dashboard/app.py
```

---

## 📊 Dashboard

El dashboard interactivo permite explorar los resultados de los tres módulos de forma visual e intuitiva, orientado a un perfil no técnico (dirección, comités de riesgo).

🔗 **[Ver dashboard en vivo](https://tu-usuario-banco-consultoria.streamlit.app)** *(disponible próximamente)*

---

## 📄 Informe ejecutivo

El informe resume los hallazgos principales de los tres módulos en formato de presentación ejecutiva, sintetizando las conclusiones de negocio más relevantes.

📥 [Descargar Executive Summary (PDF)](informe/executive_summary.pdf) *(disponible próximamente)*

---

## ⚠️ Aviso

Este proyecto es de carácter **académico y divulgativo**. Los datasets utilizados son públicos y anonimizados. Las conclusiones y recomendaciones son ilustrativas y no constituyen asesoramiento financiero.

---

## 👤 Autor

**Julen** · Máster en Big Data, Business Analytics e IA  
📍 Barakaldo, País Vasco  
🔗 [LinkedIn] https://www.linkedin.com/in/julen-corral-lop-486295219/ 
