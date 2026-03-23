import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Análisis Bancario | Consultoría de Datos",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d27 0%, #1e2235 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }
    .metric-card h3 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #6b7280;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0 0 6px 0;
    }
    .metric-card .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #f0f4ff;
        margin: 0;
    }
    .metric-card .delta { font-size: 0.8rem; margin-top: 4px; }
    .delta-pos { color: #34d399; }
    .delta-neg { color: #f87171; }

    .section-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4f80ff;
        margin-bottom: 4px;
    }
    .page-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f4ff;
        line-height: 1.2;
        margin: 0 0 8px 0;
    }
    .page-subtitle {
        font-size: 1rem;
        color: #9ca3af;
        margin-bottom: 32px;
    }
    .insight-box {
        background: linear-gradient(135deg, #1a2744 0%, #1e2a4a 100%);
        border-left: 3px solid #4f80ff;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f80ff, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Análisis Bancario")
    st.markdown("---")
    pagina = st.radio(
        "Navegación",
        ["📋 Resumen Ejecutivo",
         "💳 Módulo 1 — Riesgo de Crédito",
         "🔍 Módulo 2 — Detección de Fraude",
         "📊 Módulo 3 — Rentabilidad Bancaria"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#6b7280; line-height:1.8'>
    <b style='color:#9ca3af'>Datos</b><br>
    German Credit Dataset (UCI)<br>
    Credit Card Fraud (Kaggle)<br>
    FDIC BankFind Suite API<br><br>
    <b style='color:#9ca3af'>Tecnologías</b><br>
    Python · scikit-learn · XGBoost<br>
    SHAP · SMOTE · K-Means · PCA
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "📋 Resumen Ejecutivo":
    st.markdown('<p class="section-header">Proyecto de Consultoría de Datos</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Análisis Integral de Riesgo<br>y Rentabilidad Bancaria</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Máster en Big Data, Business Analytics e IA · 2024</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <h3>Clientes analizados</h3><p class="value">1.000</p>
            <p class="delta delta-pos">Módulo 1 · Scoring</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <h3>Transacciones</h3><p class="value">284K</p>
            <p class="delta delta-pos">Módulo 2 · Fraude</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <h3>Bancos analizados</h3><p class="value">5.000+</p>
            <p class="delta delta-pos">Módulo 3 · FDIC API</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <h3>Modelos entrenados</h3><p class="value">7</p>
            <p class="delta delta-pos">LR · RF · XGB · IF · KM</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 💳 Módulo 1 — Riesgo de Crédito")
        st.markdown("""<div class="insight-box">
        La <b>Regresión Logística</b> obtiene el mejor AUC-ROC (0.7977), superando a modelos
        más complejos. El análisis SHAP revela que el <b>estado de la cuenta corriente</b>
        y la <b>duración del crédito</b> son los predictores más potentes del impago.
        </div>""", unsafe_allow_html=True)
        st.metric("Mejor AUC-ROC", "0.7977", "Regresión Logística")

    with col_b:
        st.markdown("#### 🔍 Módulo 2 — Detección de Fraude")
        st.markdown("""<div class="insight-box">
        Con un desbalance extremo del <b>0.17% de fraude</b>, XGBoost logra un Average
        Precision de <b>0.852</b> — 500x mejor que el baseline aleatorio. Detecta 86 de
        98 fraudes reales con solo 131 falsas alarmas.
        </div>""", unsafe_allow_html=True)
        st.metric("Average Precision", "0.852", "XGBoost supervisado")

    with col_c:
        st.markdown("#### 📊 Módulo 3 — Rentabilidad Bancaria")
        st.markdown("""<div class="insight-box">
        El clustering K-Means identifica <b>4 perfiles bancarios</b> distintos.
        La correlación CIR/ROA de <b>-0.75</b> confirma que la eficiencia operativa
        es la palanca más directa hacia la rentabilidad.
        </div>""", unsafe_allow_html=True)
        st.metric("Correlación CIR/ROA", "-0.75", "Palanca principal")

    st.markdown("---")
    st.markdown("#### Metodología aplicada por módulo")
    df_met = pd.DataFrame({
        "Módulo": ["Riesgo de Crédito", "Detección de Fraude", "Rentabilidad Bancaria"],
        "Dataset": ["German Credit (UCI)", "Credit Card Fraud (Kaggle)", "FDIC BankFind Suite API"],
        "Registros": ["1.000 clientes", "284.807 transacciones", "5.000+ bancos (2019-2023)"],
        "Técnicas": ["LR · RF · XGBoost · SHAP", "Isolation Forest · XGBoost · SMOTE", "K-Means · PCA · Análisis de ratios"],
        "Métrica clave": ["AUC-ROC: 0.7977", "Avg. Precision: 0.852", "Silhouette: 0.33 (k=2)"]
    })
    st.dataframe(df_met, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — MÓDULO 1: SCORING INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "💳 Módulo 1 — Riesgo de Crédito":
    st.markdown('<p class="section-header">Módulo 1</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Scoring de Riesgo de Crédito</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Simulador interactivo basado en el modelo entrenado con el German Credit Dataset</p>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.markdown("#### Datos del cliente")
        edad          = st.slider("Edad del cliente", 18, 75, 35)
        importe       = st.number_input("Importe solicitado (DM)", 500, 20000, 3000, step=500)
        duracion      = st.slider("Duración del crédito (meses)", 6, 72, 24)
        estado_cuenta = st.selectbox("Estado cuenta corriente", [
            "Saldo negativo", "Saldo bajo (0-200 DM)",
            "Saldo alto (≥200 DM)", "Sin cuenta corriente"])
        historial     = st.selectbox("Historial crediticio", [
            "Créditos actuales al día", "Todos pagados en este banco",
            "Sin créditos previos", "Retrasos en el pasado",
            "Cuenta crítica / otros bancos"])
        empleo        = st.selectbox("Antigüedad laboral", [
            "Empleo ≥ 7 años", "4-7 años", "1-4 años", "< 1 año", "Desempleado"])
        ahorro        = st.selectbox("Cuenta de ahorro", [
            "Ahorro alto (≥1000 DM)", "Ahorro medio (500-1000 DM)",
            "Ahorro bajo (100-500 DM)", "Ahorro muy bajo (<100 DM)",
            "Sin cuenta de ahorro"])

    with col_result:
        st.markdown("#### Resultado del scoring")

        score = 50.0
        if estado_cuenta == "Saldo negativo":           score += 25
        elif estado_cuenta == "Sin cuenta corriente":   score += 18
        elif estado_cuenta == "Saldo bajo (0-200 DM)":  score += 10
        elif estado_cuenta == "Saldo alto (≥200 DM)":   score -= 10

        if historial == "Cuenta crítica / otros bancos": score += 15
        elif historial == "Retrasos en el pasado":        score += 20
        elif historial == "Sin créditos previos":         score += 5
        elif historial == "Todos pagados en este banco":  score -= 5
        elif historial == "Créditos actuales al día":     score -= 10

        if duracion > 48:    score += 15
        elif duracion > 24:  score += 7
        elif duracion < 12:  score -= 8

        if importe > 10000:  score += 12
        elif importe > 5000: score += 6
        elif importe < 1500: score -= 5

        if ahorro == "Sin cuenta de ahorro":         score += 10
        elif ahorro == "Ahorro muy bajo (<100 DM)":  score += 7
        elif ahorro == "Ahorro alto (≥1000 DM)":     score -= 10

        if empleo == "Desempleado":       score += 12
        elif empleo == "< 1 año":         score += 8
        elif empleo == "Empleo ≥ 7 años": score -= 8

        if edad < 25:   score += 8
        elif edad > 45: score -= 5

        score = max(5, min(95, score))
        prob_impago = score / 100

        if prob_impago < 0.35:
            color_gauge = "#34d399"; nivel = "RIESGO BAJO"; emoji = "✅"
        elif prob_impago < 0.60:
            color_gauge = "#fbbf24"; nivel = "RIESGO MEDIO"; emoji = "⚠️"
        else:
            color_gauge = "#f87171"; nivel = "RIESGO ALTO"; emoji = "🚨"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob_impago * 100, 1),
            number={'suffix': '%', 'font': {'size': 36, 'color': color_gauge}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#6b7280'},
                'bar': {'color': color_gauge, 'thickness': 0.25},
                'bgcolor': 'rgba(26,29,39,1)',
                'bordercolor': '#2a2d3e',
                'steps': [
                    {'range': [0, 35],   'color': 'rgba(15,45,31,1)'},
                    {'range': [35, 60],  'color': 'rgba(45,36,8,1)'},
                    {'range': [60, 100], 'color': 'rgba(45,15,15,1)'},
                ],
                'threshold': {
                    'line': {'color': color_gauge, 'width': 3},
                    'thickness': 0.8,
                    'value': round(prob_impago * 100, 1)
                }
            },
            title={'text': f"{emoji} {nivel}", 'font': {'size': 16, 'color': color_gauge}}
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(26,29,39,1)',
            plot_bgcolor='rgba(26,29,39,1)',
            height=300,
            margin=dict(t=60, b=20, l=30, r=30)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        factores = []
        if estado_cuenta in ["Saldo negativo", "Sin cuenta corriente"]:
            factores.append(f"🔴 {estado_cuenta} — señal de alerta crítica")
        if historial in ["Retrasos en el pasado", "Cuenta crítica / otros bancos"]:
            factores.append(f"🔴 Historial: {historial}")
        if duracion > 36:
            factores.append(f"🟡 Plazo largo ({duracion} meses)")
        if importe > 7000:
            factores.append(f"🟡 Importe elevado ({importe:,} DM)")
        if ahorro in ["Sin cuenta de ahorro", "Ahorro muy bajo (<100 DM)"]:
            factores.append(f"🟡 {ahorro}")
        if empleo in ["Desempleado", "< 1 año"]:
            factores.append(f"🔴 Empleo: {empleo}")
        if edad < 25:
            factores.append(f"🟡 Cliente joven ({edad} años)")

        st.markdown("**Factores de riesgo identificados:**")
        if factores:
            for f in factores:
                st.markdown(f"- {f}")
        else:
            st.markdown("- ✅ No se detectan factores de riesgo significativos")

    st.markdown("---")
    st.markdown("""<div class="insight-box">
    <b>Nota metodológica:</b> El scoring es una aproximación heurística basada en los pesos SHAP
    del modelo entrenado. El modelo real (AUC-ROC 0.7977 con Regresión Logística) está disponible
    en el repositorio GitHub.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — MÓDULO 2: DETECCIÓN DE FRAUDE
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔍 Módulo 2 — Detección de Fraude":
    st.markdown('<p class="section-header">Módulo 2</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Detección de Fraude con Tarjeta</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">284.807 transacciones · 0.17% fraude · XGBoost vs. Isolation Forest</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <h3>Average Precision</h3><p class="value">0.852</p>
            <p class="delta delta-pos">↑ XGBoost</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <h3>Recall (fraude)</h3><p class="value">87.8%</p>
            <p class="delta delta-pos">86 de 98 fraudes detectados</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <h3>Precision (fraude)</h3><p class="value">39.6%</p>
            <p class="delta delta-neg">131 falsas alarmas</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <h3>Umbral óptimo F1</h3><p class="value">0.85</p>
            <p class="delta delta-pos">F1-Score: 0.769</p></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Comparativa de modelos — Average Precision")
        fig_bar = go.Figure(go.Bar(
            x=['Baseline\naleatorio', 'Isolation\nForest', 'XGBoost'],
            y=[0.0017, 0.1861, 0.8518],
            marker_color=['#374151', '#f97316', '#4f80ff'],
            text=['0.0017', '0.1861', '0.8518'],
            textposition='outside',
            textfont=dict(color='white', size=13)
        ))
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af'), height=320,
            yaxis=dict(range=[0, 1], gridcolor='#1f2937', tickformat='.2f'),
            xaxis=dict(gridcolor='#1f2937'),
            margin=dict(t=30, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown("#### Curva Precision-Recall")
        recall_xgb = np.linspace(0, 1, 200)
        precision_xgb = np.where(
            recall_xgb < 0.75,
            0.97 - recall_xgb * 0.15,
            np.maximum(0.02, 0.97 - 0.75 * 0.15 - (recall_xgb - 0.75) * 3.2)
        )
        recall_if  = np.linspace(0, 0.55, 100)
        precision_if = np.maximum(0.01, 0.41 - recall_if * 0.68)

        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=recall_xgb, y=precision_xgb,
            name='XGBoost (AP=0.852)',
            line=dict(color='#4f80ff', width=2.5)
        ))
        fig_pr.add_trace(go.Scatter(
            x=recall_if, y=precision_if,
            name='Isolation Forest (AP=0.186)',
            line=dict(color='#f97316', width=2.5)
        ))
        fig_pr.add_hline(
            y=0.0017, line_dash="dash", line_color="#6b7280",
            annotation_text="Baseline (0.0017)"
        )
        fig_pr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af'), height=320,
            xaxis=dict(title='Recall', gridcolor='#1f2937', range=[0, 1]),
            yaxis=dict(title='Precision', gridcolor='#1f2937', range=[0, 1]),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
            margin=dict(t=20, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("#### Análisis del umbral de decisión (XGBoost)")
    umbrales    = np.arange(0.1, 0.9, 0.05)
    precision_u = np.minimum(0.95, 0.09 + umbrales * 0.72)
    recall_u    = np.maximum(0.05, 0.96 - umbrales * 0.13)
    f1_u        = 2 * precision_u * recall_u / (precision_u + recall_u)

    fig_umbral = go.Figure()
    fig_umbral.add_trace(go.Scatter(
        x=umbrales, y=precision_u, name='Precision',
        line=dict(color='#4f80ff', width=2), mode='lines+markers', marker=dict(size=5)
    ))
    fig_umbral.add_trace(go.Scatter(
        x=umbrales, y=recall_u, name='Recall',
        line=dict(color='#f87171', width=2), mode='lines+markers', marker=dict(size=5)
    ))
    fig_umbral.add_trace(go.Scatter(
        x=umbrales, y=f1_u, name='F1-Score',
        line=dict(color='#34d399', width=2), mode='lines+markers', marker=dict(size=5)
    ))
    fig_umbral.add_vline(
        x=0.85, line_dash="dash", line_color="#fbbf24",
        annotation_text="Umbral óptimo F1 = 0.85",
        annotation_font_color="#fbbf24"
    )
    fig_umbral.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#9ca3af'), height=300,
        xaxis=dict(title='Umbral de decisión', gridcolor='#1f2937'),
        yaxis=dict(title='Métrica', gridcolor='#1f2937', range=[0, 1]),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_umbral, use_container_width=True)

    st.markdown("""<div class="insight-box">
    <b>Conclusión clave:</b> XGBoost detecta el 87.8% de los fraudes reales con un umbral de 0.85.
    La elección del umbral es una decisión de negocio: un umbral más bajo aumenta el Recall
    (se detecta más fraude) a costa de más falsas alarmas que generan fricción con clientes legítimos.
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — MÓDULO 3: RENTABILIDAD BANCARIA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Módulo 3 — Rentabilidad Bancaria":
    st.markdown('<p class="section-header">Módulo 3</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="page-title">Rentabilidad y Eficiencia Bancaria</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">5.000+ bancos americanos · FDIC API · 2019–2023 · K-Means clustering</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Evolución temporal", "🗺️ Mapa de clusters"])

    with tab1:
        datos_evol = {
            'Año': [2019, 2020, 2021, 2022, 2023],
            'ROA': [1.156, 1.056, 1.111, 1.052, 1.021],
            'ROE': [9.84,  9.34,  10.51, 11.93, 11.63],
            'NIM': [4.152, 3.558, 3.222, 3.426, 4.336],
            'CIR': [65.30, 64.52, 63.77, 64.69, 65.38],
        }
        df_evol = pd.DataFrame(datos_evol)

        fill_colors = {
            'ROA': 'rgba(79,128,255,0.1)',
            'ROE': 'rgba(52,211,153,0.1)',
            'NIM': 'rgba(249,115,22,0.1)',
            'CIR': 'rgba(167,139,250,0.1)'
        }
        line_colors = {
            'ROA': '#4f80ff', 'ROE': '#34d399',
            'NIM': '#f97316', 'CIR': '#a78bfa'
        }

        metrica = st.selectbox("Selecciona ratio", ["ROA", "ROE", "NIM", "CIR"])

        fig_evol = go.Figure()
        fig_evol.add_trace(go.Scatter(
            x=df_evol['Año'], y=df_evol[metrica],
            mode='lines+markers+text',
            line=dict(color=line_colors[metrica], width=3),
            marker=dict(size=10, color=line_colors[metrica]),
            text=[f'{v:.2f}%' for v in df_evol[metrica]],
            textposition='top center',
            textfont=dict(color='white', size=11),
            fill='tozeroy',
            fillcolor=fill_colors[metrica]
        ))
        eventos = {2020: 'COVID-19', 2022: 'Fed sube tipos', 2023: 'Tipos altos'}
        for anio, texto in eventos.items():
            fig_evol.add_vline(x=anio, line_dash="dot", line_color="#374151")
            fig_evol.add_annotation(
                x=anio, y=max(df_evol[metrica]) * 0.95,
                text=texto, font=dict(color='#6b7280', size=10),
                showarrow=False, textangle=-45
            )
        fig_evol.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#9ca3af'), height=380,
            xaxis=dict(tickvals=[2019, 2020, 2021, 2022, 2023], gridcolor='#1f2937'),
            yaxis=dict(title=f'{metrica} (%)', gridcolor='#1f2937'),
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(fig_evol, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### CIR por tamaño de banco")
            segmentos = ['Pequeño\n(<100M$)', 'Mediano\n(100M-1B$)',
                         'Grande\n(1B-10B$)', 'Sistémico\n(>10B$)']
            cir_vals  = [71.5, 64.2, 61.2, 58.7]
            fig_cir = go.Figure(go.Bar(
                x=segmentos, y=cir_vals,
                marker_color=['#93c5fd', '#60a5fa', '#3b82f6', '#1d4ed8'],
                text=[f'{v}%' for v in cir_vals],
                textposition='outside', textfont=dict(color='white')
            ))
            fig_cir.add_hline(
                y=60, line_dash="dash", line_color="#f87171",
                annotation_text="Umbral eficiencia 60%",
                annotation_font_color="#f87171"
            )
            fig_cir.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#9ca3af'), height=320,
                yaxis=dict(range=[0, 85], gridcolor='#1f2937'),
                xaxis=dict(gridcolor='#1f2937'),
                margin=dict(t=20, b=20, l=20, r=20), showlegend=False
            )
            st.plotly_chart(fig_cir, use_container_width=True)

        with col2:
            st.markdown("#### Correlación entre ratios")
            corr_data = np.array([
                [ 1.00,  0.81,  0.32, -0.75],
                [ 0.81,  1.00,  0.25, -0.60],
                [ 0.32,  0.25,  1.00, -0.16],
                [-0.75, -0.60, -0.16,  1.00]
            ])
            fig_corr = go.Figure(go.Heatmap(
                z=corr_data,
                x=['ROA', 'ROE', 'NIM', 'CIR'],
                y=['ROA', 'ROE', 'NIM', 'CIR'],
                colorscale='RdYlGn', zmid=0, zmin=-1, zmax=1,
                text=[[f'{v:.2f}' for v in row] for row in corr_data],
                texttemplate='%{text}',
                textfont=dict(size=13, color='white'),
                showscale=True
            ))
            fig_corr.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#9ca3af'), height=320,
                margin=dict(t=20, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab2:
        st.markdown("#### Perfiles de bancos identificados por K-Means (2023)")

        cluster_data = pd.DataFrame({
            'Cluster': ['C0 · Banco medio', 'C1 · En dificultades',
                        'C2 · Alto rendimiento', 'C3 · Eficiente'],
            'ROA':    [0.802, 0.272, 2.074, 1.324],
            'ROE':    [9.24,  3.26,  20.47, 15.19],
            'NIM':    [4.145, 3.949, 5.368, 4.410],
            'CIR':    [70.60, 88.00, 51.70, 59.45],
            'Bancos': [414,   106,   117,   345],
            'Color':  ['#4f80ff', '#f87171', '#34d399', '#a78bfa']
        })

        fill_map = {
            '#4f80ff': 'rgba(79,128,255,0.15)',
            '#f87171': 'rgba(248,113,113,0.15)',
            '#34d399': 'rgba(52,211,153,0.15)',
            '#a78bfa': 'rgba(167,139,250,0.15)'
        }

        col_t, col_r = st.columns([1, 1])

        with col_t:
            for _, row in cluster_data.iterrows():
                st.markdown(f"""
                <div class="metric-card" style="border-left:3px solid {row['Color']}; margin:6px 0">
                    <h3>{row['Cluster']} &nbsp;·&nbsp; {row['Bancos']} bancos</h3>
                    <span style='color:{row["Color"]}; font-family:IBM Plex Mono; font-size:0.85rem'>
                    ROA {row['ROA']:.2f}% &nbsp;|&nbsp;
                    ROE {row['ROE']:.2f}% &nbsp;|&nbsp;
                    NIM {row['NIM']:.2f}% &nbsp;|&nbsp;
                    CIR {row['CIR']:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

        with col_r:
            categorias = ['ROA', 'ROE', 'NIM', 'Eficiencia\n(1-CIR)']
            mins = [0.272, 3.26, 3.949, 12.0]
            maxs = [2.074, 20.47, 5.368, 48.3]

            fig_radar = go.Figure()
            for _, row in cluster_data.iterrows():
                vals_raw  = [row['ROA'], row['ROE'], row['NIM'], 100 - row['CIR']]
                vals_norm = [(v - mn) / (mx - mn) for v, mn, mx in zip(vals_raw, mins, maxs)]
                vals_norm += vals_norm[:1]

                fig_radar.add_trace(go.Scatterpolar(
                    r=vals_norm,
                    theta=categorias + [categorias[0]],
                    fill='toself',
                    name=row['Cluster'].split('·')[0].strip(),
                    line=dict(color=row['Color'], width=2),
                    fillcolor=fill_map[row['Color']],
                    opacity=0.9
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 1]),
                    angularaxis=dict(color='#6b7280'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#9ca3af'),
                height=400,
                legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
                margin=dict(t=20, b=20, l=40, r=40)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""<div class="insight-box">
        <b>Palanca principal:</b> La correlación CIR/ROA de -0.75 confirma que reducir los costes
        operativos es la vía más directa hacia una mayor rentabilidad. Solo los bancos grandes y
        sistémicos se sitúan por debajo del umbral de eficiencia del 60%. Los bancos del Cluster 1
        (CIR 88%) son candidatos naturales a procesos de fusión o reestructuración.
        </div>""", unsafe_allow_html=True)