import joblib  # Carga el diccionario con los dos modelos Random Forest y las estadísticas de sensores
# Construye los DataFrames de entrada respetando el orden exacto de las columnas
import pandas as pd
# Dibuja las barras comparativas dinámicas de recuperación por etapa
import plotly.graph_objects as go
# Genera la gráfica de importancia de variables reales del modelo
import plotly.express as px
# Construye toda la interfaz web nativa sin necesidad de escribir CSS
import streamlit as st

st.set_page_config(  # Configura la pestaña del navegador y el modo ancho (wide) en una sola pantalla
    page_title="Zyfra | Simulador de Recuperación de Oro",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Guarda el modelo en caché para que cada movimiento de slider sea instantáneo
@st.cache_resource
def load_artifact():
    return joblib.load(
        "modelo_zyfra.joblib"
    )  # Lee el archivo ligero generado desde tu notebook


# Extrae los modelos entrenados y rangos reales de la planta
artefacto = load_artifact()
# Modelo Random Forest de flotación primaria (Rougher)
model_rougher = artefacto["model_rougher"]
# Modelo Random Forest de purificación (Final)
model_final = artefacto["model_final"]
# Columnas exactas requeridas por model_rougher
rougher_features = artefacto["rougher_features"]
# Columnas exactas requeridas por model_final
final_features = artefacto["final_features"]
# Valores medianos por defecto para los sensores no ajustados
medianas = artefacto["medianas"]
# Percentil 5% de cada variable para fijar límites realistas
minimos = artefacto["minimos"]
# Percentil 95% de cada variable para fijar límites realistas
maximos = artefacto["maximos"]

# --- BARRA LATERAL: CONTROLES OPERATIVOS DE PLANTA ---
with st.sidebar:  # Agrupa los controles operativos del ingeniero de planta en el panel izquierdo
    # Título de los controles de simulación
    st.header("⚙️ Control Operativo de Planta")

    # Ley de Oro (%) en la alimentación inicial
    col_au = "rougher.input.feed_au"
    # Flujo de carga mineral (ton/h) hacia flotación
    col_rate = "rougher.input.feed_rate"
    # Reactivo colector Xantato en flotación primaria
    col_xanthate = "rougher.input.floatbank10_xanthate"
    # Reactivo Sulfato en etapa de limpieza primaria
    col_sulfate = "primary_cleaner.input.sulfate"
    # Reactivo Depresor en limpieza primaria
    col_depressant = "primary_cleaner.input.depressant"
    # Concentración de Oro que entra a purificación
    col_p_au = "primary_cleaner.input.feed_au"

    feed_au = st.slider(  # Control deslizante para ajustar la concentración de Oro en materia prima
        "Oro en Alimentación (feed_au %)",
        float(minimos[col_au]),
        float(maximos[col_au]),
        float(medianas[col_au]),
        step=0.1,
    )
    feed_rate = st.slider(  # Control deslizante para la tasa de flujo de alimentación del mineral
        "Flujo de Carga (feed_rate)",
        float(minimos[col_rate]),
        float(maximos[col_rate]),
        float(medianas[col_rate]),
        step=5.0,
    )
    xanthate = st.slider(  # Control deslizante para el reactivo Xantato en flotación Rougher
        "Reactivo Xantato (Rougher)",
        float(minimos[col_xanthate]),
        float(maximos[col_xanthate]),
        float(medianas[col_xanthate]),
        step=0.1,
    )
    p_au = st.slider(  # Control deslizante para la ley de Oro entrando a limpieza primaria
        "Oro en Limpieza Primaria (primary_au %)",
        float(minimos[col_p_au]),
        float(maximos[col_p_au]),
        float(medianas[col_p_au]),
        step=0.2,
    )
    sulfate = st.slider(  # Control deslizante para la dosificación de Sulfato en purificación
        "Reactivo Sulfato (Primary Cleaner)",
        float(minimos[col_sulfate]),
        float(maximos[col_sulfate]),
        float(medianas[col_sulfate]),
        step=2.0,
    )
    depressant = st.slider(  # Control deslizante para el reactivo Depresor en purificación
        "Reactivo Depresor (Primary Cleaner)",
        float(minimos[col_depressant]),
        float(maximos[col_depressant]),
        float(medianas[col_depressant]),
        step=0.2,
    )

# --- INFERENCIA EN TIEMPO REAL ---
# Toma la operación mediana estable de los 52 sensores como base
fila_simulacion = medianas.copy()
fila_simulacion[col_au] = feed_au  # Asigna el valor del slider de Oro inicial
# Asigna el valor del slider de flujo de alimentación
fila_simulacion[col_rate] = feed_rate
# Asigna el valor del slider de Xantato
fila_simulacion[col_xanthate] = xanthate
# Asigna el valor del slider de Oro en limpieza primaria
fila_simulacion[col_p_au] = p_au
fila_simulacion[col_sulfate] = sulfate  # Asigna el valor del slider de Sulfato
# Asigna el valor del slider de Depresor
fila_simulacion[col_depressant] = depressant

# Convierte los 52 sensores en un DataFrame de una fila
df_input = pd.DataFrame([fila_simulacion])
pred_rougher = float(model_rougher.predict(df_input[rougher_features])[
                     0])  # Predice recuperación Rougher (%)
pred_final = float(model_final.predict(df_input[final_features])[
                   0])  # Predice recuperación Final (%)

# Crea la fila de referencia estándar (medianas de planta)
df_base = pd.DataFrame([medianas])
base_rougher = float(model_rougher.predict(df_base[rougher_features])[
                     0])  # Recuperación Rougher estándar
base_final = float(model_final.predict(df_base[final_features])[
                   0])  # Recuperación Final estándar

# --- PANEL PRINCIPAL (UNA SOLA PANTALLA) ---
# Encabezado ejecutivo nativo
st.header("⛏️ Zyfra — Simulador de Recuperación de Oro en Planta")
st.caption(  # Subtítulo técnico con la arquitectura del modelo y enlace al reporte de Quarto
    "Inferencia en dos etapas con **Random Forest** (sin Data Leakage) | "
    "Reporte en Quarto: andagui-ds.github.io/zyfra-gold-recovery"
)

# Crea 4 columnas superiores para los indicadores KPI
m1, m2, m3, m4 = st.columns(4)
m1.metric(  # Muestra la recuperación primaria calculada por el modelo y su variación contra la base
    "🧪 Recuperación Rougher",
    f"{pred_rougher:.2f} %",
    f"{pred_rougher - base_rougher:+.2f} % vs base",
)
m2.metric(  # Muestra la recuperación final calculada por el modelo y su variación contra la base
    "🏆 Recuperación Final",
    f"{pred_final:.2f} %",
    f"{pred_final - base_final:+.2f} % vs base",
)
m3.metric(  # Muestra el error sMAPE ponderado real validado en el conjunto de prueba
    "🎯 sMAPE Ponderado (Test)",
    "7.72 %",
    "Meta < 8.0 % cumplida",
)
m4.metric(  # Muestra el salto de concentración de Oro desde materia prima hasta concentrado final
    "📈 Enriquecimiento Au",
    f"{feed_au:.1f}% → ~45.0%",
    "Feed a Concentrado Final",
)

st.divider()  # Línea divisoria nativa entre los indicadores y las gráficas

# Divide la mitad inferior en dos paneles de visualización
col_izq, col_der = st.columns([1.15, 1])

# Panel izquierdo: Comparativa dinámica en vivo (Simulación Actual vs. Operación Base)
with col_izq:
    # Título de la gráfica dinámica
    st.subheader("📊 Eficiencia de Recuperación: Simulación Actual vs. Base")
    fig = go.Figure()  # Inicializa gráfica de barras agrupadas interactiva
    fig.add_trace(go.Bar(  # Barras de referencia estándar de la planta (mediana histórica)
        x=["1. Flotación Primaria (Rougher)",
           "2. Concentrado Purificado (Final)"],
        y=[base_rougher, base_final],
        name="Operación Base (Mediana)",
        text=[f"{base_rougher:.2f}%", f"{base_final:.2f}%"],
        textposition="outside",
        marker_color="#64748B",
    ))
    fig.add_trace(go.Bar(  # Barras en vivo que cambian con los sliders del operador
        x=["1. Flotación Primaria (Rougher)",
           "2. Concentrado Purificado (Final)"],
        y=[pred_rougher, pred_final],
        name="Simulación Actual (Sliders)",
        text=[f"{pred_rougher:.2f}%", f"{pred_final:.2f}%"],
        textposition="outside",
        marker_color="#0EA5E9",
    ))
    fig.update_layout(  # Configura barras agrupadas y escala fija de 0 a 105% para máxima claridad
        barmode="group",
        yaxis_title="Recuperación de Oro (%)",
        yaxis_range=[0, 105],
        height=305,
        margin=dict(l=10, r=10, t=15, b=10),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
    )
    # Renderiza la gráfica responsiva
    st.plotly_chart(fig, use_container_width=True)

with col_der:  # Panel derecho: Variables reales más influyentes extraídas del modelo Random Forest
    # Título de importancia de sensores
    st.subheader("🔍 Top Variables Más Influyentes (Random Forest)")
    df_imp = (  # Extrae las 6 variables con mayor peso real del modelo final entrenado
        pd.DataFrame({
            "Sensor": final_features,
            "Importancia (%)": model_final.feature_importances_ * 100,
        })
        .sort_values("Importancia (%)", ascending=True)
        .tail(6)
    )
    fig2 = px.bar(  # Dibuja barras horizontales con los sensores que realmente gobiernan el modelo
        df_imp,
        x="Importancia (%)",
        y="Sensor",
        orientation="h",
        text="Importancia (%)",
        color="Importancia (%)",
        color_continuous_scale="Teal",
    )
    # Etiqueta el % de peso
    fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig2.update_layout(  # Ajusta márgenes y rango para evitar recortes de etiquetas
        xaxis_range=[0, df_imp["Importancia (%)"].max() * 1.22],
        height=305,
        margin=dict(l=10, r=20, t=15, b=10),
        coloraxis_showscale=False,
    )
    # Renderiza el gráfico de importancia
    st.plotly_chart(fig2, use_container_width=True)
