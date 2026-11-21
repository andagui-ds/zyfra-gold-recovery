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

# --- SELECCIÓN AUTOMÁTICA DE LOS SENSORES MÁS INFLUYENTES DEL MODELO ---
imp_series = pd.Series(model_final.feature_importances_, index=final_features).sort_values(
    ascending=False)  # Ordena variables por peso real
top_sensores = [c for c in imp_series.index if c in minimos and maximos[c]
                > minimos[c]][:6]  # Toma los 6 sensores con mayor impacto

# --- BARRA LATERAL: CONTROLES OPERATIVOS DE PLANTA ---
with st.sidebar:  # Agrupa los controles operativos del ingeniero de planta en el panel izquierdo
    # Título de los controles de simulación
    st.header("⚙️ Control Operativo de Planta")
    # Subtítulo explicativo
    st.caption("Sensores de mayor impacto real en el modelo:")

    valores_usuario = {}  # Diccionario para almacenar los valores ajustados en los sliders
    for sensor in top_sensores:  # Genera un slider automático por cada sensor crítico
        # Formatea el nombre técnico para lectura clara
        etiqueta = sensor.replace(".", " › ").replace("_", " ").title()
        # Límite inferior realista (percentil 5%)
        val_min = float(minimos[sensor])
        # Límite superior realista (percentil 95%)
        val_max = float(maximos[sensor])
        # Valor inicial por defecto (mediana de planta)
        val_med = float(medianas[sensor])
        # Calcula incremento suave proporcional al rango
        paso = round((val_max - val_min) / 50, 3) or 0.01
        valores_usuario[sensor] = st.slider(  # Renderiza el control deslizante nativo
            etiqueta,
            min_value=val_min,
            max_value=val_max,
            value=val_med,
            step=paso,
        )

# --- INFERENCIA EN TIEMPO REAL ---
# Parte de la operación mediana estable de todos los sensores
fila_simulacion = medianas.copy()
# Actualiza únicamente los sensores modificados por el usuario
fila_simulacion.update(valores_usuario)

# Convierte los sensores en un DataFrame de una fila
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

# Obtiene la ley de Oro inicial para mostrar en KPI
feed_au_actual = fila_simulacion.get("rougher.input.feed_au", 8.5)

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
    f"{feed_au_actual:.1f}% → ~45.0%",
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
