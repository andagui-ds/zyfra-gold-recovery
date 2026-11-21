import joblib  # Carga el paquete de modelos entrenados y metadatos de los sensores desde el archivo .joblib
# Construye el DataFrame de entrada con las 52 columnas en el orden exacto que espera el modelo
import pandas as pd
# Genera gráficas interactivas nativas sin necesidad de usar HTML o CSS
import plotly.express as px
import streamlit as st  # Construye la interfaz web interactiva de la aplicación

st.set_page_config(  # Configura el título de la pestaña y el modo ancho (wide) para que quepa en una sola pantalla
    page_title="Zyfra | Predicción de Recuperación de Oro",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Mantiene el modelo cargado en memoria caché para que cada movimiento de slider sea instantáneo
@st.cache_resource
def load_artifact():
    return joblib.load(
        "modelo_zyfra.joblib"
    )  # Lee el archivo ligero generado desde tu notebook


# Extrae el diccionario con los dos Random Forest y las estadísticas de los sensores
artefacto = load_artifact()
# Modelo Random Forest para la etapa de flotación primaria (Rougher)
model_rougher = artefacto["model_rougher"]
# Modelo Random Forest para la etapa de purificación (Final)
model_final = artefacto["model_final"]
# Lista exacta de columnas que usa el modelo Rougher
rougher_features = artefacto["rougher_features"]
# Lista exacta de columnas que usa el modelo Final
final_features = artefacto["final_features"]
# Valores medianos por defecto para los sensores no modificados en el panel
medianas = artefacto["medianas"]
# Percentil 5% de cada sensor para fijar límites realistas en los sliders
minimos = artefacto["minimos"]
# Percentil 95% de cada sensor para fijar límites realistas en los sliders
maximos = artefacto["maximos"]

# --- BARRA LATERAL: CONTROLES OPERATIVOS DE PLANTA ---
with st.sidebar:  # Agrupa los controles de entrada del operador en el panel lateral izquierdo
    # Encabezado de la sección de variables de entrada
    st.header("⚙️ Parámetros de Alimentación")

    # Nombre técnico exacto de la concentración de Oro en la materia prima
    col_au = "rougher.input.feed_au"
    # Nombre técnico exacto de la concentración de Plata en la materia prima
    col_ag = "rougher.input.feed_ag"
    # Nombre técnico exacto de la concentración de Plomo en la materia prima
    col_pb = "rougher.input.feed_pb"
    # Nombre técnico exacto del tamaño de partícula en alimentación
    col_size = "rougher.input.feed_size"
    # Nombre técnico exacto del flujo de alimentación del mineral
    col_rate = "rougher.input.feed_rate"
    # Nombre técnico del reactivo Xantato en flotación
    col_xanthate = "rougher.input.floatbank10_xanthate"

    feed_au = st.slider(  # Control deslizante para ajustar la ley de Oro (%) dentro del rango real de planta
        "Concentración Au (Oro %)",
        float(minimos[col_au]),
        float(maximos[col_au]),
        float(medianas[col_au]),
        step=0.1,
    )
    feed_ag = st.slider(  # Control deslizante para ajustar la concentración de Plata (%)
        "Concentración Ag (Plata %)",
        float(minimos[col_ag]),
        float(maximos[col_ag]),
        float(medianas[col_ag]),
        step=0.1,
    )
    feed_pb = st.slider(  # Control deslizante para ajustar la concentración de Plomo (%)
        "Concentración Pb (Plomo %)",
        float(minimos[col_pb]),
        float(maximos[col_pb]),
        float(medianas[col_pb]),
        step=0.1,
    )
    feed_size = st.slider(  # Control deslizante para el tamaño de partícula de molienda
        "Tamaño de partícula (feed_size)",
        float(minimos[col_size]),
        float(maximos[col_size]),
        float(medianas[col_size]),
        step=0.5,
    )
    feed_rate = st.slider(  # Control deslizante para la tasa de flujo de mineral en toneladas/hora
        "Flujo de alimentación (feed_rate)",
        float(minimos[col_rate]),
        float(maximos[col_rate]),
        float(medianas[col_rate]),
        step=5.0,
    )
    xanthate = st.slider(  # Control deslizante para la dosificación del reactivo químico Xantato
        "Reactivo Xantato (floatbank10)",
        float(minimos[col_xanthate]),
        float(maximos[col_xanthate]),
        float(medianas[col_xanthate]),
        step=0.1,
    )

# --- INFERENCIA EN TIEMPO REAL ---
# Parte de la operación mediana estable de todos los sensores de la planta
fila_simulacion = medianas.copy()
# Actualiza el sensor de Oro con el valor elegido por el usuario
fila_simulacion[col_au] = feed_au
# Actualiza el sensor de Plata con el valor elegido por el usuario
fila_simulacion[col_ag] = feed_ag
# Actualiza el sensor de Plomo con el valor elegido por el usuario
fila_simulacion[col_pb] = feed_pb
# Actualiza el sensor de tamaño de partícula con el valor del slider
fila_simulacion[col_size] = feed_size
# Actualiza el flujo de alimentación con el valor del slider
fila_simulacion[col_rate] = feed_rate
# Actualiza la dosis de reactivo con el valor del slider
fila_simulacion[col_xanthate] = xanthate

# Convierte el diccionario de sensores en una fila tabular de Pandas
df_input = pd.DataFrame([fila_simulacion])

pred_rougher = float(model_rougher.predict(df_input[rougher_features])[
                     0])  # Predice recuperación primaria (%)
pred_final = float(model_final.predict(df_input[final_features])[
                   0])  # Predice recuperación final purificada (%)

# Cálculo de referencia base (mediana de planta) para mostrar variación real (delta)
# Crea una fila base con la operación típica de la planta
df_base = pd.DataFrame([medianas])
base_rougher = float(model_rougher.predict(df_base[rougher_features])[
                     0])  # Recuperación Rougher estándar
base_final = float(model_final.predict(df_base[final_features])[
                   0])  # Recuperación Final estándar

# --- PANEL PRINCIPAL (UNA SOLA PANTALLA) ---
# Título principal del dashboard operativo
st.title("⛏️ Zyfra — Simulador de Recuperación de Oro en Planta")
st.caption(  # Subtítulo técnico con la arquitectura del modelo y enlace al reporte en Quarto
    "Predicción en dos etapas con **Random Forest** (sin Data Leakage) | "
    "Reporte completo en Quarto: andagui-ds.github.io/zyfra-gold-recovery"
)

# Divide la fila superior en 4 tarjetas de KPIs clave
m1, m2, m3, m4 = st.columns(4)
m1.metric(  # Muestra la recuperación estimada en flotación primaria y su diferencia contra el estándar
    "🧪 Recuperación Rougher",
    f"{pred_rougher:.2f} %",
    f"{pred_rougher - base_rougher:+.2f} % vs base",
)
m2.metric(  # Muestra la recuperación estimada en el concentrado final y su diferencia contra el estándar
    "🏆 Recuperación Final",
    f"{pred_final:.2f} %",
    f"{pred_final - base_final:+.2f} % vs base",
)
m3.metric(  # Muestra el sMAPE ponderado real obtenido en validación (25% Rougher + 75% Final)
    "🎯 sMAPE Ponderado (Test)",
    "7.72 %",
    "Meta < 8.0 % cumplida",
)
m4.metric(  # Muestra el aumento real de concentración de Oro a lo largo del proceso industrial
    "📈 Enriquecimiento Au",
    f"{feed_au:.1f}% → ~45.0%",
    "Etapa Feed a Final",
)

st.divider()  # Línea separadora nativa entre los KPIs y el área de gráficas

# Divide el espacio inferior en dos columnas para visualización
col_izq, col_der = st.columns([1.15, 1])

# Columna izquierda con la curva de sensibilidad real según la ley de Oro (Au)
with col_izq:
    # Título de la gráfica de sensibilidad
    st.subheader("📈 Sensibilidad: Recuperación Final vs. Ley de Oro (Au)")
    rango_au = [  # Genera 10 puntos equidistantes entre el mínimo y máximo real de Oro
        round(minimos[col_au] + i * (maximos[col_au] - minimos[col_au]) / 9, 2)
        for i in range(10)
    ]
    sim_rows = []  # Lista para acumular los 10 escenarios simulados
    for val in rango_au:
        row_tmp = fila_simulacion.copy()  # Copia los parámetros actuales del operador
        # Varía únicamente la concentración de Oro para medir su impacto
        row_tmp[col_au] = val
        sim_rows.append(row_tmp)
    # Convierte los 10 escenarios en un DataFrame
    df_sim = pd.DataFrame(sim_rows)
    df_curva = pd.DataFrame({  # Prepara la tabla de datos para graficar con Plotly
        "Concentración Au (%)": rango_au,
        "Recuperación Final (%)": model_final.predict(df_sim[final_features]),
        "Recuperación Rougher (%)": model_rougher.predict(df_sim[rougher_features]),
    })
    fig = px.line(  # Crea una gráfica de líneas interactiva comparando ambas etapas
        df_curva,
        x="Concentración Au (%)",
        y=["Recuperación Final (%)", "Recuperación Rougher (%)"],
        markers=True,
    )
    fig.update_layout(  # Ajusta altura y márgenes para evitar scroll vertical en pantalla
        height=310,
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="Etapa",
    )
    # Renderiza la gráfica responsiva en Streamlit
    st.plotly_chart(fig, use_container_width=True)

with col_der:  # Columna derecha con la comparativa de modelos evaluados en el proyecto
    # Título del gráfico comparativo de sMAPE
    st.subheader("🏅 Benchmarking de Modelos (5-Fold CV)")
    df_bench = pd.DataFrame({  # Tabla con los resultados exactos de validación cruzada de tu notebook
        "Modelo": ["Random Forest 🏆", "Decision Tree", "Linear Regression"],
        "sMAPE Ponderado (%)": [9.16, 9.80, 10.91],
    })
    fig2 = px.bar(  # Crea un gráfico de barras horizontales ordenadas por menor error
        df_bench.sort_values("sMAPE Ponderado (%)", ascending=False),
        x="sMAPE Ponderado (%)",
        y="Modelo",
        orientation="h",
        text="sMAPE Ponderado (%)",
        color="sMAPE Ponderado (%)",
        color_continuous_scale="Teal_r",
    )
    # Muestra el porcentaje sobre cada barra
    fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig2.update_layout(  # Ajusta altura y márgenes para mantener la pantalla compacta
        height=310,
        margin=dict(l=10, r=45, t=10, b=10),
        coloraxis_showscale=False,
    )
    # Renderiza la segunda gráfica en Streamlit
    st.plotly_chart(fig2, use_container_width=True)
