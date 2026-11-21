import joblib  # Carga el diccionario con los dos modelos Random Forest y las estadísticas de sensores
# Construye los DataFrames de entrada respetando el orden exacto de las columnas
import pandas as pd
# Permite dibujar la curva de sensibilidad y el marcador del punto actual
import plotly.graph_objects as go
# Genera la gráfica de barras comparativa de los modelos evaluados
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

# --- BARRA LATERAL: VARIABLES CLAVE DE FLOTACIÓN Y PURIFICACIÓN ---
with st.sidebar:  # Agrupa los controles operativos del ingeniero de planta en el panel izquierdo
    # Título de los controles de simulación
    st.header("⚙️ Control Operativo de Planta")

    # Variables críticas que impactan tanto la etapa Rougher como la etapa Final
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
    p_au = st.slider(  # Control deslizante para la ley de Oro entrando a limpieza primaria (impacta Final)
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
# Toma la operación mediana estable de los 52 sensores como punto de partida
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
# Encabezado principal de la app
st.title("⛏️ Zyfra — Simulador de Recuperación de Oro en Planta")
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
col_izq, col_der = st.columns([1.2, 1])

with col_izq:  # Panel izquierdo con curva de sensibilidad interactiva y punto operativo en vivo
    variables_sensibilidad = {  # Diccionario para elegir qué variable analizar en el eje X
        "Oro en Limpieza Primaria (primary_au %)": col_p_au,
        "Reactivo Sulfato (Primary Cleaner)": col_sulfate,
        "Flujo de Carga (feed_rate)": col_rate,
        "Reactivo Xantato (Rougher)": col_xanthate,
        "Oro en Alimentación (feed_au %)": col_au,
    }
    var_label = st.selectbox(  # Selector nativo para cambiar la variable del eje X en la curva
        "📈 Curva de Sensibilidad en Vivo — Selecciona variable a barrer:",
        list(variables_sensibilidad.keys()),
    )
    # Obtiene el nombre técnico de la columna seleccionada
    var_col = variables_sensibilidad[var_label]

    rango_vals = [  # Genera 15 pasos equidistantes entre el mínimo y máximo real de esa variable
        round(minimos[var_col] + i *
              (maximos[var_col] - minimos[var_col]) / 14, 2)
        for i in range(15)
    ]
    sim_rows = []  # Acumula los 15 escenarios para predecir toda la curva de una sola vez
    for val in rango_vals:
        row_tmp = fila_simulacion.copy()  # Mantiene fijos los demás sliders del operador
        # Varía únicamente el parámetro seleccionado en el eje X
        row_tmp[var_col] = val
        sim_rows.append(row_tmp)
    # Convierte los 15 escenarios en un DataFrame
    df_sim = pd.DataFrame(sim_rows)

    # Predice la curva completa de Recuperación Final
    y_curve_final = model_final.predict(df_sim[final_features])
    # Predice la curva de Recuperación Rougher
    y_curve_rougher = model_rougher.predict(df_sim[rougher_features])

    fig = go.Figure()  # Inicializa una figura interactiva de Plotly
    fig.add_trace(go.Scatter(  # Dibuja la línea de Recuperación Rougher
        x=rango_vals, y=y_curve_rougher, mode="lines+markers", name="Recuperación Rougher (%)"
    ))
    fig.add_trace(go.Scatter(  # Dibuja la línea de Recuperación Final
        x=rango_vals, y=y_curve_final, mode="lines+markers", name="Recuperación Final (%)"
    ))
    fig.add_trace(go.Scatter(  # Coloca un punto destacado con tu configuración actual del slider
        x=[fila_simulacion[var_col], fila_simulacion[var_col]],
        y=[pred_rougher, pred_final],
        mode="markers",
        marker=dict(size=12, color="red", symbol="diamond"),
        name="Punto Operativo Actual",
    ))
    fig.update_layout(  # Ajusta altura y márgenes para que no exista scroll vertical
        xaxis_title=var_label,
        yaxis_title="Recuperación Predicha (%)",
        height=275,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
    )
    # Renderiza la gráfica responsiva
    st.plotly_chart(fig, use_container_width=True)

with col_der:  # Panel derecho con el contraste de modelos evaluados en el proyecto
    # Título de la comparativa de algoritmos
    st.subheader("🏅 Benchmarking de Modelos (5-Fold CV)")
    st.caption(
        "Comparativa histórica de validación cruzada que justifica el modelo en producción:")
    df_bench = pd.DataFrame({  # Resultados exactos de validación cruzada obtenidos en el notebook
        "Modelo": ["Random Forest 🏆", "Decision Tree", "Linear Regression"],
        "sMAPE Ponderado (%)": [9.16, 9.80, 10.91],
    })
    fig2 = px.bar(  # Dibuja barras horizontales ordenadas por menor error sMAPE
        df_bench.sort_values("sMAPE Ponderado (%)", ascending=False),
        x="sMAPE Ponderado (%)",
        y="Modelo",
        orientation="h",
        text="sMAPE Ponderado (%)",
        color="sMAPE Ponderado (%)",
        color_continuous_scale="Teal_r",
    )
    # Etiqueta cada barra con su %
    fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig2.update_layout(  # Mantiene la misma altura compacta de una sola pantalla
        height=275,
        margin=dict(l=10, r=45, t=10, b=10),
        coloraxis_showscale=False,
    )
    # Renderiza el gráfico comparativo
    st.plotly_chart(fig2, use_container_width=True)
