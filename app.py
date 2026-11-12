import streamlit as st

st.title("⛏️ Zyfra - Predicción de Recuperación de Oro")
st.markdown("Ajusta los parámetros de entrada del proceso químico para predecir la recuperación.")

with st.sidebar:
    st.header("Parámetros de Entrada")
    feed_au = st.slider("Concentración Au (Oro) - Alimentación", 0.0, 20.0, 8.5)
    feed_ag = st.slider("Concentración Ag (Plata) - Alimentación", 0.0, 20.0, 10.0)
    feed_pb = st.slider("Concentración Pb (Plomo) - Alimentación", 0.0, 20.0, 5.0)
    feed_size = st.number_input("Tamaño de partícula (mm)", 0.0, 100.0, 50.0)

if st.button("Predecir Recuperación"):
    # Simulación de salida del modelo
    st.subheader("Resultados de la Predicción")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Recuperación Rougher", "88.5%", "↑ 1.2%")
    col2.metric("Recuperación Final", "91.2%", "↑ 0.8%")
    col3.metric("sMAPE Final", "7.72%", "Excelente", delta_color="inverse")
    
    st.write("El modelo **Random Forest** se entrenó con 16,860 registros procesados.")
