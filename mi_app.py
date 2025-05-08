import streamlit as st
import joblib
import pandas as pd
import csv
import plotly.express as px
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="modelo_transformers", tokenizer="modelo_transformers")

def detectar_delimitador(uploaded_file):
    contenido = uploaded_file.read().decode('utf-8')
    try:
        dialect = csv.Sniffer().sniff(contenido.splitlines()[0])
        delimitador = dialect.delimiter
    except Exception:
        delimitador = ','  # Fallback
    uploaded_file.seek(0)
    return delimitador

def procesar(df, opcion, modelo):
    # Cargar el LabelEncoder
    le = joblib.load("modelo_transformers/label_encoder.pkl")  # Cargar el encoder

    resultados = []
    for text in df[opcion]:
        model_outputs = modelo(text)  # Obtener la predicción del modelo
        # Obtener la etiqueta (label) del resultado y convertirla en texto con el LabelEncoder
        prediccion_numerica = int(model_outputs[0]['label'].split('_')[1])  # Extraer el número de la etiqueta
        etiqueta_texto = le.inverse_transform([prediccion_numerica])[0]  # Convertir la etiqueta numérica a texto
        resultados.append(etiqueta_texto)  # Guardar la etiqueta en texto

    df['prediccion'] = resultados  # Añadir la predicción al DataFrame
    return df

def mostrar_distribucion_por_categoria(df: pd.DataFrame, columna: str, categoria_filtro: str = "Todas"):
    if columna not in df.columns:
        st.error(f"La columna '{columna}' no existe en el DataFrame.")
        return

    if categoria_filtro != "Todas":
        df = df[df[columna] == categoria_filtro]

    conteo = df[columna].value_counts().reset_index()
    conteo.columns = [columna, 'Cantidad']

    if conteo.empty:
        st.warning("No hay datos para mostrar.")
        return

    st.subheader(f"Conteo en '{columna}' (Filtro: {categoria_filtro})")
    st.dataframe(conteo)

    fig = px.bar(conteo, x=columna, y='Cantidad', color=columna, text='Cantidad')
    st.plotly_chart(fig, use_container_width=True)

def set_bg_hack_url():
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(rgba(255,255,255,0.8), rgba(255,255,255,0.9)),
                        url("https://images.unsplash.com/photo-1504711434969-e33886168f5c?fit=crop&w=1600&q=80") no-repeat center center fixed;
            background-size: cover;
        }
                
        </style>
    """, unsafe_allow_html=True)

def main():
    st.title("Clasificador de Noticias")

    set_bg_hack_url()

    uploaded_file = st.file_uploader(" Elige un fichero CSV", type="csv")
    if uploaded_file is not None:
        st.subheader("Configuración de lectura del archivo")
        tipo_delimitador = st.selectbox("Selecciona el delimitador del archivo", [
            "Detectar automáticamente", ", (coma)", "; (punto y coma)"
        ])

        # Determinar delimitador según selección
        if tipo_delimitador == "Detectar automáticamente":
            delimitador = detectar_delimitador(uploaded_file)
        elif tipo_delimitador == ", (coma)":
            delimitador = ','
        else:
            delimitador = ';'

        try:
            df = pd.read_csv(uploaded_file, delimiter=delimitador)
        except Exception as e:
            st.error(f"❌ Error al leer el archivo: {e}")
            return

        st.header("Previsualización del archivo")
        st.dataframe(df.head(5))

        opcion = st.selectbox('Elige la columna con texto', df.columns)

        if st.button('Predecir'):
            with st.spinner("Clasificando..."):
                modelo = load_model()
                df_resultado = procesar(df.copy(), opcion, modelo)
                st.session_state.df_resultado = df_resultado  # Guarda en sesión
                st.success("✅ ¡Clasificación completada!")

        if 'df_resultado' in st.session_state:
            df_resultado = st.session_state.df_resultado

            # Selector de categoría para filtrar después de predecir
            categorias_disponibles = sorted(df_resultado['prediccion'].dropna().unique())
            categoria_filtro = st.selectbox("Filtrar categoría para visualización", ["Todas"] + categorias_disponibles)

            # Aplicar filtro
            if categoria_filtro != "Todas":
                df_filtrado = df_resultado[df_resultado['prediccion'] == categoria_filtro]
            else:
                df_filtrado = df_resultado

            # Mostrar resultados filtrados
            st.subheader("Resultados de la clasificación (filtrados)")
            st.dataframe(df_filtrado)

            # Mostrar gráfico filtrado
            mostrar_distribucion_por_categoria(df_filtrado, 'prediccion')

            # Descargar resultados filtrados
            csv = df_filtrado.to_csv(index=False).encode('utf-8')
            nombre_archivo = f"resultados_{categoria_filtro}.csv" if categoria_filtro != "Todas" else "resultados_completos.csv"
            st.download_button("📥 Descargar resultados filtrados", data=csv, file_name=nombre_archivo, mime="text/csv")

main()
