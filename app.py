import pandas as pd
import streamlit as st
import joblib
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
        delimitador = ','
    uploaded_file.seek(0)
    return delimitador


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

    st.subheader(f"Conteo de predicciones ( Filtro: {categoria_filtro} )")
    st.dataframe(conteo)

    fig = px.bar(conteo, x=columna, y='Cantidad', color=columna, text='Cantidad')
    st.plotly_chart(fig, use_container_width=True)


def validar_file(file):
    if "archivo" in st.session_state and file != st.session_state.archivo and "df_resultado" in st.session_state:
        st.session_state.pop("df_resultado")
    st.session_state.archivo = file

# Predicciones del modelo 
def procesar(df, opcion, modelo):
    # Cargar el LabelEncoder
    le = joblib.load("modelo_transformers/label_encoder.pkl")  

    resultados = []
    for text in df[opcion]:

        # Obtener la predicción del modelo
        model_outputs = modelo(text)  
        
        # Obtener la etiqueta (label) del resultado y convertirla en texto con el LabelEncoder
        prediccion_numerica = int(model_outputs[0]['label'].split('_')[1])  
        
        # Convertir la etiqueta numérica a texto
        etiqueta_texto = le.inverse_transform([prediccion_numerica])[0]
        
        resultados.append(etiqueta_texto)  

    # Añadir la predicción al DataFrame
    df['prediccion'] = resultados 
     
    return df


def set_bg_hack_url():
    st.markdown("""
        <style>
        .stApp {
            background: 
                linear-gradient(
                    to right,
                    rgba(255, 255, 255, 0.0) 0%,
                    rgba(255, 255, 255, 0.4) 15%,
                    rgba(255, 255, 255, 0.85) 20%,
                    rgba(255, 255, 255, 1.0) 30%,
                    rgba(255, 255, 255, 1.0) 70%,
                    rgba(255, 255, 255, 0.85) 80%,
                    rgba(255, 255, 255, 0.4) 85%,
                    rgba(255, 255, 255, 0.0) 100%
                ),
                url("https://images.unsplash.com/photo-1504711434969-e33886168f5c?fit=crop&w=1600&q=80") no-repeat center center fixed;
            background-size: cover;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    st.title("Clasificador de Noticias")
    
    set_bg_hack_url()

    # Carga del archivo CSV
    uploaded_file = st.file_uploader("Elige un archivo CSV:", type="csv")

    if uploaded_file is not None:
        validar_file(uploaded_file)
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
            st.error(f"❌ Error al leer el archivo, el formato no es correcto. ")
            return
        
        st.header("Previsualización del archivo")
        st.dataframe(df)
        
        opcion = st.selectbox("Elige la columna del texto a procesar:", df.columns)
        if st.button("Predecir!"):
            with st.spinner("Clasificando..."):
                modelo = load_model()
                df_resultado = procesar(df.copy(), opcion, modelo)
                # Guarda en sesión
                st.session_state.df_resultado = df_resultado[[opcion, "prediccion"]]

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
            st.subheader(f"Resultados de la clasificación ( {categoria_filtro} )")
            st.dataframe(df_filtrado)

            # Mostrar gráfico filtrado
            mostrar_distribucion_por_categoria(df_filtrado, 'prediccion',categoria_filtro)

            # Descargar resultados filtrados
            csv = df_filtrado.to_csv(index=False).encode('utf-8')
            nombre_archivo = f"resultados_{categoria_filtro}.csv" if categoria_filtro != "Todas" else "resultados_completos.csv"
            st.download_button("📥 Descargar resultados filtrados", data=csv, file_name=nombre_archivo, mime="text/csv")
    else:
        st.warning("No se ha cargado ningún archivo.")
        st.button("Procesar!", disabled=True)

main()