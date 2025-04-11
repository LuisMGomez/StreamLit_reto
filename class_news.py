import streamlit as st
import joblib
import pandas as pd
import csv

@st.cache_resource
def load_model():
    return joblib.load("clasificador_temas_es.pkl")

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
    resultados = []
    for text in df[opcion]:
        model_outputs = modelo.predict([text])
        resultados.append(model_outputs[0])  # Ajustado para evitar TypeError
    df['prediccion'] = resultados
    return df

def main():
    st.title("Clasificador de Noticias")
    
    uploaded_file = st.file_uploader("Elige un fichero CSV", type="csv")
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
                df_resultado = procesar(df, opcion, modelo)
                st.success("✅ ¡Clasificación completada!")
                st.dataframe(df_resultado)

                # Descargar resultados
                csv = df_resultado.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Descargar resultados", data=csv, file_name="resultados.csv", mime="text/csv")

main()

