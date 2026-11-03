import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="Inventario de mi cuarto", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Architects+Daughter&display=swap');
.stApp {
    background-color: #0b3d91;
    background-image: linear-gradient(rgba(255,255,255,0.08) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.08) 1px, transparent 1px);
    background-size: 28px 28px;
}
[data-testid="stHeader"] {
    background: transparent;
}
section[data-testid="stSidebar"] {
    background-color: #082a66;
}
.stApp p, .stApp label, .stApp li {
    color: #ffffff;
}
h1, h2, h3 {
    font-family: 'Architects Daughter', cursive;
    color: #ffd23f !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: #ffffff !important;
}
[data-testid="stAlert"] {
    background-color: #082a66 !important;
    border: 1px solid #ffd23f;
}
[data-testid="stAlert"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #082a66 !important;
    border: 1px dashed #ffd23f;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #ffffff !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: #ffd23f !important;
    color: #082a66 !important;
}
div.stButton > button, div.stDownloadButton > button {
    background-color: #ffd23f;
    color: #082a66 !important;
    border: none;
    border-radius: 6px;
    font-weight: bold;
    padding: 8px 24px;
}
</style>
""", unsafe_allow_html=True)

nombres = ["persona", "bicicleta", "carro", "moto", "avion", "bus", "tren", "camion", "bote", "semaforo",
           "hidrante", "senal de pare", "parquimetro", "banca", "pajaro", "gato", "perro", "caballo", "oveja", "vaca",
           "elefante", "oso", "cebra", "jirafa", "mochila", "paraguas", "bolso", "corbata", "maleta", "frisbee",
           "esquis", "snowboard", "balon", "cometa", "bate de beisbol", "guante de beisbol", "patineta", "tabla de surf",
           "raqueta de tenis", "botella", "copa de vino", "taza", "tenedor", "cuchillo", "cuchara", "bol", "banano",
           "manzana", "sandwich", "naranja", "brocoli", "zanahoria", "perro caliente", "pizza", "dona", "torta",
           "silla", "sofa", "planta", "cama", "mesa", "inodoro", "tv", "portatil", "mouse", "control remoto",
           "teclado", "celular", "microondas", "horno", "tostadora", "lavamanos", "nevera", "libro", "reloj",
           "jarron", "tijeras", "oso de peluche", "secador de pelo", "cepillo de dientes"]


@st.cache_resource
def cargar_modelo():
    from ultralytics import YOLO
    return YOLO("yolov5su.pt")


with st.sidebar:
    st.subheader("Parámetros")
    confianza = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
    max_det = st.number_input("Detecciones máximas", 10, 1000, 100, 10)

st.title("📐 Inventario de mi cuarto")
st.write("Toma una foto de tu escritorio, tu cuarto o tu cocina, y la IA hace el inventario de lo que encuentra.")

try:
    with st.spinner("Cargando el modelo..."):
        modelo = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

modo = st.radio("¿De dónde viene la foto?", ["Tomar foto", "Subir imagen"], horizontal=True)

if modo == "Tomar foto":
    foto = st.camera_input("Toma la foto")
else:
    foto = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if foto is not None:
    imagen = ImageOps.exif_transpose(Image.open(io.BytesIO(foto.getvalue()))).convert("RGB")

    with st.spinner("Contando objetos..."):
        try:
            resultados = modelo(np.array(imagen)[..., ::-1], conf=confianza, iou=iou, max_det=int(max_det))
        except Exception as e:
            st.error(f"Error durante la detección: {e}")
            st.stop()

    r = resultados[0]
    if len(r.names) == 80:
        r.names = {i: n for i, n in enumerate(nombres)}
    anotada = r.plot()[:, :, ::-1]

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Foto con detecciones")
        st.image(anotada, use_container_width=True)
        buffer = io.BytesIO()
        Image.fromarray(anotada).save(buffer, format="PNG")
        st.download_button("Descargar imagen", buffer.getvalue(), file_name="inventario.png", mime="image/png")

    with col2:
        st.subheader("Inventario")
        cajas = r.boxes
        if cajas is not None and len(cajas) > 0:
            conteo = {}
            niveles = {}
            for caja in cajas:
                nombre = r.names[int(caja.cls.item())]
                conteo[nombre] = conteo.get(nombre, 0) + 1
                niveles.setdefault(nombre, []).append(float(caja.conf.item()))

            tabla = pd.DataFrame({
                "Objeto": list(conteo.keys()),
                "Cantidad": list(conteo.values()),
                "Confianza": [f"{np.mean(niveles[n]):.0%}" for n in conteo],
            }).sort_values("Cantidad", ascending=False)

            m1, m2 = st.columns(2)
            m1.metric("Objetos", len(cajas))
            m2.metric("Tipos distintos", len(conteo))
            st.write(f"Lo que más hay: **{tabla.iloc[0]['Objeto']}** ({tabla.iloc[0]['Cantidad']}).")
            st.dataframe(tabla, hide_index=True, use_container_width=True)
            st.bar_chart(tabla.set_index("Objeto")["Cantidad"])
        else:
            st.info("No encontré objetos con estos parámetros.")
            st.caption("Prueba a bajar la confianza mínima en la barra lateral.")

with st.expander("¿Qué puede detectar?"):
    st.write(", ".join(nombres))
