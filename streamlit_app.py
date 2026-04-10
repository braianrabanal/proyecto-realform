import os
import io
from pathlib import Path

import requests
from PIL import Image
import streamlit as st


"""

Flujo de trabajo:

- Botón "Capturar imagen"
- Muestra la última imagen capturada
- Botón "Predecir última imagen" → Analiza la imagen y muestra las predicciones
- Botón "Predecir todas las imágenes guardadas" → En la carpeta images, analiza todas las imagenes y muestra las predicciones en annotated_images.

"""


st.set_page_config(page_title=" PROYECTO REALFORM - Cámara con YOLO", layout="wide")
st.title(" REALFORM - 🥤 Deteccion de Tapones o Vasos 🍾 ")
#st.subheader("☢️ Realizado por Braian, supervisado por Ismael ☢️")
st.subheader("==========================================================================================================")


# --- Configuración de endpoints ---

st.sidebar.header("Configuración de APIs")
capture_base_url = st.sidebar.text_input(
    "URL servicio captura: \n\n (default: http://localhost:8001)",
    value="http://localhost:8001",
    help="Normalmente http://localhost:8001",
)
predict_base_url = st.sidebar.text_input(
    "URL servicio predicción: \n\n (default: http://localhost:8002)",
    value="http://localhost:8002",
    help="Normalmente http://localhost:8002",
)
model_info_url = f"{predict_base_url.rstrip('/')}/model_info"

capture_url = f"{capture_base_url.rstrip('/')}/capture"
predict_from_saved_url = f"{predict_base_url.rstrip('/')}/predict_from_saved"
predict_from_saved_annotated_url = (
    f"{predict_base_url.rstrip('/')}/predict_from_saved_annotated"
)
predict_all_saved_url = f"{predict_base_url.rstrip('/')}/predict_all_saved"

st.sidebar.markdown("---")
if st.sidebar.button("Consultar modelo de inferencia"):
    try:
        model_resp = requests.get(model_info_url, timeout=10)
    except Exception as e:
        st.sidebar.error(f"No se pudo consultar /model_info: {e}")
    else:
        if model_resp.status_code != 200:
            st.sidebar.error(
                f"Error al consultar modelo (status {model_resp.status_code}): {model_resp.text}"
            )
        else:
            model_data = model_resp.json()
            st.sidebar.success(
                f"Modelo activo: {model_data.get('model_filename', 'desconocido')}"
            )
            st.sidebar.json(model_data)

# Selector de confianza mínima para YOLO
st.sidebar.markdown("---")
confidence_threshold = st.sidebar.slider(
    "Confianza mínima YOLO",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.01,
    help="Solo se mostrarán detecciones con una confianza igual o superior a este valor.",
)

# Carpeta de imágenes en el host (montada por docker-compose)
HOST_IMAGES_DIR = Path("images")
HOST_ANNOTATED_DIR = Path("images_annotated")


if "last_filename" not in st.session_state:
    st.session_state["last_filename"] = None


col_left, col_right = st.columns(2)


# --- Columna izquierda: captura y visualización de imagen ---

with col_left:
    st.subheader("1. Captura de imagen desde la cámara")

    if st.button("📸 Capturar imagen desde cámara"):
        try:
            resp = requests.post(capture_url, timeout=10)
        except Exception as e:
            st.error(f"No se pudo contactar con el servicio de captura: {e}")
        else:
            if resp.status_code != 200:
                st.error(
                    f"Error al capturar imagen (status {resp.status_code}):\n{resp.text}"
                )
            else:
                data = resp.json()
                filename = data.get("filename")
                st.session_state["last_filename"] = filename

                st.success(f"Imagen capturada y guardada como: {filename}")
                st.json(data)

    st.markdown("---")
    st.subheader("2. Última imagen capturada")

    last_filename = st.session_state.get("last_filename")
    if last_filename:
        st.write(f"Último archivo: `{last_filename}`")
        img_path = HOST_IMAGES_DIR / last_filename
        if img_path.exists():
            try:
                image = Image.open(img_path)
                st.image(image, caption=last_filename, use_column_width=True)
            except Exception as e:
                st.warning(f"No se pudo abrir la imagen {img_path}: {e}")
        else:
            st.warning(
                f"No se encontró la imagen en el host en {img_path}. "
                "Asegúrate de que docker-compose monta ./images correctamente."
            )
    else:
        st.info("Aún no has capturado ninguna imagen en esta sesión.")


# --- Columna derecha: predicciones YOLO ---

with col_right:
    st.subheader("3. Predicción YOLO + imagen anotada de la última captura")

    if st.button("🖼️ Predecir y guardar imagen anotada"):
        if not st.session_state.get("last_filename"):
            st.warning("Primero captura una imagen para tener un filename.")
        else:
            payload = {
                "filename": st.session_state["last_filename"],
                "confidence_threshold": confidence_threshold,
            }
            try:
                resp = requests.post(
                    predict_from_saved_annotated_url, params=payload, timeout=30
                )
            except Exception as e:
                st.error(f"No se pudo contactar con el servicio de predicción: {e}")
            else:
                if resp.status_code != 200:
                    st.error(
                        f"Error al hacer la predicción anotada (status {resp.status_code}):\n{resp.text}"
                    )
                else:
                    data = resp.json()
                    st.success(
                        f"Predicción anotada completada para {st.session_state['last_filename']}"
                    )
                    st.json(data)

                    annotated_filename = data.get("annotated_filename")
                    if annotated_filename:
                        annotated_path = HOST_ANNOTATED_DIR / annotated_filename
                        if annotated_path.exists():
                            try:
                                annotated_image = Image.open(annotated_path)
                                st.image(
                                    annotated_image,
                                    caption=f"Imagen anotada: {annotated_filename}",
                                    use_column_width=True,
                                )
                            except Exception as e:
                                st.warning(
                                    f"No se pudo abrir la imagen anotada {annotated_path}: {e}"
                                )
                        else:
                            st.warning(
                                f"No se encontró la imagen anotada en {annotated_path}. "
                                "Asegúrate de que el volumen ./images_annotated está disponible en el host."
                            )

    st.markdown("---")
    st.subheader("4. Predicción YOLO sobre TODAS las imágenes guardadas")

    if st.button(
        "📂 Predecir todas las imágenes de la carpeta images\n\n 🗃️ Se guardan en images_annotated"
    ):
        try:
            resp = requests.get(
                predict_all_saved_url,
                params={"confidence_threshold": confidence_threshold, "limit": 30},
                timeout=300,
            )
        except Exception as e:
            st.error(f"No se pudo contactar con el servicio de predicción: {e}")
        else:
            if resp.status_code != 200:
                st.error(
                    f"Error al hacer la predicción masiva (status {resp.status_code}):\n{resp.text}"
                )
            else:
                st.success("Predicciones completadas para todas las imágenes encontradas.")
                results = resp.json()
                st.json(results)

 