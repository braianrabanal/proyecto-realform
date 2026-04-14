import requests
import streamlit as st
import streamlit.components.v1 as components

"""
Aplicación Streamlit para visualizar:
- Stream de video en tiempo real desde la cámara (sin procesar)
- Stream de video con predicciones YOLO en tiempo real (con bounding boxes)
"""

st.set_page_config(
    page_title="PROYECTO REALFORM - Detección en Tiempo Real",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("PROYECTO REALFORM")
st.subheader("Visualización en Tiempo Real con YOLO")

# --- Configuración de APIs en sidebar ---
st.sidebar.header("⚙️ Configuración de APIs")
capture_base_url = st.sidebar.text_input(
    "URL servicio captura:",
    value="http://localhost:8001",
    help="Normalmente http://localhost:8001",
)
predict_base_url = st.sidebar.text_input(
    "URL servicio predicción:",
    value="http://localhost:8002",
    help="Normalmente http://localhost:8002",
)

# URLs de los endpoints de video
capture_video_url = f"{capture_base_url.rstrip('/')}/video"
predict_video_url = f"{predict_base_url.rstrip('/')}/video"
capture_image_url = f"{capture_base_url.rstrip('/')}/capture_image"
predict_upload_url = f"{predict_base_url.rstrip('/')}/predict_upload"

# URLs de los endpoints de health check
capture_health_url = f"{capture_base_url.rstrip('/')}/health"
predict_health_url = f"{predict_base_url.rstrip('/')}/health"

# --- Verificar conexión con servicios ---
st.sidebar.markdown("---")
st.sidebar.subheader("📡 Estado de Servicios")

# Verificar captura
try:
    resp = requests.get(capture_health_url, timeout=2)
    if resp.status_code == 200:
        st.sidebar.success("✅ Servicio de Captura: OK")
    else:
        st.sidebar.error("❌ Servicio de Captura: Error")
except:
    st.sidebar.error("❌ Servicio de Captura: No disponible")

# Verificar predicción
try:
    resp = requests.get(predict_health_url, timeout=2)
    if resp.status_code == 200:
        st.sidebar.success("✅ Servicio de Predicción: OK")
    else:
        st.sidebar.error("❌ Servicio de Predicción: Error")
except:
    st.sidebar.error("❌ Servicio de Predicción: No disponible")

# --- Seleccionar modelo YOLO ---
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Seleccionar Modelo YOLO")

models_url = f"{predict_base_url.rstrip('/')}/models"
select_model_url = f"{predict_base_url.rstrip('/')}/select_model"

try:
    # Obtener lista de modelos disponibles
    resp = requests.get(models_url, timeout=2)
    if resp.status_code == 200:
        data = resp.json()
        available_models = data.get("available_models", [])
        current_model = data.get("current_model", "")
        
        if available_models:
            selected_model = st.sidebar.selectbox(
                "Elige un modelo:",
                available_models,
                index=available_models.index(current_model) if current_model in available_models else 0,
                help="Selecciona un modelo .pt para usar en las predicciones"
            )
            
            # Si cambió el modelo, solicitar al servidor que lo cargue
            if selected_model != current_model:
                with st.sidebar.spinner(f"Cargando módelo {selected_model}..."):
                    try:
                        change_resp = requests.post(
                            f"{select_model_url}?model_filename={selected_model}",
                            timeout=5
                        )
                        if change_resp.status_code == 200:
                            st.sidebar.success(f"✅ Modelo cambiado a: {selected_model}")
                        else:
                            st.sidebar.error(f"❌ Error al cambiar modelo: {change_resp.text}")
                    except Exception as e:
                        st.sidebar.error(f"❌ Error: {str(e)}")
            else:
                st.sidebar.info(f"📌 Modelo activo: {current_model}")
        else:
            st.sidebar.warning("⚠️ No hay modelos .pt encontrados en la carpeta principal")
    else:
        st.sidebar.error("❌ Error al obtener lista de modelos")
except:
    st.sidebar.error("❌ No se puede conectar con el servicio de predicción")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Controles")

# Selector principal de modo
if "app_mode" not in st.session_state:
    st.session_state.app_mode = None
    st.session_state.captured_image = None
    st.session_state.annotated_image = None
    st.session_state.capture_error = None

st.markdown("## Selecciona el modo de uso")
modo_col1, modo_col2 = st.columns(2)
with modo_col1:
    if st.button("TIEMPO REAL", key="mode_realtime"):
        st.session_state.app_mode = "TIEMPO REAL"
with modo_col2:
    if st.button("CAPTURA", key="mode_capture"):
        st.session_state.app_mode = "CAPTURA"

if st.session_state.app_mode is None:
    st.info("Elige un modo para comenzar: TIEMPO REAL o CAPTURA.")
else:
    if st.button("← Cambiar modo"):
        st.session_state.app_mode = None
        st.session_state.captured_image = None
        st.session_state.annotated_image = None
        st.session_state.capture_error = None
    st.markdown(f"### Modo seleccionado: **{st.session_state.app_mode}**")

    if st.session_state.app_mode == "TIEMPO REAL":
        show_stream = st.sidebar.checkbox("Mostrar stream en directo", value=True)

        if show_stream:
            # --- Mostrar dos streams lado a lado ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📹 Captura en Vivo")
                st.write("Imagen sin procesar desde la cámara")
                components.html(
                    f"""
                    <div style="width:100%; max-height:520px; overflow:hidden;">
                      <img
                        src="{capture_video_url}"
                        style="width:100%; height:520px; object-fit:contain; border:1px solid #ddd;"
                        alt="Captura en vivo"
                      />
                    </div>
                    """,
                    height=540,
                )
            
            with col2:
                st.subheader("🤖 Predicción en Vivo (YOLO)")
                st.write("Imagen con detecciones YOLO en tiempo real")
                components.html(
                    f"""
                    <div style="width:100%; max-height:520px; overflow:hidden;">
                      <img
                        src="{predict_video_url}"
                        style="width:100%; height:520px; object-fit:contain; border:1px solid #ddd;"
                        alt="Predicción en vivo"
                      />
                    </div>
                    """,
                    height=540,
                )
        else:
            st.info("📴 Streams desactivados. Activa la casilla para visualizar.")
    else:
        st.subheader("📸 Captura y análisis bajo demanda")
        st.write("Pulsa el botón para tomar una fotografía y enviarla al servicio de inferencia.")

        if st.button("Capturar imagen"):
            try:
                resp = requests.get(capture_image_url, timeout=5)
                if resp.status_code == 200:
                    st.session_state.captured_image = resp.content
                    st.session_state.capture_error = None

                    predict_resp = requests.post(
                        predict_upload_url,
                        files={"file": ("captura.jpg", resp.content, "image/jpeg")},
                        timeout=20,
                    )

                    if predict_resp.status_code == 200:
                        st.session_state.annotated_image = predict_resp.content
                    else:
                        st.session_state.annotated_image = None
                        st.session_state.capture_error = (
                            f"Error de predicción: {predict_resp.status_code} - {predict_resp.text}"
                        )
                else:
                    st.session_state.capture_error = (
                        f"Error al capturar imagen: {resp.status_code} - {resp.text}"
                    )
            except Exception as e:
                st.session_state.capture_error = str(e)

        if st.session_state.capture_error:
            st.error(st.session_state.capture_error)

        if st.session_state.captured_image is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(
                    st.session_state.captured_image,
                    caption="Imagen capturada",
                    use_column_width=True,
                )
            with col2:
                st.image(
                    st.session_state.annotated_image or st.session_state.captured_image,
                    caption="Resultado de la inferencia",
                    use_column_width=True,
                )
st.markdown("""
### 📋 Instrucciones de Uso

1. **Asegúrate de que los dos servicios estén corriendo:**
   ```bash
   # Terminal 1: Servicio de Captura
   uvicorn app.app_capture:app --host 0.0.0.0 --port 8001
   
   # Terminal 2: Servicio de Predicción
   uvicorn app.app_predict:app --host 0.0.0.0 --port 8002
   
   # Terminal 3: Streamlit (esta aplicación)
   streamlit run streamlit_app.py
   ```

2. **Verifica la conexión** en el panel lateral izquierdo

3. **Visualiza los streams** - Los videos se actualizan automáticamente

### ⚠️ Notas Importantes

- **Cámara USB:** Asegúrate de tener una cámara USB conectada
- **Puertos:** Los servicios deben estar en los puertos 8001 y 8002
- **Modelo YOLO:** Coloca tu archivo `best.pt` en la carpeta raíz del proyecto
- **Performance:** La inferencia YOLO puede consumir GPU si está disponible

### 🔧 Solución de Problemas

- **"No disponible":** Verifica que los servicios estén corriendo
- **No hay imagen:** Verifica permisos de acceso a la cámara
- **Lento:** Reduce la resolución de la cámara o aumenta la calidad JPEG
""")