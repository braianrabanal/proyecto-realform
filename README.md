# 🥤 REALFORM - Detección en Tiempo Real con YOLO

Sistema de detección de objetos en tiempo real usando YOLO, con arquitectura de microservicios basada en FastAPI y una interfaz Streamlit.

## 📋 Descripción

El proyecto está compuesto por:
- **app_capture.py**: Servicio de captura de video en tiempo real (puerto 8001)
- **app_predict.py**: Servicio de predicción YOLO con detecciones (puerto 8002)
- **streamlit_app.py**: Interfaz web para visualización y control

### Características principales:
✅ Captura de video en tiempo real desde cámara  
✅ Inferencia YOLO optimizada con GPU + FP16  
✅ Selección dinámica de modelos desde UI  
✅ Stream MJPEG de baja latencia  
✅ API REST para predicciones batch  
✅ Arquitectura escalable en microservicios  

---

## 🚀 Instalación sin Docker

### Requisitos previos:
- Python 3.8+
- pip o conda
- Cámara web conectada (opcional, para captura en vivo)

### Paso 1: Clonar/Descargar el proyecto

```bash
cd proyecto-realform
```

### Paso 2: Crear entorno virtual

```bash
# Con venv
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# O con conda
conda create -n realform python=3.11
conda activate realform
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Descargar/Preparar modelos YOLO

Coloca tus modelos (`.pt`) en la carpeta principal:
```bash
proyecto-realform/
├── best.pt                 # Modelo por defecto
├── modelo_2.pt            # (opcional)
├── modelo_3.pt            # (opcional)
└── ...
```

Puedes descargar modelos pre-entrenados o entrenar los tuyos:
```bash
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Nano (recomendado para GPU débil)
# yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
```

### Paso 5: Ejecutar los servicios

**En 3 terminales diferentes:**

```bash
# Terminal 1: Servicio de Captura (puerto 8001)
uvicorn app.app_capture:app --host 0.0.0.0 --port 8001

# Terminal 2: Servicio de Predicción (puerto 8002)
uvicorn app.app_predict:app --host 0.0.0.0 --port 8002

# Terminal 3: Interfaz Streamlit (puerto 8501)
streamlit run streamlit_app.py
```

✅ Accede a la interfaz en: **http://localhost:8501**

---

## 🐳 Instalación con Docker

### Requisitos previos:
- Docker instalado
- Docker Compose instalado
- Privilegios para acceder a cámara (en Linux: `--device=/dev/video0`)

### Paso 1: Preparar modelos

Coloca tus modelos `.pt` en la carpeta principal (igual que sin Docker).

### Paso 2: Revisar docker-compose.yaml

El archivo ya incluye los tres servicios:

```yaml
version: '3.8'

services:
  capture:
    build: .
    ports:
      - "8001:8001"
    volumes:
      - ./images:/app/images
      - ./images_annotated:/app/images_annotated
    devices:
      - /dev/video0:/dev/video0  # Para acceso a cámara (Linux)
    environment:
      - PYTHONUNBUFFERED=1
    command: uvicorn app.app_capture:app --host 0.0.0.0 --port 8001

  predict:
    build: .
    ports:
      - "8002:8002"
    volumes:
      - ./images:/app/images
      - ./images_annotated:/app/images_annotated
      - ./best.pt:/app/best.pt  # Montar modelos
    environment:
      - PYTHONUNBUFFERED=1
    command: uvicorn app.app_predict:app --host 0.0.0.0 --port 8002
    depends_on:
      - capture

  streamlit:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./:/app
    environment:
      - PYTHONUNBUFFERED=1
    command: streamlit run streamlit_app.py --server.address=0.0.0.0
    depends_on:
      - predict
```

### Paso 3: Construir e iniciar los contenedores

```bash
# Construir imágenes
docker-compose build

# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

✅ Accede a la interfaz en: **http://localhost:8501**

---

## 📡 Endpoints disponibles

### Servicio Captura (puerto 8001)

```bash
GET /health
# Respuesta: {"status": "ok"}

GET /video
# Stream MJPEG en vivo
# Acceso: http://localhost:8001/video
```

### Servicio Predicción (puerto 8002)

```bash
GET /health
# Respuesta: {"status": "ok", "current_model": "best.pt"}

GET /models
# Lista modelos disponibles
# Respuesta:
{
  "available_models": ["best.pt", "modelo_2.pt"],
  "current_model": "best.pt",
  "total": 2
}

POST /select_model?model_filename=best.pt
# Cambia modelo de predicción
# Respuesta: {"status": "success", "model": "best.pt"}

GET /video
# Stream MJPEG con predicciones
# Acceso: http://localhost:8002/video

POST /predict_from_saved?filename=image.jpg&confidence_threshold=0.25
# Predicción sobre imagen guardada
# Respuesta:
{
  "Numero de objetos": 5,
  "Objetos por tipo": {"clase1": 3, "clase2": 2},
  "detections": [
    {
      "class_id": 0,
      "class_name": "clase1",
      "confidence": 0.95,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

POST /predict_from_saved_annotated?filename=image.jpg
# Predicción + genera imagen anotada
# Respuesta: (igual que anterior + annotated_filename)

GET /predict_all_saved?confidence_threshold=0.25
# Predicción batch sobre todas las imágenes
```

---

## 🎨 Uso de la Interfaz Streamlit

### Panel lateral (Sidebar):
1. **Configuración de APIs**: URLs de los servicios
2. **Estado de Servicios**: Verificar conexión
3. **Seleccionar Modelo YOLO**: Desplegable para cambiar modelo
4. **Controles**: Mostrar/ocultar streams

### Área principal:
- **Captura en Vivo**: Video sin procesar
- **Predicción en Vivo**: Video con detectiones YOLO

### Cambiar modelo en tiempo real:
1. Abre Streamlit
2. Ve a "🤖 Seleccionar Modelo YOLO"
3. Elige un modelo del desplegable
4. Se carga automáticamente (espera spinner)
5. El stream se actualiza con las nuevas predicciones

---

## ⚙️ Configuración

### Variables de entorno (docker-compose):

```yaml
PYTHONUNBUFFERED=1  # Output en tiempo real
```

### Parámetros en app_predict.py:

```python
MODEL_PATH = Path("best.pt")  # Modelo por defecto
MODELS_DIR = Path(".")        # Carpeta donde buscar .pt
IMAGES_DIR = Path("images")   # Donde se guardan capturas
ANNOTATED_DIR = Path("images_annotated")  # Imágenes anotadas
```

### Optimizaciones de inferencia:

```python
# En process_video_stream()
min_process_interval = 1.0 / 15  # Limitar a 15 FPS
frame = cv2.resize(frame, (640, 480))  # Redimensionar para velocidad

# En inference_video_stream()
cv2.IMWRITE_JPEG_QUALITY = 40  # Compresión para ancho de banda
```

---

## 🔧 Troubleshooting

### Error: "HTTPConnectionPool ... Failed to establish connection"
**Solución**: Asegúrate de que ambos servicios están corriendo en los puertos correctos.

```bash
# Verifica que los puertos estén en uso
netstat -ano | findstr :8001  # Windows
lsof -i :8001                  # Linux/Mac
```

### Error: "Torch not compiled with CUDA enabled"
**Solución**: Ya está corregido en el código, caerá a CPU automáticamente.

Para forzar GPU CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Error: "No se pudo abrir la cámara"
**Solución (Linux en Docker)**: Agrega el device a docker-compose:
```yaml
devices:
  - /dev/video0:/dev/video0
```

**Solución (Windows)**: Asegúrate que la cámara no está siendo usada por otra aplicación.

### Performance lento
**Soluciones**:
1. Bajar FPS: Modifica `min_process_interval` en `app_predict.py`
2. Usar modelo más pequeño: `yolov8n.pt` en lugar de `yolov8x.pt`
3. Activar GPU: Instala PyTorch con CUDA
4. Aumentar confianza: Usa `confidence_threshold=0.5` o mayor

---

## 📦 Estructura de carpetas

```
proyecto-realform/
├── app/
│   ├── app_capture.py        # Servicio captura
│   └── app_predict.py        # Servicio predicción
├── images/                   # Imágenes capturadas
├── images_annotated/         # Imágenes con bounding boxes
├── best.pt                   # Modelo (puedes agregar más)
├── streamlit_app.py          # Interfaz UI
├── requirements.txt          # Dependencias
├── Dockerfile                # Para Docker
├── docker-compose.yaml       # Orquestación
└── README.md                 # Este archivo
```

---

## 📝 Modelos YOLO disponibles

```
YOLOv8 Nano    (yolov8n.pt)   - Rápido, bajo requerimiento
YOLOv8 Small   (yolov8s.pt)   - Equilibrio velocidad/precisión
YOLOv8 Medium  (yolov8m.pt)   - Mayor precisión
YOLOv8 Large   (yolov8l.pt)   - Muy preciso, lento
YOLOv8 XLarge  (yolov8x.pt)   - Máxima precisión, muy lento
```

Descarga desde: https://github.com/ultralytics/assets/releases/

---

## 🚀 Performance esperado

| Modelo | GPU (RTX 3060) | CPU (i5-10400) | Latencia | Tamaño |
|--------|---|---|---|---|
| yolov8n | 50 FPS | 8 FPS | 20ms | 7MB |
| yolov8s | 35 FPS | 4 FPS | 28ms | 23MB |
| yolov8m | 20 FPS | 2 FPS | 50ms | 50MB |

*Nota: Con GPU CUDA activada. Sin CUDA los tiempos son 5-10x más lento.*

---

## 📞 Soporte

Para problemas o preguntas, revisa:
1. Los logs de la aplicación (`docker-compose logs`)
2. Estado de servicios en el sidebar de Streamlit
3. Los endpoints `/health` para verificar estado

---

## 📄 Licencia

Basado en [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)

---

**Última actualización**: Abril 2026
