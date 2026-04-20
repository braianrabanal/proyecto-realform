# REALFORM - Deteccion en tiempo real y analisis por lotes

Proyecto con arquitectura de microservicios para:

- captura de imagen/video desde camara
- inferencia de deteccion con YOLO
- inferencia de anomalias con Anomalib
- visualizacion y control desde Streamlit

## Resumen de servicios

- `app/app_capture.py` (puerto `8001`): servicio de captura y stream MJPEG.
- `app/app_predict.py` (puerto `8002`): servicio de prediccion (YOLO + Anomalib).
- `streamlit_app.py` (puerto `8501`): interfaz de usuario.

## Estructura principal

```text
proyecto-realform/
├── app/
│   ├── app_capture.py
│   └── app_predict.py
├── images/
├── images_anomalibtest/
├── images_annotated/
├── streamlit_app.py
├── Dockerfile
├── docker-compose.yaml
└── README.md
```

## Ejecucion con Docker (recomendado)

### Requisitos

- Docker
- Docker Compose
- Camara USB (si usas captura en vivo)

### Levantar servicios

```bash
docker compose up --build
```

### Levantar en segundo plano

```bash
docker compose up -d --build
```

### Ver logs

```bash
docker compose logs -f
```

### Parar servicios

```bash
docker compose down
```

## URLs utiles

- Streamlit: `http://localhost:8501`
- Swagger servicio de prediccion: `http://localhost:8002/docs`
- Health captura: `http://localhost:8001/health`
- Health prediccion: `http://localhost:8002/health`

## Endpoints principales de prediccion

### YOLO

- `GET /models`: lista modelos `.pt` disponibles.
- `POST /select_model?model_filename=...`: cambia modelo YOLO activo.
- `POST /predict_from_saved`: inferencia YOLO sobre una imagen de `images`.
- `POST /predict_from_saved_annotated`: igual, guardando anotada en `images_annotated`.
- `GET /predict_all_saved`: inferencia masiva YOLO sobre `images`.
- `GET /video`: stream MJPEG con detecciones en vivo.

### Anomalib

- `GET /predict_anomalib_folder`: inferencia masiva sobre carpeta de test (detallado abajo).

---

## Nuevo endpoint Anomalib: `GET /predict_anomalib_folder`

Este endpoint permite inferenciar una carpeta completa con un modelo Anomalib y guardar resultados anotados en `images_annotated`.

### Que hace

1. Carga el modelo `.pt` de Anomalib indicado por parametro.
2. Recorre imagenes del directorio indicado (por defecto `images_anomalibtest`).
3. Ejecuta inferencia por imagen.
4. Guarda cada salida anotada en `images_annotated` con prefijo `annotated_anomalib_`.
5. Devuelve un JSON con:
   - conteo de defectos vs normales
   - tiempos de inferencia (total y promedio)
   - resultado por imagen

### Parametros

- `model_filename` (obligatorio): nombre del modelo anomalib, por ejemplo `stfpm_telas_27112025.pt`.
- `image_dir` (opcional): carpeta a procesar. Default: `images_anomalibtest`.
- `score_threshold` (opcional): umbral para clasificar anomalia usando `pred_score`.

### Ejemplo de llamada (curl)

```bash
curl "http://localhost:8002/predict_anomalib_folder?model_filename=stfpm_telas_27112025.pt&image_dir=images_anomalibtest&score_threshold=0.5"
```

### Como usarlo desde Swagger

1. Abre `http://localhost:8002/docs`.
2. Busca `GET /predict_anomalib_folder`.
3. Pulsa **Try it out**.
4. Rellena:
   - `model_filename`: por ejemplo `stfpm_telas_27112025.pt`
   - `image_dir`: `images_anomalibtest` (o tu carpeta)
   - `score_threshold`: por ejemplo `0.5`
5. Pulsa **Execute**.
6. Revisa:
   - codigo `200`
   - JSON de salida
   - imagenes guardadas en `images_annotated`.

### Campos clave de respuesta JSON

- `model_filename`
- `source_dir`
- `processed_images`
- `defect_count`
- `normal_count`
- `timing.total_inference_time_ms`
- `timing.avg_inference_time_ms`
- `timing.total_pipeline_time_ms`
- `results[<archivo>]` con:
  - `pred_label`
  - `pred_score`
  - `score_threshold_used`
  - `effective_label`
  - `is_anomalous`
  - `inference_time_ms`
  - `annotated_relative_path`

---

## Notas importantes

- El endpoint de Anomalib requiere que el modelo sea compatible con `TorchInferencer`.
- Si el modelo usa pickle seguro restringido, se habilita `TRUST_REMOTE_CODE` en entorno Docker para carga controlada.
- Las anotaciones de Anomalib muestran etiqueta efectiva segun `score_threshold`.
- Para comparar modelos, usa los campos de `timing` y `defect_count`.

## Troubleshooting rapido

### Error: `No module named 'anomalib'`

Reconstruye imagen de `app_predict`:

```bash
docker compose build app_predict
docker compose up -d app_predict
```

### Error 400/500 al cargar modelo Anomalib

- Verifica que el archivo existe en la raiz del proyecto.
- Verifica que tenga extension `.pt`.
- Revisa logs:

```bash
docker compose logs -f app_predict
```

### No se detectan imagenes de una carpeta

- Verifica ruta y nombre en `image_dir`.
- Revisa extensiones soportadas (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).

---

## Licencia

Basado en Ultralytics YOLO y Anomalib.
