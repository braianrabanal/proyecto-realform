from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, Response
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
import requests
import threading
from queue import Queue
import time

"""
Aplicación FastAPI para:
- Cargar y ejecutar un modelo YOLO entrenado
- Servir stream de video con predicciones en tiempo real
"""

app = FastAPI(title="Predict Service", version="1.0.0")

# Variable global para compartir frames procesados
current_frame_queue = Queue(maxsize=1)
frame_lock = threading.Lock()


# --- Configuración y carga perezosa del modelo YOLO ---

MODELS_DIR = Path(".")  # Carpeta principal donde buscar modelos .pt
MODEL_PATH = Path("best.pt")  # Modelo por defecto
_yolo_model: Optional[YOLO] = None
_model_lock = threading.Lock()  # Lock para cambiar modelo de forma segura

# Directorio donde se encuentran las imágenes capturadas.
# En docker-compose se monta todo el proyecto en /app,
# por lo que el host ./images se ve aquí como /app/images → Path("images")
IMAGES_DIR = Path("images")

# Directorio donde se guardarán las imágenes anotadas (con bounding boxes)
ANNOTATED_DIR = Path("images_annotated")
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


def get_model() -> YOLO:
    """
    Carga el modelo YOLO una sola vez (lazy load) y lo reutiliza
    en las siguientes peticiones. Optimizado con FP16 para mayor velocidad si CUDA está disponible.
    """
    global _yolo_model
    if _yolo_model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Modelo no encontrado en {MODEL_PATH}")
        _yolo_model = YOLO(str(MODEL_PATH))
        
        # Intentar usar GPU, si no está disponible usar CPU
        try:
            _yolo_model.to('cuda')
            _yolo_model.half()
            print(f"✓ Modelo {MODEL_PATH.name} optimizado con GPU + FP16")
        except Exception as e:
            print(f"ℹ GPU no disponible, usando CPU: {e}")
    return _yolo_model


def set_model(model_filename: str) -> dict:
    """
    Cambia el modelo YOLO actual a uno disponible en la carpeta principal.
    """
    global _yolo_model, MODEL_PATH
    
    new_model_path = MODELS_DIR / model_filename
    
    # Validar que el archivo existe
    if not new_model_path.exists():
        return {"error": f"Modelo no encontrado: {model_filename}"}
    
    # Validar que es un archivo .pt
    if new_model_path.suffix.lower() != '.pt':
        return {"error": f"Archivo debe ser .pt, se recibió: {new_model_path.suffix}"}
    
    with _model_lock:
        # Descargar modelo anterior si existe
        if _yolo_model is not None:
            try:
                _yolo_model = None
            except:
                pass
        
        # Cambiar ruta del modelo
        MODEL_PATH = new_model_path
        _yolo_model = None
        
        # Cargar nuevo modelo
        try:
            loaded_model = YOLO(str(MODEL_PATH))
            
            # Intentar usar GPU, si no está disponible usar CPU
            try:
                loaded_model.to('cuda')
                loaded_model.half()
                print(f"✓ Modelo {model_filename} optimizado con GPU + FP16")
            except Exception as cuda_error:
                print(f"ℹ GPU no disponible para {model_filename}, usando CPU: {cuda_error}")
            
            _yolo_model = loaded_model
            return {"status": "success", "model": model_filename}
        except Exception as e:
            print(f"⚠ Error al cambiar modelo: {e}")
            return {"error": str(e)}


@app.get("/health")
async def health() -> dict:
    """
    Comprobación sencilla de salud del servicio.
    """
    return {"status": "ok", "current_model": MODEL_PATH.name}


@app.get("/models")
async def list_models() -> JSONResponse:
    """
    Lista todos los modelos YOLO disponibles (.pt) en la carpeta principal.
    """
    try:
        models = sorted([f.name for f in MODELS_DIR.glob("*.pt")])
        return JSONResponse({
            "available_models": models,
            "current_model": MODEL_PATH.name,
            "total": len(models)
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error listando modelos: {str(e)}"}
        )


@app.post("/select_model")
async def select_model(model_filename: str) -> JSONResponse:
    """
    Selecciona un modelo YOLO diferente para usar en las predicciones.
    
    Parámetros:
    - model_filename: nombre del archivo .pt (ej: "best.pt", "modelo_2.pt")
    
    Ejemplo:
    POST /select_model?model_filename=best.pt
    """
    result = set_model(model_filename)
    
    if "error" in result:
        return JSONResponse(status_code=400, content=result)
    
    return JSONResponse(result)


def _run_inference_on_image(
    img,
    save_annotated_path: Path | None = None,
    confidence_threshold: float = 0.25,
) -> dict:
    """
    Ejecuta YOLO sobre una imagen ya cargada (matriz de OpenCV)
    y devuelve un dict con las detecciones.
    """
    model = get_model()
    results = model(img)[0]  # primer resultado

    detections: list[dict] = []
    class_counts = {}

    for box in results.boxes:
        cls_id = int(box.cls.item())
        score = float(box.conf.item())

        # Filtrar por umbral de confianza deseado
        if score < confidence_threshold:
            continue
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())

        # Obtener nombre de la clase desde el modelo / resultados
        class_name = None
        if hasattr(results, "names") and isinstance(results.names, dict):
            class_name = results.names.get(cls_id)
        if class_name is None and hasattr(model, "names") and isinstance(model.names, dict):
            class_name = model.names.get(cls_id)
        if class_name is None:
            class_name = str(cls_id)

        # Acumular conteo por tipo de objeto
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        detections.append(
            {
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": score,
                "bbox": [x1, y1, x2, y2],
            }
        )

    result = {
        "Numero de objetos": len(detections),
        "Objetos por tipo": class_counts,
        "detections": detections,
    }

    if save_annotated_path is not None:
        annotated_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            class_name = det["class_name"]
            conf = det["confidence"]

            cv2.rectangle(
                annotated_img,
                (x1_i, y1_i),
                (x2_i, y2_i),
                (0, 255, 0),
                2,
            )

            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                annotated_img,
                label,
                (x1_i, max(y1_i - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite(str(save_annotated_path), annotated_img)
        result["annotated_img"] = annotated_img

    return result


def _annotate_image(img, detections):
    annotated_img = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
        class_name = det["class_name"]
        conf = det["confidence"]

        cv2.rectangle(
            annotated_img,
            (x1_i, y1_i),
            (x2_i, y2_i),
            (0, 255, 0),
            2,
        )

        label = f"{class_name} {conf:.2f}"
        cv2.putText(
            annotated_img,
            label,
            (x1_i, max(y1_i - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return annotated_img


@app.post("/predict_upload")
async def predict_upload(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.25,
) -> Response:
    """
    Recibe una imagen por upload y devuelve la imagen anotada en JPEG.
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No se pudo leer la imagen enviada"},
        )

    data = _run_inference_on_image(
        img,
        confidence_threshold=confidence_threshold,
    )
    annotated_img = _annotate_image(img, data["detections"])

    _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.post("/predict_from_saved")
async def predict_from_saved(
    filename: str, confidence_threshold: float = 0.25
) -> JSONResponse:
    """
    Lee una imagen ya guardada en disco (por ejemplo capturada por /capture)
    y ejecuta la inferencia YOLO sobre ella.
    """
    img_path = IMAGES_DIR / filename

    if not img_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Imagen no encontrada: {img_path}"},
        )

    img = cv2.imread(str(img_path))
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No se pudo leer la imagen desde disco"},
        )

    data = _run_inference_on_image(
        img,
        confidence_threshold=confidence_threshold,
    )
    return JSONResponse(data)


@app.post("/predict_from_saved_annotated")
async def predict_from_saved_annotated(
    filename: str, confidence_threshold: float = 0.25
) -> JSONResponse:
    """
    Igual que /predict_from_saved, pero además genera y guarda una imagen
    anotada con los bounding boxes en ANNOTATED_DIR.

    - Entrada: {"filename": "mi_imagen.jpg"}
    - Salida: detecciones + nombre del archivo anotado.
    """
    img_path = IMAGES_DIR / filename

    if not img_path.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Imagen no encontrada: {img_path}"},
        )

    img = cv2.imread(str(img_path))
    if img is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No se pudo leer la imagen desde disco"},
        )

    annotated_filename = f"annotated_{filename}"
    annotated_path = ANNOTATED_DIR / annotated_filename

    data = _run_inference_on_image(
        img,
        save_annotated_path=annotated_path,
        confidence_threshold=confidence_threshold,
    )
    data.update(
        {
            "annotated_filename": annotated_filename,
            "annotated_relative_path": f"images_annotated/{annotated_filename}",
        }
    )

    return JSONResponse(data)



@app.get("/predict_all_saved")
async def predict_all_saved(confidence_threshold: float = 0.25) -> JSONResponse:
    """
    Recorre todas las imágenes guardadas en IMAGES_DIR y ejecuta YOLO sobre cada una.
    Para cada imagen, también genera y guarda una versión anotada con bounding boxes
    en ANNOTATED_DIR.

    Devuelve un diccionario:
    {
        "nombre_original.jpg": {
            ...detecciones...,
            "annotated_filename": "annotated_nombre_original.jpg",
            "annotated_relative_path": "images_annotated/annotated_nombre_original.jpg"
        },
        ...
    }
    """
    if not IMAGES_DIR.exists():
        return JSONResponse(
            status_code=404,
            content={"error": f"Directorio de imágenes no encontrado: {IMAGES_DIR}"},
        )

    image_paths: List[Path] = sorted(IMAGES_DIR.glob("*.jpg"))

    if not image_paths:
        return JSONResponse(
            status_code=404,
            content={"error": "No se encontraron imágenes en el directorio"},
        )

    results_by_file = {}

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            results_by_file[img_path.name] = {
                "error": "No se pudo leer la imagen desde disco"
            }
            continue

        annotated_filename = f"annotated_{img_path.name}"
        annotated_path = ANNOTATED_DIR / annotated_filename

        data = _run_inference_on_image(
            img,
            save_annotated_path=annotated_path,
            confidence_threshold=confidence_threshold,
        )
        data.update(
            {
                "annotated_filename": annotated_filename,
                "annotated_relative_path": f"images_annotated/{annotated_filename}",
            }
        )

        results_by_file[img_path.name] = data

    return JSONResponse(results_by_file)


def process_video_stream(
    capture_url: str = "http://localhost:8001/video",
    confidence_threshold: float = 0.25
):
    """
    Obtiene frames del servicio de captura, ejecuta YOLO y envía frame procesado.
    Limitado a 15 FPS para evitar procesar frames innecesarios.
    """
    model = get_model()
    last_process_time = 0
    min_process_interval = 1.0 / 15  # Procesar máximo 15 frames por segundo
    
    try:
        response = requests.get(capture_url, stream=True)
        bytes_data = b""
        
        for chunk in response.iter_content(chunk_size=1024):
            bytes_data += chunk
            
            # Buscar límite de frame MJPEG
            a = bytes_data.find(b'\xff\xd8')  # JPG start
            b = bytes_data.find(b'\xff\xd9')  # JPG end
            
            if a != -1 and b != -1:
                jpg_data = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]
                
                # Decodificar JPEG
                frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Control de FPS: solo procesar cada X milisegundos
                    current_time = time.time()
                    if current_time - last_process_time < min_process_interval:
                        continue  # Saltar este frame, procesar solo 15 FPS
                    
                    last_process_time = current_time
                    
                    # Redimensionar frame para acelerar inferencia (~40% más rápido)
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
                    
                    # Ejecutar YOLO
                    results = model(frame)[0]
                    annotated_frame = frame.copy()
                    
                    for box in results.boxes:
                        cls_id = int(box.cls.item())
                        score = float(box.conf.item())
                        
                        if score < confidence_threshold:
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        
                        # Obtener nombre de la clase
                        class_name = None
                        if hasattr(results, "names") and isinstance(results.names, dict):
                            class_name = results.names.get(cls_id)
                        if class_name is None and hasattr(model, "names"):
                            class_name = model.names.get(cls_id, str(cls_id))
                        if class_name is None:
                            class_name = str(cls_id)
                        
                        # Dibujar bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name} {score:.2f}"
                        cv2.putText(
                            annotated_frame, label, (x1, max(y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
                        )
                    
                    # Guardar frame procesado en queue
                    try:
                        current_frame_queue.put_nowait(annotated_frame)
                    except:
                        pass
                        
    except Exception as e:
        print(f"Error en video stream: {e}")


def inference_video_stream():
    """
    Genera stream MJPEG con las predicciones de YOLO.
    Optimizado: calidad reducida (40), FPS limitado a 15 para menor latencia.
    """
    last_frame_time = 0
    min_frame_interval = 1.0 / 15  # Limitar a 15 FPS máximo
    
    while True:
        try:
            frame = current_frame_queue.get(timeout=1)
            
            # Control de FPS: evitar enviar frames muy rápido
            current_time = time.time()
            if current_time - last_frame_time < min_frame_interval:
                continue  # Saltar frame si es demasiado pronto
            
            last_frame_time = current_time
            
            # Comprimir a JPEG con calidad baja (40) para transmisión más rápida
            # Reduce tamaño ~60-70% comparado con calidad 80
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
        except:
            pass


@app.on_event("startup")
async def startup():
    """Inicia thread de procesamiento de video al arrancar"""
    thread = threading.Thread(
        target=process_video_stream,
        kwargs={"capture_url": "http://localhost:8001/video"},
        daemon=True
    )
    thread.start()


@app.get("/video")
async def video():
    """
    Stream de video MJPEG con predicciones de YOLO en tiempo real
    """
    return StreamingResponse(
        inference_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

