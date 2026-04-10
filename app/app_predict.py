from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from pathlib import Path
from typing import Optional, List
import cv2
import numpy as np
import os
from datetime import datetime

try:
    from clearml import Task
except Exception:
    Task = None

"""
Aplicación FastAPI para:
- Cargar y ejecutar un modelo YOLO entrenado:
  - /predict: inferencia desde archivo subido
  - /predict_from_saved: inferencia desde imagen guardada en disco
"""

app = FastAPI(title="Predict Service", version="1.0.0")


# --- Configuración y carga perezosa del modelo YOLO ---

DEFAULT_MODEL_NAME = "best.pt"
_yolo_models: dict[str, YOLO] = {}

# Umbral de confianza por defecto (común a todos los endpoints)
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.25

# Directorio donde se encuentran las imágenes capturadas.
# En docker-compose se monta todo el proyecto en /app,
# por lo que el host ./images se ve aquí como /app/images → Path("images")
IMAGES_DIR = Path("images")

# Directorio donde se guardarán las imágenes anotadas (con bounding boxes)
# En docker-compose se ha mapeado ./images_annotated -> /app/data/images_annotated
# Por tanto, aquí usamos la ruta absoluta dentro del contenedor.
ANNOTATED_DIR = Path("/app/data/images_annotated")
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# --- Configuración opcional de ClearML ---
ENABLE_CLEARML = os.getenv("ENABLE_CLEARML", "false").lower() == "true"
CLEARML_PROJECT = os.getenv("CLEARML_PROJECT", "REALFORM")
CLEARML_TASK_NAME = os.getenv("CLEARML_TASK_NAME", "predict-service")
_clearml_task = None


def _get_clearml_task():
    """
    Inicializa una tarea de ClearML solo cuando está habilitado por entorno.
    Si ClearML no está disponible o falla la conexión, se desactiva sin romper la API.
    """
    global _clearml_task
    if not ENABLE_CLEARML or Task is None:
        return None
    if _clearml_task is not None:
        return _clearml_task
    try:
        _clearml_task = Task.init(
            project_name=CLEARML_PROJECT,
            task_name=CLEARML_TASK_NAME,
            task_type=Task.TaskTypes.inference,
            auto_connect_frameworks=False,
        )
        _clearml_task.connect(
            {
                "default_confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
                "default_model_path": DEFAULT_MODEL_NAME,
                "images_dir": str(IMAGES_DIR),
                "annotated_dir": str(ANNOTATED_DIR),
            },
            name="service_config",
        )
    except Exception:
        _clearml_task = None
    return _clearml_task


def _resolve_model_path(model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """
    Resuelve el archivo de modelo dentro de /app evitando rutas arbitrarias.
    """
    safe_name = Path(model_name).name
    if not safe_name.endswith(".pt"):
        raise RuntimeError("El modelo debe ser un archivo .pt")
    return Path(safe_name)


def get_model(model_name: str = DEFAULT_MODEL_NAME) -> YOLO:
    """
    Carga el modelo YOLO una sola vez (lazy load) y lo reutiliza
    en las siguientes peticiones.
    """
    model_path = _resolve_model_path(model_name)
    model_key = str(model_path)
    if model_key not in _yolo_models:
        if not model_path.exists():
            raise RuntimeError(f"Modelo no encontrado en {model_path}")
        _yolo_models[model_key] = YOLO(str(model_path))
    return _yolo_models[model_key]


@app.get("/health")
async def health() -> dict:
    """
    Comprobación sencilla de salud del servicio.
    """
    return {"status": "ok"}


@app.get("/model_info")
async def model_info(model_name: str = DEFAULT_MODEL_NAME) -> dict:
    """
    Devuelve información del archivo de modelo utilizado para inferencia.
    """
    model_path = _resolve_model_path(model_name)
    return {
        "model_filename": model_path.name,
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
    }


def _run_inference_on_image(
    img,
    save_annotated_path: Path | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_name: str = DEFAULT_MODEL_NAME,
) -> dict:
    """
    Ejecuta YOLO sobre una imagen ya cargada (matriz de OpenCV)
    y devuelve un dict con las detecciones.
    """
    model = get_model(model_name=model_name)
    # Alinear el umbral del endpoint con el umbral interno de YOLO.
    # Sin esto, YOLO aplica su conf por defecto y puede descartar cajas
    # antes de nuestro filtro manual.
    yolo_conf = max(0.0, min(1.0, float(confidence_threshold)))
    results = model(img, conf=yolo_conf)[0]  # primer resultado

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

    # Si se solicita, generar y guardar una imagen anotada con los bounding boxes
    if save_annotated_path is not None:
        annotated_img = img.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            x1_i, y1_i, x2_i, y2_i = map(int, [x1, y1, x2, y2])
            class_name = det["class_name"]
            conf = det["confidence"]

            # Dibujar rectángulo
            cv2.rectangle(
                annotated_img,
                (x1_i, y1_i),
                (x2_i, y2_i),
                (0, 255, 0),
                2,
            )

            # Dibujar etiqueta con clase y confianza
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

    return result


def _log_inference_to_clearml(
    endpoint: str,
    filename: str,
    confidence_threshold: float,
    result_data: dict,
    annotated_path: Path | None = None,
) -> None:
    """
    Registra metadata de inferencia en ClearML (si está habilitado).
    """
    task = _get_clearml_task()
    if task is None:
        return
    try:
        now_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        total_objects = result_data.get("Numero de objetos", 0)
        class_counts = result_data.get("Objetos por tipo", {})

        task.get_logger().report_scalar(
            title="inference",
            series="objects_detected",
            value=total_objects,
            iteration=0,
        )
        for class_name, count in class_counts.items():
            task.get_logger().report_scalar(
                title="classes_detected",
                series=str(class_name),
                value=int(count),
                iteration=0,
            )

        task.get_logger().report_text(
            f"[{now_str}] endpoint={endpoint} filename={filename} "
            f"confidence_threshold={confidence_threshold} total_objects={total_objects}"
        )
        task.upload_artifact(
            name=f"inference_result_{filename}_{now_str}",
            artifact_object={
                "endpoint": endpoint,
                "filename": filename,
                "confidence_threshold": confidence_threshold,
                "result": result_data,
            },
        )

        if annotated_path is not None and annotated_path.exists():
            task.upload_artifact(
                name=f"annotated_image_{filename}_{now_str}",
                artifact_object=str(annotated_path),
            )
    except Exception:
        # Nunca bloquear el flujo de inferencia por errores de tracking.
        return


@app.post("/predict_from_saved")
async def predict_from_saved(
    filename: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_name: str = DEFAULT_MODEL_NAME,
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
        model_name=model_name,
    )
    data["model_filename"] = Path(model_name).name
    _log_inference_to_clearml(
        endpoint="/predict_from_saved",
        filename=filename,
        confidence_threshold=confidence_threshold,
        result_data=data,
    )
    return JSONResponse(data)


@app.post("/predict_from_saved_annotated")
async def predict_from_saved_annotated(
    filename: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    model_name: str = DEFAULT_MODEL_NAME,
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
        model_name=model_name,
    )
    data.update(
        {
            "annotated_filename": annotated_filename,
            "annotated_relative_path": f"images_annotated/{annotated_filename}",
            "model_filename": Path(model_name).name,
        }
    )
    _log_inference_to_clearml(
        endpoint="/predict_from_saved_annotated",
        filename=filename,
        confidence_threshold=confidence_threshold,
        result_data=data,
        annotated_path=annotated_path,
    )

    return JSONResponse(data)



@app.get("/predict_all_saved")
async def predict_all_saved(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    limit: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> JSONResponse:
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

    if limit is not None:
        try:
            limit_i = int(limit)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": "El parámetro limit debe ser un entero"},
            )
        if limit_i <= 0:
            return JSONResponse(
                status_code=400,
                content={"error": "El parámetro limit debe ser > 0"},
            )
        image_paths = image_paths[:limit_i]

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
            model_name=model_name,
        )
        data.update(
            {
                "annotated_filename": annotated_filename,
                "annotated_relative_path": f"images_annotated/{annotated_filename}",
                "model_filename": Path(model_name).name,
            }
        )

        results_by_file[img_path.name] = data
        _log_inference_to_clearml(
            endpoint="/predict_all_saved",
            filename=img_path.name,
            confidence_threshold=confidence_threshold,
            result_data=data,
            annotated_path=annotated_path,
        )

    return JSONResponse(results_by_file)

