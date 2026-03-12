from fastapi import FastAPI
from fastapi.responses import JSONResponse
import cv2
from pathlib import Path
from datetime import datetime

"""
Aplicación FastAPI para:
- Monitorizar el servicio: GET /health
- Capturar imágenes desde una cámara: POST /capture
"""

app = FastAPI(title="Capture Service", version="1.0.0")

# Directorio donde se guardarán las imágenes dentro del contenedor.
# En docker-compose se mapea ./images -> /app/data/images
IMAGES_DIR = Path("/app/data/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health() -> dict:
    """
    Comprobación sencilla de salud del servicio.
    """
    return {"status": "ok"}


@app.post("/capture")
async def capture_frame() -> JSONResponse:
    """
    Captura un frame de la cámara USB y lo guarda como imagen en disco.
    """
    # Abrir la cámara principal (dispositivo 0, mapeado como /dev/video0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return JSONResponse(
            status_code=500,
            content={"error": "No se pudo abrir la cámara"},
        )

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return JSONResponse(
            status_code=500,
            content={"error": "No se pudo capturar el frame"},
        )

    # Nombre único basado en fecha y hora
    filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    save_path = IMAGES_DIR / filename

    ok = cv2.imwrite(str(save_path), frame)
    if not ok:
        return JSONResponse(
            status_code=500,
            content={"error": "No se pudo guardar la imagen"},
        )

    return JSONResponse(
        {
            "message": "Imagen capturada y guardada",
            "filename": filename,
            # Ruta relativa vista desde el host (./images/filename)
            "relative_path": f"images/{filename}",
        }
    )


