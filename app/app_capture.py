from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, Response
import cv2
from pathlib import Path
from datetime import datetime
import threading
import time

"""
Aplicación FastAPI para:
- Monitorizar el servicio: GET /health
- Servir stream de video en tiempo real: GET /video
"""

app = FastAPI(title="Capture Service", version="1.0.0")

# Directorio donde se guardarán las imágenes
IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Variable global para el frame actual
current_frame = None
frame_lock = threading.Lock()


def capture_frames():
    """
    Captura frames de la cámara en tiempo real (thread background)
    """
    global current_frame
    # Usar DirectShow en Windows para evitar bloqueos comunes con OpenCV
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    # Ajustar resolución para reducir carga
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame
        else:
            # Si falla la lectura, esperar un poco antes de reintentar
            time.sleep(0.01)
            continue
    
    cap.release()


def video_stream():
    """
    Genera stream MJPEG con los frames de la cámara
    """
    while True:
        with frame_lock:
            if current_frame is not None:
                frame = current_frame.copy()
            else:
                time.sleep(0.01)
                continue
        
        # Comprimir frame a JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# Iniciar thread de captura al arrancar
@app.on_event("startup")
async def startup():
    thread = threading.Thread(target=capture_frames, daemon=True)
    thread.start()


@app.get("/health")
async def health() -> dict:
    """
    Comprobación sencilla de salud del servicio.
    """
    return {"status": "ok"}


@app.get("/video")
async def video():
    """
    Stream de video MJPEG desde la cámara en tiempo real
    """
    return StreamingResponse(
        video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/capture_image")
async def capture_image():
    """
    Devuelve una única imagen JPEG capturada del frame actual.
    """
    with frame_lock:
        if current_frame is None:
            return JSONResponse(
                status_code=503,
                content={"error": "Frame no disponible aún"},
            )
        frame = current_frame.copy()

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


