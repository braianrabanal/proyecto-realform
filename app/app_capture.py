from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
from pathlib import Path
from datetime import datetime
import threading

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
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                current_frame = frame
        else:
            break
    
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


