FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 
   

WORKDIR /app

# Dependencias de sistema mínimas para OpenCV, Ultralytics y compilación básica
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python para entrenamiento, inferencia y API
RUN pip install --upgrade pip && pip install \
    ultralytics \
    anomalib \
    clearml \
    fastapi \
    "uvicorn[standard]" \
    streamlit \
    requests \
    pillow \
    opencv-python-headless \
    numpy \
    pydantic \
    python-multipart

# Copiamos el código del proyecto
COPY . /app

# Exponemos el puerto de la API FastAPI
EXPOSE 8001

# Arrancamos la app definida en app/main.py -> app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

