# syntax=docker/dockerfile:1
# Dos imágenes (targets): `api` (GPU, Chatterbox + LoRA) y `ui` (Streamlit, ligera).
# Streamlit va aparte para no mezclar su starlette con el de gradio/fastapi de Chatterbox.

# ---------- API ----------
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS api
ARG CHATTERBOX_COMMIT=5de7a54
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/root/.cache/huggingface
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Chatterbox oficial fijado en el mismo commit que el submódulo del repo (carga_ligera.py lo busca en ./chatterbox/src)
RUN git clone https://github.com/resemble-ai/chatterbox.git chatterbox \
    && git -C chatterbox checkout ${CHATTERBOX_COMMIT}

# torch 2.6.0 ya viene en la imagen base. El resto con las versiones con las que se probó en local.
RUN pip install -e chatterbox \
    peft==0.21.2 soundfile==0.14.0 \
    fastapi==0.115.5 uvicorn==0.32.1 python-multipart==0.0.22 \
    boto3 huggingface_hub

COPY carga_ligera.py pesos_s3.py ./
COPY tts_app/api.py tts_app/
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "api:app", "--app-dir", "tts_app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- UI ----------
FROM python:3.12-slim AS ui
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install streamlit==1.60.0 requests==2.34.2
COPY tts_app/app.py tts_app/
EXPOSE 8501
CMD ["streamlit", "run", "tts_app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
