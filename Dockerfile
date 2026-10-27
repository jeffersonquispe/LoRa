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
COPY docker/entrypoint.sh docker/serve.py /
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=4)" || exit 1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "/serve.py"]

# ---------- UI ----------
FROM python:3.12-slim AS ui
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install streamlit==1.60.0 requests==2.34.2
COPY tts_app/app.py tts_app/
EXPOSE 8501
CMD ["streamlit", "run", "tts_app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# ---------- STANDALONE: la API con el adaptador de 2 voces y las voces ya dentro ----------
# Es el último stage, así que `docker build -t <nombre> .` produce esta imagen. Funciona con:
#   docker run --gpus all -p 8000:8000 <nombre>
# No lleva los pesos base (~3 GB): se bajan de Hugging Face en la primera petición. Para no repetir
# la descarga en cada contenedor:  -v ckpt_latam:/app/ckpt_latam
FROM api AS standalone
ENV PRECARGAR_MODELO=lora_out_2voces
COPY lora_out_2voces/ lora_out_2voces/
COPY tts_app/voces/ tts_app/voces/
