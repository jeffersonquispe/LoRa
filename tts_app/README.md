# Chatterbox TTS: API + Streamlit

API (FastAPI) y una interfaz web (Streamlit) para generar voz con Chatterbox Latam, con o sin adaptador LoRA.
Usa los módulos de la carpeta raíz (`carga_ligera.py`, `ckpt_latam/`, los `lora_out*`).

## Arrancar

Dos terminales (o doble clic en `iniciar_api.bat` e `iniciar_ui.bat`), desde la carpeta raíz del proyecto:

```
python -m uvicorn api:app --app-dir tts_app --port 8000
tts_app\.venv_ui\Scripts\python.exe -m streamlit run tts_app/app.py
```

Streamlit vive en su propio entorno (`tts_app/.venv_ui`) porque su versión actual necesita un `starlette`
más nuevo que el del entorno global, y actualizarlo puede romper `gradio`/`mcp`. Si falta el entorno:
`python -m venv tts_app/.venv_ui` y `tts_app\.venv_ui\Scripts\pip install streamlit requests`.

La primera petición carga el modelo en la GPU (~1 min). Cambiar de adaptador lo recarga: la GPU de 6 GB
solo aguanta un modelo a la vez. Docs de la API: http://localhost:8000/docs

## API

Guía completa para quien la consume (parámetros, errores, ejemplos en curl/Python/JS): **[API.md](API.md)**.

| Ruta | Qué hace |
|---|---|
| `GET /health` | Estado y modelo cargado (sin clave) |
| `GET /modelos` | `base` y cada carpeta `lora_out*` con adaptador |
| `GET /voces` · `POST /voces` | Lista / sube audios de referencia a `tts_app/voces/` |
| `GET /idiomas` | Idiomas soportados |
| `POST /tts` | Genera audio (`audio/wav`) |

```
curl -X POST http://localhost:8000/tts -F "texto=Hola, ¿cómo estás?" -F "modelo=lora_out_2000" \
     -F "voz=por_defecto" -F "idioma=es" -o salida.wav
```

Campos de `/tts`: `texto` (máx. 600 caracteres), `modelo`, `voz`, `idioma`, `exaggeration` (0-2),
`cfg_weight` (0-1), `temperature`, y opcionalmente `voz_archivo` para una referencia puntual.

La voz `por_defecto` es `dataset_audios/001.mp3`. Para usar otra, súbela con `POST /voces`.

## Seguridad

Sin configurar nada, la API no pide clave y escucha en `localhost`. Si la expones en red, define
`TTS_API_KEY` antes de arrancar (el cliente la manda en `X-API-Key`); Streamlit la lee de la misma variable.
