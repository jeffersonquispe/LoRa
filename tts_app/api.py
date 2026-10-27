"""API de texto a voz con Chatterbox (modelo Latam) y, opcionalmente, un adaptador LoRA.

Ejecutar desde cualquier sitio:
    python -m uvicorn api:app --app-dir tts_app --port 8000
Docs interactivas: http://localhost:8000/docs

Un solo modelo en GPU a la vez (6 GB no alcanzan para dos): al pedir otro adaptador se recarga el modelo.
Si defines la variable de entorno TTS_API_KEY, todas las rutas (salvo /health) exigen el encabezado X-API-Key.
"""
import gc
import io
import json
import os
import re
import sys
import tempfile
import threading
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
os.chdir(RAIZ)  # carga_ligera usa rutas relativas (ckpt_latam)

import carga_ligera  # noqa: E402  (debe ir antes de usar chatterbox)
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from chatterbox.mtl_tts import SUPPORTED_LANGUAGES  # noqa: E402
from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from fastapi.security import APIKeyHeader  # noqa: E402
from peft import PeftModel  # noqa: E402

VOCES_DIR = os.path.join(RAIZ, "tts_app", "voces")
VOZ_POR_DEFECTO = os.path.join(RAIZ, "dataset_audios", "001.mp3")
REGISTRO = os.path.join(VOCES_DIR, "voces.json")  # {id: {archivo, nombre, genero}}
ID_VALIDO = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
EXT_AUDIO = (".wav", ".mp3", ".flac", ".ogg")
MAX_CARACTERES = 600
API_KEY = os.environ.get("TTS_API_KEY")

app = FastAPI(title="Chatterbox TTS (Latam + LoRA)", version="1.0")
_cabecera = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar(clave: str | None = Security(_cabecera)) -> None:
    if API_KEY and clave != API_KEY:
        raise HTTPException(401, "X-API-Key inválida o ausente")


# ---------- catálogo de modelos y voces ----------
def listar_modelos() -> list[str]:
    """'base' más cada carpeta lora_out* de la raíz que tenga un adaptador."""
    out = ["base"]
    for d in sorted(os.listdir(RAIZ)):
        if d.startswith("lora_out") and os.path.exists(os.path.join(RAIZ, d, "adapter_config.json")):
            out.append(d)
    return out


def listar_voces() -> list[str]:
    os.makedirs(VOCES_DIR, exist_ok=True)
    extra = sorted(f for f in os.listdir(VOCES_DIR) if f.lower().endswith(EXT_AUDIO))
    return ["por_defecto"] + extra


def cargar_registro() -> dict:
    """IDs de voz -> {archivo, nombre, genero}. Vacío si no hay voces.json."""
    try:
        with open(REGISTRO, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def guardar_registro(reg: dict) -> None:
    os.makedirs(VOCES_DIR, exist_ok=True)
    with open(REGISTRO, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)


def detalle_voces() -> list[dict]:
    """Una fila por voz: con id si está registrada, con id null si es un archivo suelto."""
    reg = cargar_registro()
    por_archivo = {v.get("archivo"): (i, v) for i, v in reg.items()}
    filas = [{"id": "por_defecto", "archivo": None, "nombre": "Por defecto (masculina)", "genero": "m"}]
    for f in listar_voces()[1:]:
        i, v = por_archivo.get(f, (None, {}))
        filas.append({"id": i, "archivo": f, "nombre": v.get("nombre") or f, "genero": v.get("genero")})
    return filas


# ---------- gestor del modelo (uno a la vez) ----------
class Gestor:
    def __init__(self):
        self.modelo = None
        self.nombre = None
        self.lock = threading.Lock()

    def obtener(self, nombre: str):
        """Debe llamarse con self.lock tomado."""
        if self.nombre == nombre and self.modelo is not None:
            return self.modelo
        self.modelo, self.nombre = None, None
        gc.collect()
        torch.cuda.empty_cache()
        modelo = carga_ligera.cargar_modelo_completo()
        if nombre != "base":
            # merge_and_unload modifica los pesos: por eso se parte siempre de un modelo recién cargado
            peft = PeftModel.from_pretrained(modelo.t3.tfmr, os.path.join(RAIZ, nombre))
            modelo.t3.tfmr = peft.merge_and_unload()
        self.modelo, self.nombre = modelo, nombre
        return modelo


gestor = Gestor()
registro_lock = threading.Lock()


def ruta_voz(voz: str) -> str:
    """`voz` puede ser 'por_defecto', un ID registrado en voces.json o el nombre de un archivo de voces/."""
    if voz == "por_defecto":
        return VOZ_POR_DEFECTO
    if voz in (reg := cargar_registro()):
        voz = reg[voz]["archivo"]
    ruta = os.path.join(VOCES_DIR, os.path.basename(voz))  # basename: evita salir de la carpeta
    if not os.path.exists(ruta):
        raise HTTPException(404, f"Voz '{voz}' no encontrada. IDs: {list(reg)}. Archivos: {listar_voces()}")
    return ruta


# ---------- rutas ----------
@app.get("/health")
def health():
    return {"ok": True, "modelo_cargado": gestor.nombre, "cuda": torch.cuda.is_available()}


@app.get("/modelos", dependencies=[Depends(verificar)])
def modelos():
    return {"modelos": listar_modelos(), "cargado": gestor.nombre}


@app.get("/voces", dependencies=[Depends(verificar)])
def voces():
    return {"voces": listar_voces(), "voces_detalle": detalle_voces()}


@app.get("/idiomas", dependencies=[Depends(verificar)])
def idiomas():
    return SUPPORTED_LANGUAGES


@app.post("/voces", dependencies=[Depends(verificar)])
async def subir_voz(
    archivo: UploadFile = File(...),
    id: str | None = Form(None, description="ID estable para usar en /tts (letras, números, _ y -). Si ya existe, se reasigna"),
    nombre: str | None = Form(None, description="nombre legible (solo informativo)"),
    genero: str | None = Form(None, description="'m' o 'f' (solo informativo)"),
):
    """Guarda un audio de referencia (5 a 15 s, una sola voz, sin ruido). Úsalo luego por su ID o por el nombre del archivo."""
    fichero = os.path.basename(archivo.filename or "")
    if not fichero.lower().endswith(EXT_AUDIO):
        raise HTTPException(400, "Formato no soportado: usa wav, mp3, flac u ogg")
    if id is not None and (not ID_VALIDO.match(id) or id == "por_defecto"):
        raise HTTPException(400, "id inválido: 1 a 40 caracteres entre letras, números, _ y -; 'por_defecto' está reservado")
    if genero is not None and genero not in ("m", "f"):
        raise HTTPException(400, "genero debe ser 'm' o 'f'")
    os.makedirs(VOCES_DIR, exist_ok=True)
    with open(os.path.join(VOCES_DIR, fichero), "wb") as f:
        f.write(await archivo.read())
    if id:
        with registro_lock:
            reg = cargar_registro()
            reg[id] = {"archivo": fichero, "nombre": nombre or fichero, "genero": genero}
            guardar_registro(reg)
    return {"voz": fichero, "id": id}


@app.post("/tts", dependencies=[Depends(verificar)], response_class=Response,
          responses={200: {"content": {"audio/wav": {}}}})
def tts(
    texto: str = Form(...),
    modelo: str = Form("base", description="'base' o una carpeta lora_out* (ver /modelos)"),
    voz: str = Form("por_defecto", description="ID de voz (voces_detalle de /voces) o nombre de archivo; se ignora si subes voz_archivo"),
    idioma: str = Form("es"),
    exaggeration: float = Form(0.5, ge=0.0, le=2.0),
    cfg_weight: float = Form(0.5, ge=0.0, le=1.0),
    temperature: float = Form(0.8, ge=0.05, le=2.0),
    voz_archivo: UploadFile | None = File(None, description="audio de referencia puntual"),
):
    """Devuelve el audio generado (audio/wav). Cabeceras: X-Segundos-Generacion, X-Duracion-Audio."""
    texto = texto.strip()
    if not texto:
        raise HTTPException(400, "El texto está vacío")
    if len(texto) > MAX_CARACTERES:
        raise HTTPException(400, f"Texto demasiado largo ({len(texto)} > {MAX_CARACTERES} caracteres)")
    if modelo not in listar_modelos():
        raise HTTPException(404, f"Modelo '{modelo}' no existe. Disponibles: {listar_modelos()}")
    if idioma.lower() not in SUPPORTED_LANGUAGES:
        raise HTTPException(400, f"Idioma no soportado. Usa uno de: {list(SUPPORTED_LANGUAGES)}")

    tmp = None
    try:
        if voz_archivo is not None and voz_archivo.filename:
            sufijo = os.path.splitext(voz_archivo.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo) as f:
                f.write(voz_archivo.file.read())
                tmp = f.name
            ref = tmp
        else:
            ref = ruta_voz(voz)

        t0 = time.time()
        with gestor.lock:
            m = gestor.obtener(modelo)
            wav = m.generate(texto, language_id=idioma.lower(), audio_prompt_path=ref,
                             exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
            sr = m.sr
        seg = time.time() - t0

        datos = wav.squeeze().detach().cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, datos, sr, format="WAV")
        return Response(buf.getvalue(), media_type="audio/wav", headers={
            "X-Segundos-Generacion": f"{seg:.1f}", "X-Duracion-Audio": f"{len(datos) / sr:.2f}",
            "X-Modelo": modelo})
    except HTTPException:
        raise
    except torch.cuda.OutOfMemoryError:
        gestor.modelo, gestor.nombre = None, None
        gc.collect()
        torch.cuda.empty_cache()
        raise HTTPException(503, "Sin memoria de GPU: prueba con un texto más corto")
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
