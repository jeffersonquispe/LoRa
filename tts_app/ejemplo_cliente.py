"""Ejemplo para quien consume la API (estilo ElevenLabs). Solo necesita `requests`.

    set TTS_URL=https://<dominio>.salad.cloud
    set SALAD_API_KEY=<clave de Salad>        (solo si la API está en Salad)
    set TTS_API_KEY=<clave de la API>         (solo si el servidor la exige)
    python ejemplo_cliente.py
"""
import os
import sys

import requests

URL = os.environ.get("TTS_URL", "http://localhost:8000").rstrip("/")
CABECERAS = {}
if os.environ.get("SALAD_API_KEY"):
    CABECERAS["Salad-Api-Key"] = os.environ["SALAD_API_KEY"]
if os.environ.get("TTS_API_KEY"):
    CABECERAS["xi-api-key"] = os.environ["TTS_API_KEY"]


def voces() -> list[dict]:
    r = requests.get(f"{URL}/v1/voices", headers=CABECERAS, timeout=30)
    r.raise_for_status()
    return r.json()["voices"]


def hablar(texto: str, voice_id: str, salida: str, stability: float = 0.5, style: float = 0.3) -> float:
    """Genera `texto` con la voz `voice_id` y lo guarda como WAV. Devuelve la duración en segundos."""
    r = requests.post(
        f"{URL}/v1/text-to-speech/{voice_id}",
        headers=CABECERAS,
        json={"text": texto, "language_code": "es",
              "voice_settings": {"stability": stability, "style": style}},
        timeout=100,  # el gateway de Salad corta a los 100 s
    )
    if not r.ok:
        raise RuntimeError(f"{r.status_code}: {r.text[:200]}")
    with open(salida, "wb") as f:
        f.write(r.content)
    return float(r.headers["X-Duracion-Audio"])


if __name__ == "__main__":
    ids = [v["voice_id"] for v in voces()]
    print("voces:", ids)
    for vid in [i for i in ids if i != "por_defecto"]:
        seg = hablar(f"Hola, esta es una prueba con la voz {vid}.", vid, f"prueba_{vid}.wav")
        print(f"{vid}: prueba_{vid}.wav ({seg:.1f} s)")
    sys.exit(0)
