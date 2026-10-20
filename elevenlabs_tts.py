import os
import sys
import requests

from secretos import obtener

API_KEY = obtener("ELEVENLABS_API_KEY")  # desde .env o variable de entorno
VOICE_ID = "GcbypXUfJn5DbptRc2U7"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_DIR = "audios_elevenlabs"

# Textos a generar: nombre_de_archivo -> texto
TEXTOS = {
    "audio_01": "Hola, ¿cómo estás? Este es un audio generado con ElevenLabs.",
    "audio_02": "Bienvenido. Aquí puedes agregar todos los textos que necesites.",
}


def generar_audio(texto: str, ruta: str, voice_id: str = VOICE_ID) -> None:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    body = {
        "text": texto,
        "model_id": MODEL_ID,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    r = requests.post(url, headers=headers, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"Error {r.status_code}: {r.text}")
    with open(ruta, "wb") as f:
        f.write(r.content)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for nombre, texto in TEXTOS.items():
        ruta = os.path.join(OUTPUT_DIR, f"{nombre}.mp3")
        print(f"Generando {ruta} ...")
        try:
            generar_audio(texto, ruta)
        except Exception as e:
            print(f"  Falló: {e}", file=sys.stderr)
    print("Listo.")


if __name__ == "__main__":
    main()
