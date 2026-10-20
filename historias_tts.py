import os
import sys
from openai import OpenAI

from secretos import obtener

from elevenlabs_tts import generar_audio, OUTPUT_DIR

NVIDIA_API_KEY = obtener("NVIDIA_API_KEY")  # desde .env o variable de entorno
MODEL = "z-ai/glm-5.3-flash"

TEMAS = [
    "un gato que descubre una puerta secreta en la casa",
    "un niño que encuentra una estrella caída en el jardín",
]

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)


def generar_historia(tema: str) -> str:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Eres un cuentacuentos. Escribe en español, solo el texto narrado, sin títulos ni listas ni markdown.",
            },
            {
                "role": "user",
                "content": f"Escribe una historia breve (unas 100 palabras) sobre {tema}.",
            },
        ],
        temperature=0.7,
        top_p=1,
        max_tokens=1024,
        stream=False,
    )
    return completion.choices[0].message.content.strip()


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, tema in enumerate(TEMAS, 1):
        print(f"[{i}] Historia sobre: {tema}")
        try:
            historia = generar_historia(tema)
            print(historia, "\n")
            base = os.path.join(OUTPUT_DIR, f"historia_{i:02d}")
            with open(base + ".txt", "w", encoding="utf-8") as f:
                f.write(historia)
            generar_audio(historia, base + ".mp3")
            print(f"  -> {base}.mp3\n")
        except Exception as e:
            print(f"  Falló: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
