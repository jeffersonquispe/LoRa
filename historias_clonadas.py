import glob
import os
import sys

import torchaudio as ta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "chatterbox", "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from historias_tts import generar_historia

DOWNLOADS = r"C:\Users\Jeff\Downloads"
CARPETAS = {
    "mexico": ("mexico_casual-*", "una niña mexicana que encuentra un barrilete mágico en el mercado"),
    "venezuela": ("venezuela_casual-*", "un pescador venezolano que conversa con una tortuga en la playa"),
    "voces_dataset": ("voces dataset-*", "un viajero que llega de noche a un pueblo donde todos sueñan en voz alta"),
}
OUT = "historias_clonadas"


def clips(patron: str) -> list[str]:
    archivos = []
    for carpeta in glob.glob(os.path.join(DOWNLOADS, patron)):
        if os.path.isdir(carpeta):
            for ext in ("wav", "flac", "mp3"):
                archivos += glob.glob(os.path.join(carpeta, "**", f"*.{ext}"), recursive=True)
    return sorted(archivos)


def main() -> None:
    model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
    os.makedirs(OUT, exist_ok=True)
    for nombre, (patron, tema) in CARPETAS.items():
        print(f"== {nombre}: {tema}")
        ruta_txt = os.path.join(OUT, f"{nombre}.txt")
        if os.path.exists(ruta_txt):
            historia = open(ruta_txt, encoding="utf-8").read()
        else:
            historia = generar_historia(tema)
        print(historia, "\n")
        with open(os.path.join(OUT, f"{nombre}.txt"), "w", encoding="utf-8") as f:
            f.write(historia)
        for ref in clips(patron):
            salida = os.path.join(OUT, f"{nombre}_{os.path.splitext(os.path.basename(ref))[0]}.wav")
            if os.path.exists(salida):
                continue
            try:
                wav = model.generate(historia, language_id="es", audio_prompt_path=ref)
                ta.save(salida, wav, model.sr)
                print(f"  -> {salida}")
            except Exception as e:
                print(f"  Falló {ref}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
