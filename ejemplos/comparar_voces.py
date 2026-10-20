from pathlib import Path

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

CARPETA = Path(r"C:\Users\Jeff\Desktop\cortador de audio")
SALIDA = Path("comparacion")
PARTES = range(1, 11)  # parte_1 ... parte_10
TEXTO = "Hola, ¿cómo estás? Este es un ejemplo de mi voz clonada con Chatterbox."

SALIDA.mkdir(exist_ok=True)
model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")

for n in PARTES:
    ref = CARPETA / f"parte_{n}.mp3"
    if not ref.exists():
        print(f"Salto parte_{n}: no existe")
        continue
    wav = model.generate(TEXTO, language_id="es", audio_prompt_path=str(ref))
    destino = SALIDA / f"voz_parte_{n}.wav"
    ta.save(str(destino), wav, model.sr)
    print(f"Listo: {destino}")
