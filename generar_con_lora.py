"""Paso 3: genera las mismas frases con el modelo base (Latam) y con el LoRA, para comparar.

Salida: <salida>/base_XX.wav y <salida>/lora_XX.wav
"""
import argparse
import os

import carga_ligera
import torchaudio as ta
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--lora", default="lora_out")
ap.add_argument("--ref", default="dataset_audios/001.mp3", help="audio de referencia de la voz")
ap.add_argument("--salida", default="lora_prueba")
ap.add_argument("--idioma", default="es")
ap.add_argument("--frases", nargs="*", default=[
    "Dime que no vas a dejarme solo con todo este papeleo.",
    "Venga, no pongas esa cara, que no es tan grave.",
    "Esta noche parece que la tormenta no se va a acabar nunca.",
    "No sé cómo lo haces, pero siempre consigues hacerme reír.",
])
args = ap.parse_args()


def generar(model, prefijo):
    for i, f in enumerate(args.frases, 1):
        wav = model.generate(f, language_id=args.idioma, audio_prompt_path=args.ref)
        ruta = os.path.join(args.salida, f"{prefijo}_{i:02d}.wav")
        ta.save(ruta, wav, model.sr)
        print("  ->", ruta, flush=True)


def main() -> None:
    os.makedirs(args.salida, exist_ok=True)
    model = carga_ligera.cargar_modelo_completo()

    print("Modelo base (Latam):")
    generar(model, "base")

    print(f"Con LoRA ({args.lora}):")
    peft = PeftModel.from_pretrained(model.t3.tfmr, args.lora)
    model.t3.tfmr = peft.merge_and_unload()  # LlamaModel con los pesos fusionados
    generar(model, "lora")


if __name__ == "__main__":
    main()
