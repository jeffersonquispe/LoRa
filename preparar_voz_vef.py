"""Convierte un hablante de es_ve_female (OpenSLR 75) en un dataset listo para preparar_datos_lora.py.

Salida (por defecto en voces/vef_07508/):
  dataset_frases.csv   id,texto,palabras   (mismo formato que dataset_frases.csv)
  dataset_audios/NNN.wav   mono, 24 kHz, 16 bit, recorte de silencios y nivel normalizado

Luego:
  python preparar_datos_lora.py --csv voces/vef_07508/dataset_frases.csv \
      --audios voces/vef_07508/dataset_audios --ext wav --salida lora_data/features_vef_07508.pt
"""
import argparse
import csv
import os
import re

import librosa
import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--origen", default="es_ve_female")
ap.add_argument("--prefijo", default="vef_07508")
ap.add_argument("--salida", default=None)
ap.add_argument("--sr", type=int, default=24000)
ap.add_argument("--pico", type=float, default=0.95, help="pico objetivo tras normalizar")
ap.add_argument("--margen", type=float, default=0.15, help="segundos de silencio que se conservan en cada borde")
args = ap.parse_args()
salida = args.salida or os.path.join("voces", args.prefijo)


def limpiar_texto(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?…":
        t += "."
    return t


def main() -> None:
    with open(os.path.join(args.origen, "line_index.tsv"), encoding="utf-8") as f:
        textos = dict(l.rstrip("\n").split("\t", 1) for l in f if l.startswith(args.prefijo))

    dir_audio = os.path.join(salida, "dataset_audios")
    os.makedirs(dir_audio, exist_ok=True)

    filas, vistos = [], set()
    for nombre in sorted(textos):
        ruta = os.path.join(args.origen, nombre + ".wav")
        if not os.path.exists(ruta):
            print("sin audio:", nombre)
            continue
        texto = limpiar_texto(textos[nombre])
        if texto.lower() in vistos:
            print("duplicado omitido:", texto)
            continue
        vistos.add(texto.lower())

        y, _ = librosa.load(ruta, sr=args.sr, mono=True)
        recorte, idx = librosa.effects.trim(y, top_db=35)
        m = int(args.margen * args.sr)
        y = y[max(idx[0] - m, 0): idx[1] + m]
        y = y - y.mean()
        pico = np.abs(y).max()
        if pico > 0:
            y = y * (args.pico / pico)

        id_ = f"{len(filas) + 1:03d}"
        sf.write(os.path.join(dir_audio, id_ + ".wav"), y, args.sr, subtype="PCM_16")
        filas.append({"id": id_, "texto": texto, "palabras": len(texto.split())})

    with open(os.path.join(salida, "dataset_frases.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "texto", "palabras"])
        w.writeheader()
        w.writerows(filas)

    dur = sum(sf.info(os.path.join(dir_audio, r["id"] + ".wav")).duration for r in filas)
    print(f"{len(filas)} clips, {dur / 60:.1f} min, {args.sr} Hz mono -> {salida}")


if __name__ == "__main__":
    main()
