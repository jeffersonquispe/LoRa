"""Convierte el corpus de voz mexicana (frases_new_mex.txt + vozMexico_N.wav) en un dataset listo para preparar_datos_lora.py.

La línea N de frases_new_mex.txt es el texto del audio vozMexico_N.wav (verificado contra DS_frases_mexico_new.csv).

Salida (por defecto en voces/mex_nueva/):
  dataset_frases.csv   id,texto,palabras   (mismo formato que dataset_frases.csv)
  dataset_audios/NNN.wav   mono, 24 kHz, 16 bit, recorte de silencios y nivel normalizado

Luego:
  python preparar_datos_lora.py --csv voces/mex_nueva/dataset_frases.csv \
      --audios voces/mex_nueva/dataset_audios --ext wav --salida lora_data/features_mex_nueva.pt
"""
import argparse
import csv
import os
import re

import librosa
import numpy as np
import soundfile as sf

BASE = r"E:\Trabajo\TTS\voz mexicana nueva\audios total"

ap = argparse.ArgumentParser()
ap.add_argument("--textos", default=os.path.join(BASE, "frases_new_mex.txt"))
ap.add_argument("--audios", default=os.path.join(BASE, "audios total"))
ap.add_argument("--prefijo", default="vozMexico_")
ap.add_argument("--salida", default=os.path.join("voces", "mex_nueva"))
ap.add_argument("--sr", type=int, default=24000)
ap.add_argument("--pico", type=float, default=0.95, help="pico objetivo tras normalizar")
ap.add_argument("--margen", type=float, default=0.15, help="segundos de silencio que se conservan en cada borde")
ap.add_argument("--min_seg", type=float, default=1.0)
ap.add_argument("--max_seg", type=float, default=15.0)
ap.add_argument("--min_palabras", type=int, default=2)
args = ap.parse_args()


def limpiar_texto(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?…":
        t += "."
    return t


def main() -> None:
    with open(args.textos, encoding="utf-8") as f:
        lineas = f.read().splitlines()

    dir_audio = os.path.join(args.salida, "dataset_audios")
    os.makedirs(dir_audio, exist_ok=True)

    filas, vistos, omitidos = [], set(), {"vacia": 0, "sin_audio": 0, "duplicado": 0, "corto": 0, "duracion": 0}
    for n, linea in enumerate(lineas, start=1):
        if len(linea.split()) < args.min_palabras:
            omitidos["vacia" if not linea.strip() else "corto"] += 1
            continue
        ruta = os.path.join(args.audios, f"{args.prefijo}{n}.wav")
        if not os.path.exists(ruta):
            omitidos["sin_audio"] += 1
            continue
        texto = limpiar_texto(linea)
        if texto.lower() in vistos:
            omitidos["duplicado"] += 1
            continue

        y, _ = librosa.load(ruta, sr=args.sr, mono=True)
        _, idx = librosa.effects.trim(y, top_db=35)
        m = int(args.margen * args.sr)
        y = y[max(idx[0] - m, 0): idx[1] + m]
        if not args.min_seg <= len(y) / args.sr <= args.max_seg:
            omitidos["duracion"] += 1
            continue
        y = y - y.mean()
        pico = np.abs(y).max()
        if pico > 0:
            y = y * (args.pico / pico)

        vistos.add(texto.lower())
        id_ = f"{len(filas) + 1:04d}"
        sf.write(os.path.join(dir_audio, id_ + ".wav"), y, args.sr, subtype="PCM_16")
        filas.append({"id": id_, "texto": texto, "palabras": len(texto.split())})

    with open(os.path.join(args.salida, "dataset_frases.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "texto", "palabras"])
        w.writeheader()
        w.writerows(filas)

    dur = sum(sf.info(os.path.join(dir_audio, r["id"] + ".wav")).duration for r in filas)
    print(f"{len(filas)} clips, {dur / 60:.1f} min, {args.sr} Hz mono -> {args.salida}")
    print("omitidos:", {k: v for k, v in omitidos.items() if v})


if __name__ == "__main__":
    main()
