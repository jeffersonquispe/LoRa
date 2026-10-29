"""Toma los primeros N clips válidos de E:\archive\transcript.txt (carpeta 19demarzo) y los deja como dataset.

transcript.txt: ruta|texto|texto|segundos. Solo se usan las filas de la carpeta indicada.

Salida (por defecto en voces/archive_19marzo/): dataset_frases.csv (id,texto,palabras) y dataset_audios/NNNN.wav
(mono, 24 kHz, 16 bit, recorte de silencios y nivel normalizado).

Luego:
  python preparar_datos_lora.py --csv voces/archive_19marzo/dataset_frases.csv \
      --audios voces/archive_19marzo/dataset_audios --ext wav --salida lora_data/features_archive_19marzo.pt
"""
import argparse
import csv
import os
import re

import librosa
import numpy as np
import soundfile as sf

BASE = r"E:\archive"

ap = argparse.ArgumentParser()
ap.add_argument("--transcript", default=os.path.join(BASE, "transcript.txt"))
ap.add_argument("--carpeta", default="19demarzo")
ap.add_argument("--n", type=int, default=1000, help="cuántos clips válidos conservar (los primeros)")
ap.add_argument("--salida", default=os.path.join("voces", "archive_19marzo"))
ap.add_argument("--sr", type=int, default=24000)
ap.add_argument("--pico", type=float, default=0.95)
ap.add_argument("--margen", type=float, default=0.15)
ap.add_argument("--min_seg", type=float, default=1.0)
ap.add_argument("--max_seg", type=float, default=15.0)
ap.add_argument("--min_palabras", type=int, default=3)
args = ap.parse_args()


def limpiar_texto(t: str) -> str:
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"[,;:\-–—\s]+$", "", t)  # los cortes de frase acaban en coma: se cierran con punto
    t = t[0].upper() + t[1:]
    if t[-1] not in ".!?…":
        t += "."
    return t


def main() -> None:
    dir_audio = os.path.join(args.salida, "dataset_audios")
    os.makedirs(dir_audio, exist_ok=True)

    filas, vistos, omitidos = [], set(), {"sin_audio": 0, "duplicado": 0, "corto": 0, "duracion": 0}
    with open(args.transcript, encoding="utf-8") as f:
        for linea in f:
            partes = linea.rstrip("\n").split("|")
            if len(partes) < 2 or not partes[0].startswith(args.carpeta + "/"):
                continue
            if len(filas) >= args.n:
                break
            ruta = os.path.join(os.path.dirname(args.transcript), partes[0])
            if not os.path.exists(ruta):
                omitidos["sin_audio"] += 1
                continue
            if len(partes[1].split()) < args.min_palabras:
                omitidos["corto"] += 1
                continue
            texto = limpiar_texto(partes[1])
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
