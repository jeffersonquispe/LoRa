import glob
import json
import struct
import os
import shutil
import sys

import torch
import torchaudio as ta
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatterbox", "src"))

import chatterbox.mtl_tts as mtl
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# Carga el safetensors tensor por tensor directo a la GPU, sin mmap
# (poca memoria virtual libre en este equipo)
_DT = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
       "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def _load_sin_mmap(path, device="cuda"):
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for k, v in header.items():
            if k == "__metadata__":
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
            t = torch.frombuffer(buf, dtype=_DT[v["dtype"]]) if buf else torch.empty(0, dtype=_DT[v["dtype"]])
            out[k] = t.reshape(v["shape"]).to(device)
    return out


mtl.load_safetensors = _load_sin_mmap

# s3gen_v3.pt no trae los buffers del tokenizador (se recalculan al crear el modelo)
_s3_load = mtl.S3Gen.load_state_dict
mtl.S3Gen.load_state_dict = lambda self, sd, strict=True, **kw: _s3_load(self, sd, strict=False, **kw)

LATAM = "ResembleAI/Chatterbox-Multilingual-es-mx-latam"
BASE = "ResembleAI/chatterbox"
CKPT = "ckpt_latam"
OUT = "prueba_latam"
REF = glob.glob(r"C:\Users\Jeff\Downloads\mexico_casual-*\**\*.flac", recursive=True)[0]
TEXTO = "Tomás despertó con una luz tibia asomando por la ventana y bajó descalzo al jardín. Entre las margaritas brillaba una estrella pequeña."


def preparar_ckpt() -> None:
    os.makedirs(CKPT, exist_ok=True)
    for repo, archivo, destino in [
        (LATAM, "t3_es_mx_latam.safetensors", "t3_es_mx_latam.safetensors"),
        (LATAM, "s3gen_v3.pt", "s3gen.pt"),
        (LATAM, "grapheme_mtl_merged_expanded_v1.json", "grapheme_mtl_merged_expanded_v1.json"),
        (BASE, "ve.pt", "ve.pt"),
    ]:
        if not os.path.exists(os.path.join(CKPT, destino)):
            shutil.copy(hf_hub_download(repo, archivo), os.path.join(CKPT, destino))


def main() -> None:
    preparar_ckpt()
    os.makedirs(OUT, exist_ok=True)
    model = ChatterboxMultilingualTTS.from_local(CKPT, "cuda", t3_model="t3_es_mx_latam.safetensors")
    print("Modelo Latam cargado. Referencia:", REF)
    wav = model.generate(TEXTO, language_id="es", audio_prompt_path=REF)
    ta.save(os.path.join(OUT, "latam_mexico.wav"), wav, model.sr)
    print("OK ->", os.path.join(OUT, "latam_mexico.wav"))


if __name__ == "__main__":
    main()
