"""Utilidades de carga para equipos con poca RAM / memoria virtual libre.

Importa este módulo ANTES de usar ChatterboxMultilingualTTS: parchea la carga de safetensors
(sin mmap, directo a GPU) y la de S3Gen (s3gen_v3.pt no trae los buffers del tokenizador).
"""
import json
import os
import shutil
import struct
import sys

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatterbox", "src"))

import chatterbox.mtl_tts as mtl  # noqa: E402
from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: E402
from chatterbox.models.t3 import T3  # noqa: E402
from chatterbox.models.t3.modules.t3_config import T3Config  # noqa: E402

LATAM = "ResembleAI/Chatterbox-Multilingual-es-mx-latam"
BASE = "ResembleAI/chatterbox"
CKPT_LATAM = "ckpt_latam"
T3_LATAM = "t3_es_mx_latam.safetensors"

_DT = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "I64": torch.int64,
       "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def cargar_safetensors(path, device="cuda"):
    """Lee un .safetensors tensor por tensor, sin mmap."""
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


mtl.load_safetensors = lambda path, device="cuda": cargar_safetensors(path, device)

_s3_load = mtl.S3Gen.load_state_dict
mtl.S3Gen.load_state_dict = lambda self, sd, strict=True, **kw: _s3_load(self, sd, strict=False, **kw)


def preparar_ckpt_latam(ckpt=CKPT_LATAM):
    """Arma la carpeta con los pesos Latam + ve.pt del repo base (descarga si falta)."""
    os.makedirs(ckpt, exist_ok=True)
    for repo, archivo, destino in [
        (LATAM, T3_LATAM, T3_LATAM),
        (LATAM, "s3gen_v3.pt", "s3gen.pt"),
        (LATAM, "grapheme_mtl_merged_expanded_v1.json", "grapheme_mtl_merged_expanded_v1.json"),
        (BASE, "ve.pt", "ve.pt"),
    ]:
        if not os.path.exists(os.path.join(ckpt, destino)):
            shutil.copy(hf_hub_download(repo, archivo), os.path.join(ckpt, destino))
    return ckpt


def cargar_modelo_completo(ckpt=CKPT_LATAM, t3_file=T3_LATAM, device="cuda"):
    preparar_ckpt_latam(ckpt)
    return ChatterboxMultilingualTTS.from_local(ckpt, device, t3_model=t3_file)


def cargar_t3(ckpt=CKPT_LATAM, t3_file=T3_LATAM, device="cuda") -> T3:
    """Solo el T3 (lo único que se entrena)."""
    preparar_ckpt_latam(ckpt)
    t3 = T3(T3Config.multilingual())
    t3.load_state_dict(cargar_safetensors(os.path.join(ckpt, t3_file), device))
    return t3.to(device).eval()
