"""Paso 1 del LoRA: convierte el dataset (CSV + audios) en tensores listos para entrenar.

Por cada frase guarda:
  - texto_tokens : tokens de texto con [SOT] ... [EOT]
  - speech_tokens: tokens de voz del audio completo (lo que T3 aprende a predecir)
  - prompt_tokens: tokens del primer tramo del audio (condicionamiento, máx. 150)
  - speaker_emb  : embedding de voz (VoiceEncoder, 256)
"""
import argparse
import csv
import os

import carga_ligera  # noqa: F401  (debe ir antes de usar chatterbox)
import librosa
import torch
import torch.nn.functional as F
from chatterbox.mtl_tts import punc_norm, S3_SR

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="dataset_frases.csv")
ap.add_argument("--audios", default="dataset_audios")
ap.add_argument("--ext", default="mp3")
ap.add_argument("--salida", default="lora_data/features.pt")
ap.add_argument("--idioma", default="es")
ap.add_argument("--max_speech_tokens", type=int, default=500, help="descarta clips más largos (25 tokens = 1 s)")
args = ap.parse_args()


def main() -> None:
    model = carga_ligera.cargar_modelo_completo()
    t3, tok, s3tok, ve = model.t3, model.tokenizer, model.s3gen.tokenizer, model.ve
    sot, eot = t3.hp.start_text_token, t3.hp.stop_text_token
    plen = t3.hp.speech_cond_prompt_len

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))

    datos, omitidas = [], 0
    for i, fila in enumerate(filas, 1):
        ruta = os.path.join(args.audios, f"{fila['id']}.{args.ext}")
        if not os.path.exists(ruta):
            omitidas += 1
            continue
        wav, _ = librosa.load(ruta, sr=S3_SR)

        with torch.inference_mode():
            speech, _ = s3tok.forward([wav])
            prompt, _ = s3tok.forward([wav[: 6 * S3_SR]], max_len=plen)
            spk = torch.from_numpy(ve.embeds_from_wavs([wav], sample_rate=S3_SR)).mean(0)

        speech = speech[0].cpu()
        if speech.numel() > args.max_speech_tokens or speech.numel() < 10:
            omitidas += 1
            continue

        texto = punc_norm(fila["texto"])
        tt = tok.text_to_tokens(texto, language_id=args.idioma).squeeze(0).cpu()
        tt = F.pad(F.pad(tt, (1, 0), value=sot), (0, 1), value=eot)

        datos.append({
            "id": fila["id"],
            "texto": fila["texto"],
            "texto_tokens": tt.long(),
            "speech_tokens": speech.long(),
            "prompt_tokens": prompt[0].cpu().long(),
            "speaker_emb": spk.cpu().float().reshape(-1),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(filas)}", flush=True)

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    torch.save(datos, args.salida)
    seg = sum(d["speech_tokens"].numel() for d in datos) / 25
    print(f"{len(datos)} ejemplos ({seg / 60:.1f} min de audio), {omitidas} omitidos -> {args.salida}")


if __name__ == "__main__":
    main()
