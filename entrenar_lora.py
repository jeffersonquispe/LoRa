"""Paso 2: entrena un adaptador LoRA sobre el T3 (el Llama de ~520M) de Chatterbox.

Solo se entrenan las matrices LoRA de las capas de atención y MLP del backbone Llama;
todo lo demás (embeddings, cabezas, perceiver, s3gen, voice encoder) queda congelado.

Pérdida: predicción del siguiente token de voz.
  entrada  : [cond | texto | BOS, s1 ... sn]
  objetivo : [s1 ... sn, EOS]   (la cabeza de voz en la posición de BOS predice s1, etc.)
Batch = 1 con acumulación de gradiente: T3 no usa máscara de atención, así que no hay padding.
"""
import argparse
import math
import os
import random

import carga_ligera
import torch
import torch.nn.functional as F
from chatterbox.models.t3.modules.cond_enc import T3Cond
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--datos", nargs="+", default=["lora_data/features.pt"],
                help="uno o varios features_*.pt; cada archivo = un hablante")
ap.add_argument("--salida", default="lora_out")
ap.add_argument("--epocas", type=int, default=5)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--acum", type=int, default=8, help="pasos de gradiente acumulados por actualización")
ap.add_argument("--rango", type=int, default=16)
ap.add_argument("--alpha", type=int, default=32)
ap.add_argument("--val", type=float, default=0.05, help="fracción de validación")
ap.add_argument("--text_drop", type=float, default=0.1, help="prob. de anular el texto (mantiene el CFG sano)")
ap.add_argument("--max_rep", type=int, default=4, help="máx. veces que se repite cada ejemplo de una voz pequeña por época")
ap.add_argument("--sin_balance", action="store_true", help="no igualar el número de ejemplos por hablante en cada época")
ap.add_argument("--semilla", type=int, default=0)
ap.add_argument("--max_pasos", type=int, default=0, help="para pruebas rápidas; 0 = sin límite")
args = ap.parse_args()

DEV = "cuda"
# Solo las capas lineales del backbone Llama (no toca el perceiver ni las cabezas)
OBJETIVO = r".*layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def fmt(det):
    return "  ".join(f"[{k}] {x:.4f}" for k, x in det.items())


def armar_cond(t3, d, otro, emotion=0.5):
    """Voz/prompt de OTRO clip del mismo hablante, para que el modelo no copie el audio objetivo."""
    return T3Cond(
        speaker_emb=otro["speaker_emb"][None],
        cond_prompt_speech_tokens=otro["prompt_tokens"][None],
        emotion_adv=emotion * torch.ones(1, 1, 1),
    ).to(device=DEV)


def perdida(t3, d, otro, drop_texto=False):
    hp = t3.hp
    texto = d["texto_tokens"][None].to(DEV)
    habla = d["speech_tokens"].to(DEV)
    speech_in = torch.cat([habla.new_tensor([hp.start_speech_token]), habla])[None]
    objetivo = torch.cat([habla, habla.new_tensor([hp.stop_speech_token])])

    cond_emb = t3.prepare_conditioning(armar_cond(t3, d, otro))
    te = t3.text_emb(texto)
    if drop_texto:
        te = torch.zeros_like(te)  # igual que la rama incondicional del CFG en inferencia
    te = te + t3.text_pos_emb(texto)
    se = t3.speech_emb(speech_in) + t3.speech_pos_emb(speech_in)

    embeds = torch.cat([cond_emb, te, se], dim=1)
    h = t3.tfmr(inputs_embeds=embeds, use_cache=False, return_dict=True).last_hidden_state
    ini = cond_emb.size(1) + te.size(1)
    logits = t3.speech_head(h[:, ini:])[0]  # (n+1, vocab)
    return F.cross_entropy(logits.float(), objetivo)


@torch.no_grad()
def evaluar(t3, val, por_voz):
    """Devuelve (val medio por hablante, detalle por hablante)."""
    tot = {}
    for d in val:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            tot.setdefault(d["voz"], []).append(perdida(t3, d, random.choice(por_voz[d["voz"]])).item())
    det = {v: sum(x) / len(x) for v, x in tot.items()}
    return sum(det.values()) / max(len(det), 1), det


def epoca_balanceada(por_voz, balancear, max_rep=4):
    """Lista de ejemplos de la época; con balance, las voces pequeñas se repiten hasta igualar a la mayor,
    sin pasar de max_rep veces cada ejemplo (más allá se sobreajusta)."""
    if not balancear:
        lista = [d for g in por_voz.values() for d in g]
    else:
        n = max(len(g) for g in por_voz.values())
        lista = []
        for g in por_voz.values():
            m = min(n, len(g) * max_rep)
            lista += g * (m // len(g)) + random.sample(g, m % len(g))
    random.shuffle(lista)
    return lista


def main() -> None:
    random.seed(args.semilla)
    torch.manual_seed(args.semilla)

    # Un archivo = un hablante. El prompt/embedding de condicionamiento siempre sale del mismo hablante.
    por_voz, val = {}, []
    for ruta in args.datos:
        voz = os.path.splitext(os.path.basename(ruta))[0]
        g = torch.load(ruta, weights_only=False)
        for d in g:
            d["voz"] = voz
        random.shuffle(g)
        n_val = max(1, int(len(g) * args.val))
        val += g[:n_val]
        por_voz[voz] = g[n_val:]
        print(f"{voz}: train={len(g) - n_val}  val={n_val}")
    n_epoca = len(epoca_balanceada(por_voz, not args.sin_balance, args.max_rep))

    t3 = carga_ligera.cargar_t3()
    for p in t3.parameters():
        p.requires_grad_(False)
    cfg = LoraConfig(r=args.rango, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=OBJETIVO, bias="none")
    t3.tfmr = get_peft_model(t3.tfmr, cfg)
    t3.tfmr.print_trainable_parameters()
    t3.eval()

    params = [p for p in t3.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    pasos_totales = math.ceil(n_epoca * args.epocas / args.acum)
    warm = max(1, int(0.05 * pasos_totales))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + math.cos(math.pi * min(s / pasos_totales, 1.0))))
    )

    os.makedirs(args.salida, exist_ok=True)
    v0, det0 = evaluar(t3, val, por_voz)
    print(f"val inicial: {v0:.4f}  {fmt(det0)}")
    mejor, paso, micro = float("inf"), 0, 0
    for ep in range(1, args.epocas + 1):
        train = epoca_balanceada(por_voz, not args.sin_balance, args.max_rep)
        acum_loss, n = 0.0, 0
        for d in train:
            g = por_voz[d["voz"]]
            otro = random.choice(g)
            while otro is d and len(g) > 1:
                otro = random.choice(g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = perdida(t3, d, otro, drop_texto=random.random() < args.text_drop)
            (loss / args.acum).backward()
            acum_loss += loss.item()
            n += 1
            micro += 1
            if micro % args.acum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                paso += 1
                if paso % 10 == 0:
                    print(f"  paso {paso}/{pasos_totales}  loss {acum_loss / n:.4f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
                if args.max_pasos and paso >= args.max_pasos:
                    break
        v, det = evaluar(t3, val, por_voz)
        print(f"época {ep}/{args.epocas}  train {acum_loss / max(n, 1):.4f}  val {v:.4f}  {fmt(det)}", flush=True)
        if v < mejor:
            mejor = v
            t3.tfmr.save_pretrained(args.salida)
            print(f"  -> mejor val, adaptador guardado en {args.salida}")
        if args.max_pasos and paso >= args.max_pasos:
            break
    print(f"Listo. Mejor val: {mejor:.4f}  (pico de GPU: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB)")


if __name__ == "__main__":
    main()
