import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from elevenlabs_tts import generar_audio

ENTRADA = "dataset_frases.csv"
OUT = "dataset_audios"
HILOS = 4
LIMITE = int(sys.argv[1]) if len(sys.argv) > 1 else None  # p. ej. `python audios_dataset.py 10`

cuota_agotada = False


def procesar(fila: dict) -> str:
    global cuota_agotada
    ruta = os.path.join(OUT, f"{fila['id']}.mp3")
    if cuota_agotada or os.path.exists(ruta):
        return "skip"
    try:
        generar_audio(fila["texto"], ruta)
        return "ok"
    except Exception as e:
        if "quota" in str(e).lower() or "401" in str(e) or "402" in str(e):
            cuota_agotada = True
        print(f"  {fila['id']} falló: {str(e)[:200]}", file=sys.stderr)
        return "error"


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(ENTRADA, encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    if LIMITE:
        filas = filas[:LIMITE]
    cuenta = {"ok": 0, "skip": 0, "error": 0}
    with ThreadPoolExecutor(HILOS) as ex:
        for i, r in enumerate(ex.map(procesar, filas), 1):
            cuenta[r] += 1
            if i % 25 == 0:
                print(f"  {i}/{len(filas)} {cuenta}", flush=True)
    print(f"Listo: {cuenta}")
    if cuota_agotada:
        print("Se detuvo por cuota/credenciales de ElevenLabs.")


if __name__ == "__main__":
    main()
