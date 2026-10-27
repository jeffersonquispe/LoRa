"""Sube o baja adaptadores LoRA (carpetas lora_out*) al bucket S3 de pesos.

Credenciales: variables de entorno estándar de AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION),
nunca en el código. Bucket por defecto: variable PESOS_BUCKET.

  python pesos_s3.py listar
  python pesos_s3.py subir lora_out_2voces [lora_out_2000 ...]
  python pesos_s3.py bajar lora_out_2voces
"""
import os
import sys

import boto3

BUCKET = os.environ.get("PESOS_BUCKET", "chatterbox-lora-pesos-866923470465")
s3 = boto3.client("s3")


def subir(carpeta: str) -> None:
    carpeta = carpeta.rstrip("/\\")
    n = 0
    for raiz, _, archivos in os.walk(carpeta):
        for a in archivos:
            ruta = os.path.join(raiz, a)
            clave = os.path.relpath(ruta, ".").replace(os.sep, "/")
            s3.upload_file(ruta, BUCKET, clave)
            n += 1
    print(f"{carpeta}: {n} archivos -> s3://{BUCKET}/{carpeta}/")


def bajar(carpeta: str) -> None:
    n = 0
    for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=carpeta.rstrip("/") + "/"):
        for o in pag.get("Contents", []):
            os.makedirs(os.path.dirname(o["Key"]), exist_ok=True)
            s3.download_file(BUCKET, o["Key"], o["Key"])
            n += 1
    print(f"{carpeta}: {n} archivos descargados")


def listar() -> None:
    for pag in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        for o in pag.get("Contents", []):
            print(f"{o['Size'] / 2**20:8.1f} MB  {o['Key']}")


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["listar"]
    if cmd == "listar":
        listar()
    elif cmd in ("subir", "bajar") and args:
        for a in args:
            (subir if cmd == "subir" else bajar)(a)
    else:
        sys.exit(__doc__)
