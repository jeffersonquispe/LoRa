"""Lee las claves desde variables de entorno o desde un archivo .env (que no se sube a git)."""
import os
from pathlib import Path


def cargar_env(ruta: str | Path | None = None) -> None:
    ruta = Path(ruta) if ruta else Path(__file__).with_name(".env")
    if not ruta.exists():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            k, v = linea.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def obtener(nombre: str) -> str:
    cargar_env()
    valor = os.environ.get(nombre)
    if not valor:
        raise RuntimeError(f"Falta {nombre}: defínela como variable de entorno o en el archivo .env (ver .env.example)")
    return valor
