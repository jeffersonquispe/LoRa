import csv
import re
import sys

from dataset_dialogos import palabras, MIN_PAL, MAX_PAL, SALIDA as CSV_BASE
from concurrent.futures import ThreadPoolExecutor

from historias_tts import client

MODEL = "google/gemma-4-31b-it"  # glm-5.3-flash gasta los tokens razonando y devuelve vacío

OBJETIVO = 300
SALIDA = "dataset_frases.csv"
POR_LLAMADA = 40

# Situaciones para dar variedad, en el tono de los diálogos originales
SITUACIONES = [
    "dos colegas atrapados en un archivo subterráneo durante una tormenta, bromeando con nerviosismo",
    "uno intenta convencer al otro de saltarse una regla y el otro se resiste",
    "provocaciones juguetonas y retos entre dos colegas con mucha tensión",
    "una confesión torpe y tierna después de un momento incómodo",
    "órdenes fingidamente autoritarias respondidas con sarcasmo",
    "frases cortas y entrecortadas en un momento de silencio y cercanía",
    "quejas de trabajo, plazos, jefes e informes, dichas con humor",
    "frases suaves y tranquilas al final de una noche difícil",
    "preguntas retóricas, burlas y réplicas rápidas",
    "frases serias y sinceras tras dejar atrás el juego",
]


def normalizar(t: str) -> str:
    return re.sub(r"\W+", " ", t.lower()).strip()


def pedir(situacion: str) -> list[str]:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Eres guionista. Escribes en español natural y coloquial. No uses markdown."},
            {
                "role": "user",
                "content": (
                    f"Escribe {POR_LLAMADA} frases sueltas que diría un personaje masculino adulto en esta situación: {situacion}.\n"
                    f"Reglas: cada frase en una línea, entre {MIN_PAL} y {MAX_PAL} palabras, sin nombre de hablante, "
                    "sin guiones de diálogo, sin comillas, sin numeración, sin explicaciones. "
                    "Tono juguetón, tenso y emocional, sin contenido sexual explícito. Frases todas distintas entre sí."
                ),
            },
        ],
        temperature=0.9,
        top_p=1,
        max_tokens=2048,
        stream=False,
        timeout=180,
    )
    return (r.choices[0].message.content or "").splitlines()


def limpiar(linea: str) -> str:
    linea = re.sub(r"^\s*(?:[-–—•*]+|\d+[.)])\s*", "", linea).strip().strip('"“”«»')
    return linea


def main() -> None:
    frases, vistas = [], set()

    def agregar(t: str) -> None:
        k = normalizar(t)
        if k and k not in vistas and MIN_PAL <= palabras(t) <= MAX_PAL and len(frases) < OBJETIVO:
            vistas.add(k)
            frases.append(t)

    # parte de las frases originales (sin personaje)
    with open(CSV_BASE, encoding="utf-8-sig", newline="") as f:
        for fila in csv.DictReader(f):
            agregar(fila["texto"])
    print(f"Base original: {len(frases)} frases")

    def guardar() -> None:
        with open(SALIDA, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "texto", "palabras"])
            for i, t in enumerate(frases, 1):
                w.writerow([f"{i:03d}", t, palabras(t)])

    def llamar(situacion: str) -> list[str]:
        try:
            return [limpiar(l) for l in pedir(situacion)]
        except Exception as e:
            print(f"  Falló llamada: {e}", file=sys.stderr)
            return []

    ronda = 0
    while len(frases) < OBJETIVO and ronda < 15:
        lote = [SITUACIONES[(ronda * 3 + k) % len(SITUACIONES)] for k in range(3)]
        ronda += 1
        with ThreadPoolExecutor(3) as ex:
            for lineas in ex.map(llamar, lote):
                for linea in lineas:
                    agregar(linea)
        guardar()
        print(f"  ronda {ronda}: {len(frases)}/{OBJETIVO}")

    guardar()
    print(f"{len(frases)} frases -> {SALIDA}")


if __name__ == "__main__":
    main()
