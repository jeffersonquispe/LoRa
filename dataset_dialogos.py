import csv
import re

from dialogos_tts import DIALOGOS

MIN_PAL, MAX_PAL = 3, 20
SALIDA = "dataset_dialogos.csv"


def palabras(t: str) -> int:
    return len(t.split())


def frases(linea: str) -> list[str]:
    """Parte una línea en frases (., ?, !) conservando la puntuación."""
    return [f.strip() for f in re.findall(r"[^.?!¿¡]*(?:¿[^?]*\?|¡[^!]*!|[^.?!]*[.?!])", linea) if f.strip()] or [linea]


def trocear(frase: str) -> list[str]:
    """Si una frase pasa del máximo, la parte por comas."""
    if palabras(frase) <= MAX_PAL:
        return [frase]
    partes, actual = [], ""
    for trozo in re.split(r"(?<=,)\s+", frase):
        if actual and palabras(actual + " " + trozo) > MAX_PAL:
            partes.append(actual)
            actual = trozo
        else:
            actual = f"{actual} {trozo}".strip()
    return partes + [actual]


def segmentos(linea: str) -> list[str]:
    segs = [s for f in frases(linea) for s in trocear(f)]
    # une los segmentos cortos con el vecino, mientras no pase del máximo
    i = 0
    while i < len(segs):
        if palabras(segs[i]) < MIN_PAL and len(segs) > 1:
            j = i + 1 if i + 1 < len(segs) else i - 1
            a, b = sorted((i, j))
            if palabras(segs[a] + " " + segs[b]) <= MAX_PAL:
                segs[a:b + 1] = [segs[a] + " " + segs[b]]
                i = max(a - 1, 0)
                continue
        i += 1
    return segs


def main() -> None:
    filas, descartadas = [], []
    for n, (personaje, linea) in enumerate(DIALOGOS, 1):
        for s in segmentos(linea):
            if MIN_PAL <= palabras(s) <= MAX_PAL:
                filas.append((f"{len(filas) + 1:03d}", personaje, n, s, palabras(s)))
            else:
                descartadas.append((personaje, s))

    with open(SALIDA, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "personaje", "linea_original", "texto", "palabras"])
        w.writerows(filas)

    print(f"{len(filas)} filas -> {SALIDA}")
    for fila in filas:
        print(f"{fila[0]} {fila[1]:<9} ({fila[4]:>2}) {fila[3]}")
    if descartadas:
        print("\nDescartadas (fuera de 3-20 palabras):")
        for p, s in descartadas:
            print(f"  {p}: {s}")


if __name__ == "__main__":
    main()
