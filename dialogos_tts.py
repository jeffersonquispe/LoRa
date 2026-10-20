import os
import sys

from elevenlabs_tts import generar_audio, VOICE_ID

# Voz de cada personaje. Cambia el de Emiliano por otro voice ID para distinguirlos.
VOCES = {
    "Rafael": VOICE_ID,
    "Emiliano": VOICE_ID,
}

OUT = "dialogos_elevenlabs"

# (personaje, línea) en el orden de la historia "Muñeca desviada hacia la pantalla"
DIALOGOS = [
    ("Emiliano", "Rafael, ¿piensas seguir limpiando esas gafas hasta que se disuelvan o vas a admitir que estamos jodidos?"),
    ("Rafael", "Doctor Miller va a revisar ese informe en menos de una hora. Y la puerta se trabó cuando saltó la luz. Así que sí, Emiliano, estamos jodidos si no ordenamos esto ahora."),
    ("Emiliano", "O podríamos ajustar una fecha. Nadie se enteraría."),
    ("Rafael", "Prefiero perder el cargo a falsificar un registro federal."),
    ("Emiliano", "Mira esto. Ese dato está mal desde el principio."),
    ("Emiliano", "Miller nunca notaría un pequeño ajuste."),
    ("Rafael", "No voy a romper las reglas solo porque tú lo pides con esa cara."),
    ("Emiliano", "Entonces dime que me aparte, profesor."),
    ("Emiliano", "Esto es una pérdida de tiempo."),
    ("Emiliano", "Estamos atrapados aquí, Rafael. El informe puede esperar."),
    ("Rafael", "Demuestra que realmente no te importa tu carrera."),
    ("Rafael", "Ahora me toca a mí decidir cuánto nos acercamos."),
    ("Rafael", "Organiza esos archivos por año, Emiliano. Demuestra que puedes seguir una instrucción por una vez."),
    ("Emiliano", "¿Así, jefe?"),
    ("Rafael", "Estás disfrutando esto demasiado."),
    ("Emiliano", "Solo estoy comprobando cuánto aguantas antes de romper las reglas tú mismo."),
    ("Rafael", "Deja de jugar."),
    ("Emiliano", "Esto ya no es un juego, ¿verdad?"),
    ("Rafael", "Quédate."),
    ("Emiliano", "Ya lo hice."),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for i, (personaje, texto) in enumerate(DIALOGOS, 1):
        ruta = os.path.join(OUT, f"{i:02d}_{personaje.lower()}.mp3")
        if os.path.exists(ruta):
            continue
        print(f"[{i:02d}] {personaje}: {texto}")
        try:
            generar_audio(texto, ruta, voice_id=VOCES[personaje])
        except Exception as e:
            print(f"  Falló: {e}", file=sys.stderr)
    print("Listo.")


if __name__ == "__main__":
    main()
