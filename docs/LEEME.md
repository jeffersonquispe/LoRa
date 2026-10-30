# Página de demos

`index.html` es una página estática (sin dependencias ni compilación) con dos partes:

1. **Muestras**: reproduce los audios de `audio/` para cada voz.
2. **Probar la API**: consola que llama a la API desde el navegador (estado, lista de voces, generar texto y una prueba rápida con todas las voces).

Se puede abrir directamente (`docs/index.html`) o publicar con GitHub Pages.

## Publicar con GitHub Pages

En el repositorio: **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main`, carpeta `/docs`**. La página queda en `https://<usuario>.github.io/<repositorio>/`.

Ojo: en un repositorio privado, Pages exige un plan de pago, y si la publicas, **los audios y la página quedan públicos**. Las voces vienen de grabaciones de terceros (ver el aviso al pie de la página): confirma que tienes permiso antes de publicarlas.

## Cambiar las muestras

Los audios están en `audio/<voz>_1.wav` y `audio/<voz>_2.wav`, con las frases definidas en `FRASES` dentro de `index.html`. Para regenerarlos contra la API:

```bash
curl -X POST $URL/v1/text-to-speech/spk_f01 -H "Salad-Api-Key: $SALAD_KEY" -H "Content-Type: application/json" \
  -d '{"text": "Hola, buenas tardes. Le llamo para confirmar su cita de mañana a las diez."}' -o audio/venezolana_1.wav
```

Las voces y sus IDs también se declaran en `VOCES` (en `index.html`).

## Probar la API desde el navegador: requisitos

El navegador solo deja que una página llame a otro dominio si ese servidor lo permite (**CORS**). Hay dos condiciones:

1. **La API debe permitir el origen de la página.** Define en el contenedor la variable `CORS_ORIGINS` con la dirección donde esté alojada la página, sin barra final. Por ejemplo `https://<usuario>.github.io`. Sin esa variable, la API no acepta navegadores. Esta opción está en `tts_app/api.py` y exige la imagen `v6` o posterior.
2. **El gateway de Salad no puede exigir `Salad-Api-Key`.** El navegador envía antes una consulta `OPTIONS` (preflight) sin cabeceras personalizadas, y el gateway la rechazaría con 401 y bloquearía todo. Para probar desde el navegador hay dos caminos:
   - Desactiva la autenticación del gateway en el Container Group y protege la API con `TTS_API_KEY` (la página la envía en `X-API-Key`).
   - O deja el gateway autenticado y prueba con `curl`/Python (ver `tts_app/API.md`), no desde el navegador.

## Sobre las claves

- Las claves que escribes en la página **no se guardan en ningún servidor**: se envían solo a la dirección que pongas.
- Si marcas *Recordar*, quedan en el almacenamiento local de tu navegador, en texto. No lo marques en un equipo compartido.
- No pongas claves en `index.html` ni en el repositorio.
- Con una clave en la página y la API abierta a todo internet, cualquiera con acceso a esa clave puede gastar tu GPU. Usa `CORS_ORIGINS` con el origen exacto, no `*`.
