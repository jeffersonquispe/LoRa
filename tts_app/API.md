# API de texto a voz: guía para quien la consume

Servicio HTTP que convierte texto en audio (WAV) con Chatterbox Multilingual (base Latam) y, opcionalmente, un
adaptador LoRA. La voz se elige con un audio de referencia, así que una misma API sirve para varios hablantes.

- **URL base:** `http://localhost:8000` (en Docker Compose, otros contenedores la ven como `http://api:8000`)
- **Documentación interactiva (OpenAPI):** `GET /docs` · esquema JSON: `GET /openapi.json`
- **Formato de entrada:** `multipart/form-data` (o `application/x-www-form-urlencoded` si no subes archivos)
- **Formato de salida:** `audio/wav` en `/tts`; JSON en el resto
- **Concurrencia:** una sola generación a la vez (la GPU solo aguanta un modelo). Las peticiones simultáneas esperan en cola.

## Autenticación

Si el servidor se arrancó con la variable `TTS_API_KEY`, todas las rutas **menos `/health`** exigen la cabecera:

```
X-API-Key: <clave>
```

Sin clave configurada en el servidor, no se pide nada. Una clave ausente o incorrecta devuelve `401`.

## Flujo típico

1. `GET /modelos` y `GET /voces` para saber qué nombres puedes usar.
2. (Opcional) `POST /voces` para subir un audio de referencia nuevo.
3. `POST /tts` con el texto, el `modelo` y la `voz`.

## Rutas

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Liveness: el proceso responde. Sin autenticación. |
| GET | `/listo` | Readiness: 200 solo cuando el modelo está cargado. Sin autenticación. |
| GET | `/modelos` | Modelos disponibles y cuál está cargado. |
| GET | `/voces` | Voces de referencia disponibles. |
| POST | `/voces` | Sube una voz de referencia. |
| GET | `/idiomas` | Idiomas soportados. |
| POST | `/tts` | Genera el audio. |

### `GET /health`

```json
{ "ok": true, "modelo_cargado": "lora_out_2voces", "cuda": true, "estado": "listo" }
```

`estado` es `bajo_demanda`, `cargando`, `listo` o `error: ...`. Responde 200 en cuanto el proceso arranca, **aunque el
modelo aún se esté cargando**: úsalo como *liveness*. Para saber si ya se puede generar, usa `/listo`.

### `GET /listo`

200 `{ "listo": true, "modelo_cargado": "lora_out_2voces" }` cuando el modelo precargado está en la GPU y calentado;
503 `{ "detail": "No listo: cargando" }` mientras tanto. Sin autenticación. Con la variable `PRECARGAR_MODELO` (la imagen
Docker `standalone` la trae con `lora_out_2voces`) el modelo se carga al arrancar y se hace una generación corta de
calentamiento, de modo que la primera petición real tarda unos segundos en lugar de minutos. Sin esa variable responde
200 siempre y el modelo se carga con la primera petición.

En Salad, configura la **readiness probe** como HTTP `GET /listo` en el puerto 8000: así no recibe tráfico hasta que el
modelo está listo. Ten en cuenta que la imagen baja ~3 GB de pesos base en el primer arranque de cada nodo (unos 3
minutos): ajusta el tiempo máximo de arranque de la sonda en consecuencia.

### `GET /modelos`

```json
{ "modelos": ["base", "lora_out", "lora_out_1000", "lora_out_2000", "lora_out_2voces"], "cargado": "lora_out_2voces" }
```

- `base`: Chatterbox Latam sin entrenar.
- `lora_out_2voces`: adaptador entrenado con una voz masculina (ElevenLabs) y una femenina venezolana. Es el recomendado.
- `lora_out_2000`, `lora_out_1000`, `lora_out`: adaptadores anteriores, solo voz masculina, con 2000, 1000 y 300 frases.

La lista depende de las carpetas `lora_out*` presentes en el servidor: consúltala en lugar de fijarla en tu código.

### `GET /voces`

```json
{
  "voces": ["por_defecto", "femenina_venezolana.wav", "masculina_elevenlabs.mp3"],
  "voces_detalle": [
    {"id": "por_defecto", "archivo": null,                       "nombre": "Por defecto (masculina)", "genero": "m"},
    {"id": "spk_f01",     "archivo": "femenina_venezolana.wav",  "nombre": "Femenina venezolana",     "genero": "f"},
    {"id": "spk_m01",     "archivo": "masculina_elevenlabs.mp3", "nombre": "Masculina ElevenLabs",    "genero": "m"}
  ]
}
```

- `voces_detalle` es lo que debes usar: cada voz tiene un **`id`** estable para pasar en `voz` de `/tts`.
  Si un archivo no está registrado, su `id` es `null` y se usa por el nombre del archivo.
- `voces` (solo nombres de archivo) se mantiene por compatibilidad.
- `por_defecto` es la voz masculina de referencia.

### `POST /voces`

Guarda un audio de referencia para usarlo luego por nombre.

| Campo | Tipo | Obligatorio | Descripción |
|---|---|---|---|
| `archivo` | file | **sí** | `.wav`, `.mp3`, `.flac` u `.ogg`. Idealmente 5 a 15 s, una sola voz, sin ruido ni música. |
| `id` | string | no | ID estable para usar en `/tts`. 1 a 40 caracteres entre letras, números, `_` y `-`. `por_defecto` está reservado. |
| `nombre` | string | no | Nombre legible (solo informativo). |
| `genero` | string | no | `m` o `f` (solo informativo). |

Respuesta: `{ "voz": "nombre.wav", "id": "spk_f02" }` (`id` es `null` si no lo enviaste). Si ya existía un archivo con
ese nombre, **lo sobrescribe**; si el `id` ya existía, **se reasigna** al archivo nuevo. Un formato, `id` o `genero`
no válidos devuelven `400`. Los IDs se guardan en `tts_app/voces/voces.json`.

### `GET /idiomas`

Devuelve un objeto `código → nombre`: `ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv,
sw, tr, zh`. Los adaptadores LoRA se entrenaron solo con español: en otros idiomas, usa `base`.

### `POST /tts`

| Campo | Tipo | Obligatorio | Defecto | Rango / valores | Descripción |
|---|---|---|---|---|---|
| `texto` | string | **sí** | – | 1 a 600 caracteres | Texto a locutar. Se recortan los espacios de los extremos. |
| `modelo` | string | no | `base` | un valor de `/modelos` | Modelo o adaptador LoRA. |
| `voz` | string | no | `por_defecto` | un `id` de `/voces` (`spk_f01`, `spk_m01`…) o un nombre de archivo | **Speaker.** Se ignora si envías `voz_archivo`. |
| `voz_archivo` | file | no | – | wav, mp3, flac, ogg | Audio de referencia solo para esta petición (no se guarda). |
| `idioma` | string | no | `es` | un valor de `/idiomas` | Código de idioma. |
| `exaggeration` | float | no | `0.5` | 0.0 a 2.0 | Expresividad. Más alto, más emoción y variación de tono. |
| `cfg_weight` | float | no | `0.5` | 0.0 a 1.0 | Guía hacia el texto y ritmo. Más bajo, ritmo más pausado. |
| `temperature` | float | no | `0.8` | 0.05 a 2.0 | Variación al muestrear. Más bajo, más estable; más alto, más variado. |

**Respuesta `200`:** el cuerpo es el audio `audio/wav` (mono, 24 kHz). Cabeceras útiles:

| Cabecera | Significado |
|---|---|
| `X-Modelo` | Modelo con el que se generó. |
| `X-Segundos-Generacion` | Tiempo de generación en el servidor. |
| `X-Duracion-Audio` | Duración del audio resultante, en segundos. |

**Errores** (cuerpo JSON `{ "detail": "..." }`):

| Código | Cuándo |
|---|---|
| `400` | Texto vacío, de más de 600 caracteres, o idioma no soportado. |
| `401` | `X-API-Key` ausente o incorrecta (solo si el servidor tiene clave). |
| `404` | El `modelo` o la `voz` no existen (el mensaje lista los IDs y archivos disponibles). |
| `422` | Un parámetro numérico fuera de rango o con el tipo equivocado. |
| `503` | Sin memoria de GPU: prueba con un texto más corto. |

## Ejemplos

### curl

```bash
curl -X POST http://localhost:8000/tts \
  -F "texto=Hola, ¿cómo estás?" \
  -F "modelo=lora_out_2voces" \
  -F "voz=spk_f01" \
  -F "idioma=es" \
  -o salida.wav
```

Registrar una voz nueva con su ID y usarla:

```bash
curl -X POST http://localhost:8000/voces -F "archivo=@mexicana.wav" -F "id=spk_f02" -F "nombre=Femenina mexicana" -F "genero=f"
curl -X POST http://localhost:8000/tts -F "texto=Hola" -F "modelo=lora_out_2voces" -F "voz=spk_f02" -o salida.wav
```

Con una voz de referencia puntual, sin subirla antes:

```bash
curl -X POST http://localhost:8000/tts -F "texto=Buenos días" -F "modelo=lora_out_2voces" \
  -F "voz_archivo=@mi_voz.wav" -o salida.wav
```

### Python

```python
import requests

API = "http://localhost:8000"
HEADERS = {}  # {"X-API-Key": "..."} si el servidor tiene clave

r = requests.post(
    f"{API}/tts",
    headers=HEADERS,
    data={"texto": "Hola, ¿cómo estás?", "modelo": "lora_out_2voces",
          "voz": "spk_m01", "idioma": "es", "exaggeration": 0.5},
    timeout=600,  # la primera petición carga el modelo en la GPU
)
if r.ok:
    open("salida.wav", "wb").write(r.content)
    print(r.headers["X-Duracion-Audio"], "s de audio")
else:
    print(r.status_code, r.json()["detail"])
```

### JavaScript (navegador o Node 18+)

```js
const form = new FormData();
form.append("texto", "Hola, ¿cómo estás?");
form.append("modelo", "lora_out_2voces");
form.append("voz", "spk_f01");

const r = await fetch("http://localhost:8000/tts", { method: "POST", body: form /*, headers: {"X-API-Key": "..."}*/ });
if (!r.ok) throw new Error((await r.json()).detail);
new Audio(URL.createObjectURL(await r.blob())).play();
```

Si llamas desde un navegador a otro origen, la API **no** habilita CORS: haz la llamada desde tu backend o ponle un
proxy delante.

## Consejos de uso

- **Timeouts:** la primera petición carga el modelo (~1 min) y cambiar de `modelo` lo recarga. Usa un timeout de al
  menos 600 s en el cliente; una generación normal con el modelo cargado tarda unos segundos.
- **Cambiar de modelo a menudo es caro.** Si tu aplicación usa dos voces, usa el **mismo `modelo`** (`lora_out_2voces`)
  y cambia solo `voz`: no recarga nada.
- **Textos largos:** el límite es 600 caracteres por petición. Divide el texto por frases y concatena los WAV.
- **Usa IDs, no nombres de archivo.** El ID (`spk_f01`) es estable: puedes cambiar el audio que hay detrás (con
  `POST /voces` y el mismo `id`) sin tocar las aplicaciones que lo consumen.
- **Calidad de la voz de referencia:** 5 a 15 s, una sola persona, sin ruido de fondo. Es lo que más condiciona el timbre.
- **No hay streaming:** el audio llega completo al terminar la generación.
- **Seguridad:** si expones la API fuera de tu máquina, define `TTS_API_KEY` y ponla detrás de HTTPS. Las voces
  subidas son accesibles para cualquiera que tenga la clave.
