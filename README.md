# LoRA para Chatterbox Multilingüe (español Latam)

Fine-tuning con LoRA del **T3** de [Chatterbox Multilingual es-mx-latam](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam),
más los scripts para armar el dataset (frases + audios), entrenar y comparar base vs LoRA.

> El experimento inicial usó 13.4 min de audio sintético de una sola voz (300 frases); el dataset actual tiene
> 2000 frases (108.7 min, ElevenLabs, voz masculina) y se está ampliando con una voz femenina venezolana.
> Detalles, resultados y limitaciones del experimento inicial en **[ENTRENAMIENTO_LORA.md](ENTRENAMIENTO_LORA.md)**.

## Contenido

| Ruta | Qué es |
|---|---|
| `chatterbox/` | Código oficial de [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) como **submódulo** fijado en el commit `5de7a54` |
| `dataset_frases.csv`, `dataset_audios/` | 2000 frases y sus audios (ElevenLabs, voz masculina). Las primeras 300 fueron el dataset del experimento inicial |
| `voces/vef_07508/` | Voz femenina venezolana: 140 clips (7.3 min, 24 kHz mono) del corpus `es_ve_female`, con `dataset_frases.csv` |
| `lora_data/` | Características pre-calculadas: `features.pt` (300), `features_1000.pt`, `features_2000.pt` y `features_vef_07508.pt` |
| `lora_out/`, `lora_out_1000/`, `lora_out_2000/` | Adaptadores entrenados con 300, 1000 y 2000 frases |
| `lora_inferencias/`, `lora_prueba/` | Audios de comparación base vs LoRA |
| `ejemplos/` | Scripts sueltos de prueba (`multilingue.py`, `comparar_voces.py`) |
| `*.py` | Scripts del flujo (ver abajo) |

## Instalación

```bash
# El código de Chatterbox viene como submódulo: hay que traerlo
git clone --recurse-submodules https://github.com/jeffersonquispe/LoRa.git
cd LoRa
# (si ya clonaste sin --recurse-submodules:  git submodule update --init)

pip install torch torchaudio transformers peft librosa huggingface_hub openai requests
pip install -e chatterbox

cp .env.example .env     # y completa ELEVENLABS_API_KEY / NVIDIA_API_KEY (solo para generar datos)
```

Los pesos base (~3 GB) **no** están en el repo: [carga_ligera.py](carga_ligera.py) los descarga de Hugging Face la
primera vez en `ckpt_latam/`.

## Flujo

```bash
python preparar_datos_lora.py   # dataset_frases.csv + dataset_audios/ -> lora_data/features.pt
python entrenar_lora.py         # entrena y guarda el mejor adaptador en lora_out/
python generar_con_lora.py      # genera las mismas frases con el modelo base y con el LoRA
```

### Varias voces (multi-speaker)

Cada hablante es un `features_*.pt`; `entrenar_lora.py` acepta varios y el prompt de condicionamiento de cada
ejemplo sale siempre del mismo hablante. Las voces pequeñas se repiten hasta `--max_rep` veces por época
(por defecto 4) para equilibrar, y la validación se imprime por voz.

```bash
# Voz femenina: de es_ve_female (prefijo vef_07508) a un dataset en el formato del proyecto
python preparar_voz_vef.py
python preparar_datos_lora.py --csv voces/vef_07508/dataset_frases.csv --audios voces/vef_07508/dataset_audios --ext wav --salida lora_data/features_vef_07508.pt

python entrenar_lora.py --datos lora_data/features_2000.pt lora_data/features_vef_07508.pt --salida lora_out_2voces
```

La voz al generar la define el `audio_prompt_path`: un clip de cada hablante.

Para regenerar el dataset desde cero (necesita las claves del `.env`):

```bash
python dataset_dialogos.py      # parte los diálogos en frases de 3 a 20 palabras
python ampliar_dataset.py       # amplía el dataset de frases (NVIDIA, Gemma 4)
python audios_dataset.py        # genera los audios (ElevenLabs)
```

## Usar el adaptador

```python
import carga_ligera
from peft import PeftModel

model = carga_ligera.cargar_modelo_completo()
model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, "lora_out").merge_and_unload()
wav = model.generate("Hola, ¿cómo estás?", language_id="es", audio_prompt_path="mi_voz.wav")
```

## Despliegue rápido con Docker

API (GPU) + Streamlit en otra máquina, sin clonar el repo completo. Los adaptadores y las voces de referencia
se bajan solos de S3 al arrancar; los pesos base (~3 GB) de Hugging Face la primera vez.

```bash
cp .env.example .env     # rellena AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY (usuario IAM de solo lectura al bucket)
docker compose up -d --build
```

UI en http://localhost:8501 y API en http://localhost:8000/docs. Hace falta GPU NVIDIA y el
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/). La primera petición
tarda (descarga + carga del modelo); los pesos base quedan en un volumen. Sin claves AWS no se descarga nada:
monta tus carpetas `lora_out*/` y `tts_app/voces/` como volúmenes. Subir pesos nuevos al bucket: `python pesos_s3.py subir lora_out_X`.

### Imagen lista (un solo comando)

El último stage del `Dockerfile` (`standalone`) trae la API con el adaptador `lora_out_2voces` y las dos voces dentro, sin
claves AWS. Probada en local con GPU: genera audio con `spk_f01` y `spk_m01`.

```bash
docker build -t ghcr.io/<usuario>/bessage-lora:latest .
docker run --gpus all -p 8000:8000 -v ckpt_latam:/app/ckpt_latam ghcr.io/<usuario>/bessage-lora:latest
```

Los pesos base (~3 GB) no van en la imagen: la primera petición los baja de Hugging Face (~5 min); el volumen
`ckpt_latam` los conserva. La imagen pesa ~11.7 GB, así que Docker necesita unos 25 GB libres para construirla.
Publicarla: `gh auth refresh -s write:packages`, `gh auth token | docker login ghcr.io -u <usuario> --password-stdin`
y `docker push`. Los paquetes de ghcr nacen **privados**: contiene un adaptador entrenado en parte con un corpus de
terceros (`es_ve_female`) y un clip de esa voz, así que no la hagas pública sin revisar su licencia.

## Requisitos

- GPU NVIDIA (probado en RTX 3060 Laptop, 6 GB; el entrenamiento usa ~3.7 GB).
- En equipos con poca RAM/memoria virtual en Windows, `carga_ligera.py` evita el error `os error 1455`.

## Licencias y aviso

- Código base y modelo Latam: MIT (Resemble AI). Chatterbox marca el audio generado con PerTh.
- Los audios de `dataset_audios/` se generaron con ElevenLabs: revisa sus términos antes de reutilizarlos.
- No se incluyen voces clonadas de hablantes reales: el modelo se entrenó solo con audio sintético.
