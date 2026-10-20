# LoRA para Chatterbox Multilingüe (español Latam)

Fine-tuning con LoRA del **T3** de [Chatterbox Multilingual es-mx-latam](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam),
más los scripts para armar el dataset (frases + audios), entrenar y comparar base vs LoRA.

> Experimento con 13.4 min de audio sintético de una sola voz. Detalles, resultados y limitaciones en
> **[ENTRENAMIENTO_LORA.md](ENTRENAMIENTO_LORA.md)**.

## Contenido

| Ruta | Qué es |
|---|---|
| `chatterbox/` | Código oficial de [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) como **submódulo** fijado en el commit `5de7a54` |
| `lora_out/` | Adaptador LoRA entrenado (época 3) con su model card |
| `dataset_frases.csv`, `dataset_audios/` | 300 frases y sus audios (ElevenLabs) |
| `lora_data/` | Características pre-calculadas (`features.pt`) |
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

Para regenerar el dataset desde cero (necesita las claves del `.env`):

```bash
python dataset_dialogos.py      # parte los diálogos en frases de 3 a 20 palabras
python ampliar_dataset.py       # amplía a 300 frases (NVIDIA, Gemma 4)
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

## Requisitos

- GPU NVIDIA (probado en RTX 3060 Laptop, 6 GB; el entrenamiento usa ~3.7 GB).
- En equipos con poca RAM/memoria virtual en Windows, `carga_ligera.py` evita el error `os error 1455`.

## Licencias y aviso

- Código base y modelo Latam: MIT (Resemble AI). Chatterbox marca el audio generado con PerTh.
- Los audios de `dataset_audios/` se generaron con ElevenLabs: revisa sus términos antes de reutilizarlos.
- No se incluyen voces clonadas de hablantes reales: el modelo se entrenó solo con audio sintético.
