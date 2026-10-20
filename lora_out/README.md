---
license: mit
language:
- es
library_name: peft
base_model: ResembleAI/Chatterbox-Multilingual-es-mx-latam
tags:
- chatterbox
- text-to-speech
- tts
- lora
- peft
- spanish
pipeline_tag: text-to-speech
---

# Chatterbox es-mx-latam: adaptador LoRA (experimental)

Adaptador LoRA para el **T3** (backbone Llama de ~520M) de
[`ResembleAI/Chatterbox-Multilingual-es-mx-latam`](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-es-mx-latam).
Es un experimento para validar el flujo de fine-tuning, no un modelo de producción.

## Detalles

| | |
|---|---|
| Parámetros entrenados | 11.3 M (2.2 % del modelo) |
| Rango / alpha | 16 / 32 |
| Módulos | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` del backbone Llama |
| Datos | 300 frases en español (3–17 palabras), 13.4 min, una sola voz |
| Épocas | 3 (mejor validación de 5 entrenadas) |
| Pérdida de validación | 4.143 → 3.346 |

Los demás componentes (S3Gen, VoiceEncoder, tokenizadores) no se modificaron.
Detalles completos del entrenamiento: `ENTRENAMIENTO_LORA.md` en el repositorio de código.

## Uso

Requiere el código de [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) y los pesos base.

```python
from peft import PeftModel
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_local(ckpt_dir, "cuda", t3_model="t3_es_mx_latam.safetensors")
model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, "TU_USUARIO/chatterbox-es-latam-lora").merge_and_unload()
wav = model.generate("Hola, ¿cómo estás?", language_id="es", audio_prompt_path="referencia.wav")
```

## Limitaciones

- Entrenado con audio **sintético** (generado con ElevenLabs): aprende esa voz y su estilo, no un acento real.
- Poco audio (13.4 min) y un solo hablante; hay sobreajuste a partir de la época 3.
- La pérdida de validación usa solo 15 ejemplos, por lo que es ruidosa.
- No se ha hecho evaluación subjetiva (MOS ni comparación a ciegas). En pruebas informales el LoRA genera audios
  más cortos (habla más rápido) que el modelo base.
- Chatterbox incluye marca de agua PerTh en todo el audio generado.

## Licencia y datos

Adaptador bajo MIT, igual que el modelo base. Los audios de entrenamiento se generaron con ElevenLabs; revisa sus
términos antes de redistribuir el dataset o usar el adaptador comercialmente.
