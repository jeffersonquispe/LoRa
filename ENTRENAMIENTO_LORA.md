# Fine-tuning LoRA de Chatterbox Multilingüe (español Latam)

Registro del entrenamiento realizado el 2026-10-20.

## Resumen

Se entrenó un adaptador LoRA sobre el **T3** (el Llama de ~520M que convierte texto en tokens de voz) del modelo
**Chatterbox Multilingual es-mx-latam**, usando 300 frases en español con audio generado por ElevenLabs
(13.4 min, una sola voz). La pérdida de validación bajó de **4.143 a 3.346** (−19 %) en la época 3; a partir de ahí
hubo sobreajuste.

| | |
|---|---|
| Modelo base | `ResembleAI/Chatterbox-Multilingual-es-mx-latam` (`t3_es_mx_latam.safetensors`) |
| Qué se entrena | Solo LoRA en el backbone Llama de T3 (11.3 M de 514.7 M parámetros, 2.2 %) |
| Datos | 300 pares texto–audio, 13.4 min, 1 hablante |
| Mejor punto | Época 3, val 3.346 |
| Adaptador | `lora_out/adapter_model.safetensors` (45 MB) |
| Hardware | NVIDIA RTX 3060 Laptop (6 GB), pico de 3.68 GB de GPU |

## Componentes

| Componente | Origen | ¿Se entrena? |
|---|---|---|
| T3 (Llama 520M) | `t3_es_mx_latam.safetensors`, repo Latam | **LoRA** |
| S3Gen + S3Tokenizer | `s3gen_v3.pt`, repo Latam | No |
| VoiceEncoder | `ve.pt`, repo `ResembleAI/chatterbox` | No |
| Tokenizador de texto | `grapheme_mtl_merged_expanded_v1.json`, repo Latam | No |

Código base: [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) commit `5de7a54` (2026-07-21).

## Datos

1. **Diálogos de partida:** 20 líneas de diálogo → [dataset_dialogos.py](dataset_dialogos.py) las parte en frases de
   3 a 20 palabras (23 frases, [dataset_dialogos.csv](dataset_dialogos.csv)).
2. **Ampliación a 300 frases:** [ampliar_dataset.py](ampliar_dataset.py), con `google/gemma-4-31b-it` (NVIDIA), sin
   personaje, 3–17 palabras (promedio 8.3). Sin duplicados; se quitaron 3 frases en femenino.
   → [dataset_frases.csv](dataset_frases.csv).
3. **Audios:** [audios_dataset.py](audios_dataset.py) con ElevenLabs (`eleven_multilingual_v2`, voz
   `GcbypXUfJn5DbptRc2U7`) → `dataset_audios/001.mp3 … 300.mp3`.
4. **Características:** [preparar_datos_lora.py](preparar_datos_lora.py) → `lora_data/features.pt`. Por ejemplo guarda:
   tokens de texto (`[SOT]…[EOT]`), tokens de voz del clip completo (mediana 64, rango 23–163), tokens de prompt
   (≤150) y embedding de voz (256).

> `glm-5.3-flash` se descartó para generar frases: es un modelo con razonamiento y agotaba `max_tokens` pensando,
> devolviendo texto vacío.

## Configuración del entrenamiento

Script: [entrenar_lora.py](entrenar_lora.py). Comando: `python entrenar_lora.py` (valores por defecto).

| Parámetro | Valor |
|---|---|
| Rango / alpha / dropout | 16 / 32 / 0.0 |
| Módulos objetivo | `layers.N.self_attn.{q,k,v,o}_proj` y `layers.N.mlp.{gate,up,down}_proj` |
| Optimizador | AdamW, betas (0.9, 0.99), weight decay 0 |
| Learning rate | 2e-4, warmup 5 % + coseno |
| Épocas | 5 (179 actualizaciones) |
| Batch | 1, acumulación de gradiente 8 (efectivo 8) |
| Clip de gradiente | 1.0 |
| Precisión | pesos fp32, `autocast` bf16 |
| Split | 285 train / 15 val (5 %), semilla 0 |
| Text dropout | 10 % (anula el texto, como la rama incondicional del CFG) |

**Pérdida:** entropía cruzada del siguiente token de voz. Entrada `[cond | texto | BOS, s1…sn]`, objetivo
`[s1…sn, EOS]`. No se usó la `T3.loss` del repo porque no desplaza los tokens ni usa bien las dimensiones.

**Condicionamiento:** en cada paso, la voz y el prompt provienen de **otro** clip del dataset, para que el modelo no
copie el audio que debe predecir. `emotion_adv = 0.5`.

**Batch 1:** T3 no usa máscara de atención, por lo que el padding alteraría el cálculo respecto a la inferencia.

## Resultados

| Época | Train | Val |
|---|---|---|
| inicial | — | 4.1431 |
| 1 | 3.8901 | 3.4586 |
| 2 | 3.4712 | 3.3517 |
| **3** | 3.2701 | **3.3463** ← guardada |
| 4 | 3.1098 | 3.3656 |
| 5 | 3.0266 | 3.3758 |

Después de la época 3 el train sigue bajando y la validación sube: **sobreajuste** (13 min de audio, un hablante).
El script solo guarda el adaptador cuando mejora la validación.

### Inferencia (cualitativo, sin evaluar a oído todavía)

[generar_con_lora.py](generar_con_lora.py) genera cada frase con el modelo base y con el LoRA fusionado
(`merge_and_unload`), usando `dataset_audios/001.mp3` como voz de referencia. En 8 frases nuevas
(`lora_inferencias/`) el LoRA produjo audios entre un 3 % y un 25 % más cortos que el base, es decir, habla más rápido.
**No se ha hecho evaluación subjetiva ni MOS.**

## Limitaciones

- Los audios de entrenamiento son **sintéticos** (ElevenLabs): el LoRA aprende esa voz y su estilo, no un acento real.
- 13.4 min de un solo hablante es poco; sirve para validar el flujo, no para un modelo final.
- Para un acento (MX/VE) real hacen falta grabaciones reales con transcripción.
- La pérdida de validación se calculó sobre solo 15 ejemplos: es ruidosa.

## Reproducir

```bash
pip install torch torchaudio transformers peft librosa huggingface_hub openai requests
pip install -e chatterbox            # clon de resemble-ai/chatterbox
cp .env.example .env                 # y completar las claves

python preparar_datos_lora.py        # → lora_data/features.pt
python entrenar_lora.py              # → lora_out/
python generar_con_lora.py           # → lora_prueba/ (base vs LoRA)
```

En equipos con poca RAM/memoria virtual, [carga_ligera.py](carga_ligera.py) carga los safetensors sin `mmap` y
directo a la GPU (evita el error de Windows `os error 1455`).

## Cargar el adaptador

```python
import carga_ligera
from peft import PeftModel

model = carga_ligera.cargar_modelo_completo()
model.t3.tfmr = PeftModel.from_pretrained(model.t3.tfmr, "lora_out").merge_and_unload()
wav = model.generate("Hola, ¿cómo estás?", language_id="es", audio_prompt_path="mi_voz.wav")
```
