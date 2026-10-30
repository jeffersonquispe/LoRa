#!/bin/bash
# Se ejecuta DENTRO de la instancia de Vast (plantilla PyTorch 2.6 / CUDA 12.4).
# Variables obligatorias: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
# Opcionales: PESOS_BUCKET, RUN (nombre de la carpeta de salida), EPOCAS, MAX_PASOS (prueba corta), MAX_REP, LR,
#             N_POR_VOZ (clips por voz) o N_<VOZ> para una voz concreta: N_MASCULINA, N_VEF_07508, N_MEX_NUEVA, N_ARCHIVE_19MARZO,
#             HF_TOKEN, DESTRUIR=1 (borra la instancia al terminar; requiere VAST_API_KEY y CONTAINER_ID, que Vast define)
#
#   aws s3 cp s3://$PESOS_BUCKET/vast/codigo.tgz . && tar xzf codigo.tgz && bash vast/entrenar.sh
set -euo pipefail
B="${PESOS_BUCKET:-chatterbox-lora-pesos-866923470465}"
RUN="${RUN:-lora_out_4voces}"
EPOCAS="${EPOCAS:-3}"
N="${N_POR_VOZ:-500}"
MAX_REP="${MAX_REP:-4}"
LR="${LR:-2e-4}"
MAX_PASOS="${MAX_PASOS:-0}"
VOCES="masculina vef_07508 mex_nueva archive_19marzo"
CHATTERBOX_COMMIT=5de7a54

cd "$(dirname "$0")/.."
apt-get update -qq && apt-get install -y -qq git ffmpeg libsndfile1 >/dev/null
pip install -q awscli boto3 peft==0.21.2 soundfile==0.14.0 huggingface_hub
[ -d chatterbox ] || { git clone -q https://github.com/resemble-ai/chatterbox.git chatterbox && git -C chatterbox checkout -q $CHATTERBOX_COMMIT; }
pip install -q -e chatterbox

echo "[1/4] datos y modelo base desde S3 (Hugging Face puede ir muy lento en Vast)"
[ -f ckpt_latam/t3_es_mx_latam.safetensors ] || aws s3 sync "s3://$B/modelo_base/ckpt_latam" ckpt_latam --only-show-errors
for v in $VOCES; do aws s3 sync "s3://$B/datos/$v" "datos/$v" --only-show-errors; done

echo "[2/4] características"
mkdir -p lora_data
LISTA=()
for v in $VOCES; do
    # primeras N filas de cada CSV (cabecera + N)
    var="N_$(echo "$v" | tr a-z A-Z)"; nv="${!var:-$N}"
    head -n $((nv + 1)) "datos/$v/dataset_frases.csv" > "datos/$v/frases_n.csv"
    ext=wav; [ "$v" = masculina ] && ext=mp3
    python preparar_datos_lora.py --csv "datos/$v/frases_n.csv" --audios "datos/$v/dataset_audios" \
        --ext $ext --salida "lora_data/features_$v.pt"
    LISTA+=("lora_data/features_$v.pt")
done

echo "[3/4] entrenamiento -> $RUN (checkpoints a S3 cada 2 min)"
mkdir -p "$RUN"
( while true; do sleep 120; aws s3 sync "$RUN" "s3://$B/checkpoints/$RUN" --only-show-errors || true; done ) &
SYNC=$!
python entrenar_lora.py --datos "${LISTA[@]}" --salida "$RUN" --epocas "$EPOCAS" --max_pasos "$MAX_PASOS" --max_rep "$MAX_REP" --lr "$LR" --guardar_epocas 2>&1 | tee "$RUN/entrenar.log"

echo "[4/4] subida final"
kill $SYNC 2>/dev/null || true
aws s3 sync "$RUN" "s3://$B/checkpoints/$RUN" --only-show-errors
aws s3 ls "s3://$B/checkpoints/$RUN/" --human-readable

if [ "${DESTRUIR:-0}" = 1 ] && [ -n "${CONTAINER_ID:-}" ] && [ -n "${VAST_API_KEY:-}" ]; then
    pip install -q vastai && echo y | vastai destroy instance "$CONTAINER_ID" --api-key "$VAST_API_KEY"
fi
