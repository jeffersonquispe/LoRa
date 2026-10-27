#!/bin/sh
# Trae de S3 los adaptadores y las voces que falten, y luego arranca la API.
# Sin credenciales AWS no descarga nada: monta tus propios lora_out*/ y tts_app/voces/ como volúmenes.
set -e
cd /app

if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$PESOS_BUCKET" ]; then
    for m in ${MODELOS_LORA:-lora_out_2voces lora_out_2000}; do
        if [ ! -f "$m/adapter_config.json" ]; then
            echo "[entrypoint] bajando $m desde s3://$PESOS_BUCKET"
            python pesos_s3.py bajar "$m"
        fi
    done
    if [ -z "$(ls -A tts_app/voces 2>/dev/null)" ]; then
        echo "[entrypoint] bajando voces de referencia"
        python pesos_s3.py bajar tts_app/voces
    fi
else
    echo "[entrypoint] sin AWS_ACCESS_KEY_ID/PESOS_BUCKET: no se descargan pesos de S3"
fi

# La voz "por_defecto" de la API es dataset_audios/001.mp3 (no viaja en la imagen): se usa la masculina de referencia
if [ ! -f dataset_audios/001.mp3 ] && [ -f tts_app/voces/masculina_elevenlabs.mp3 ]; then
    mkdir -p dataset_audios
    cp tts_app/voces/masculina_elevenlabs.mp3 dataset_audios/001.mp3
fi

exec "$@"
