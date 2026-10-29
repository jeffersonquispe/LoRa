#!/bin/sh
# Sube a S3 (desde tu PC, una vez) el código mínimo y los datos de las 4 voces. Requiere AWS_ACCESS_KEY_ID/SECRET/REGION
# con acceso al bucket (ver politica_iam.json) y la CLI de aws.
#   sh vast/subir_datos.sh
set -e
B="${PESOS_BUCKET:-chatterbox-lora-pesos-866923470465}"
cd "$(dirname "$0")/.."

tar czf /tmp/codigo.tgz carga_ligera.py entrenar_lora.py preparar_datos_lora.py pesos_s3.py vast/entrenar.sh
aws s3 cp /tmp/codigo.tgz "s3://$B/vast/codigo.tgz"

# voz masculina (ElevenLabs): dataset_audios/*.mp3 + dataset_frases.csv
aws s3 sync dataset_audios "s3://$B/datos/masculina/dataset_audios" --only-show-errors
aws s3 cp dataset_frases.csv "s3://$B/datos/masculina/dataset_frases.csv"

# voces preparadas en voces/<nombre>/{dataset_frases.csv,dataset_audios/}
for v in vef_07508 mex_nueva archive_19marzo; do
    aws s3 sync "voces/$v" "s3://$B/datos/$v" --only-show-errors
done
aws s3 ls "s3://$B/datos/" --recursive --summarize | tail -2
