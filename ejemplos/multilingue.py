import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")

wav = model.generate("Hola, ¿cómo estás? Este es el modelo multilingüe de Chatterbox.", language_id="es")
ta.save("test-es.wav", wav, model.sr)
print("Listo: test-es.wav")
