"""Interfaz Streamlit para la API de TTS (api.py).

    streamlit run tts_app/app.py
La API debe estar corriendo (por defecto en http://localhost:8000).
"""
import os

import requests
import streamlit as st

st.set_page_config(page_title="Chatterbox TTS", page_icon="🎙️", layout="centered")

API_URL = os.environ.get("TTS_API_URL", "http://localhost:8000")
API_KEY = os.environ.get("TTS_API_KEY", "")


def llamar(metodo, ruta, **kw):
    cab = {"X-API-Key": st.session_state.get("clave", API_KEY)} if st.session_state.get("clave", API_KEY) else {}
    return requests.request(metodo, st.session_state.get("url", API_URL).rstrip("/") + ruta,
                            headers=cab, timeout=kw.pop("timeout", 30), **kw)


@st.cache_data(ttl=10, show_spinner=False)
def catalogo(url, clave):
    cab = {"X-API-Key": clave} if clave else {}
    g = lambda r: requests.get(url.rstrip("/") + r, headers=cab, timeout=10)
    modelos, voces, idiomas = g("/modelos"), g("/voces"), g("/idiomas")
    for r in (modelos, voces, idiomas):
        r.raise_for_status()
    vj = voces.json()
    # voces_detalle trae IDs y nombres; con una API antigua se cae a la lista de archivos
    detalle = vj.get("voces_detalle") or [{"id": v, "archivo": v, "nombre": v} for v in vj["voces"]]
    return modelos.json(), detalle, idiomas.json()


# ---------- barra lateral ----------
with st.sidebar:
    st.header("Conexión")
    st.session_state["url"] = st.text_input("URL de la API", API_URL)
    st.session_state["clave"] = st.text_input("API key (si la hay)", API_KEY, type="password")

try:
    info_modelos, voces, idiomas = catalogo(st.session_state["url"], st.session_state["clave"])
except Exception as e:
    st.title("🎙️ Chatterbox TTS")
    st.error(f"No se pudo conectar con la API en {st.session_state['url']}.\n\n"
             "Arráncala con:  `python -m uvicorn api:app --app-dir tts_app --port 8000`\n\n"
             f"Detalle: {e}")
    st.stop()

with st.sidebar:
    st.header("Modelo")
    modelo = st.selectbox("Modelo / adaptador", info_modelos["modelos"],
                          help="'base' es Chatterbox Latam sin entrenar; lora_out_* son adaptadores LoRA. "
                               "Cambiar de modelo recarga la GPU (~1 min la primera vez).")
    if info_modelos.get("cargado"):
        st.caption(f"En GPU ahora: {info_modelos['cargado']}")
    idioma = st.selectbox("Idioma", list(idiomas), index=list(idiomas).index("es"),
                          format_func=lambda c: f"{c} · {idiomas[c]}")

    st.header("Voz")
    origen = st.radio("Referencia", ["De la lista", "Subir audio"], horizontal=True)
    voz, voz_archivo = "por_defecto", None
    if origen == "De la lista":
        # se envía el ID si la voz lo tiene, y si no el nombre del archivo
        claves = {v["id"] or v["archivo"]: v for v in voces}
        voz = st.selectbox("Voz", list(claves), format_func=lambda k: (
            f"{claves[k]['nombre']} · {k}" if claves[k]["id"] else claves[k]["nombre"]))
    else:
        voz_archivo = st.file_uploader("Audio de referencia (5-15 s, una voz, sin ruido)",
                                       type=["wav", "mp3", "flac", "ogg"])
        if voz_archivo is not None and st.button("Guardar en la lista"):
            r = llamar("POST", "/voces", files={"archivo": (voz_archivo.name, voz_archivo.getvalue())})
            st.success(f"Guardada como {r.json().get('voz')}") if r.ok else st.error(r.text)
            catalogo.clear()

    st.header("Ajustes")
    exaggeration = st.slider("Expresividad (exaggeration)", 0.0, 2.0, 0.5, 0.05,
                             help="Más alto = más emoción y variación de tono.")
    cfg_weight = st.slider("Fidelidad al texto / ritmo (cfg_weight)", 0.0, 1.0, 0.5, 0.05,
                           help="Más bajo = ritmo más pausado y natural.")
    temperature = st.slider("Variación (temperature)", 0.05, 2.0, 0.8, 0.05)

# ---------- principal ----------
st.title("🎙️ Chatterbox TTS")
texto = st.text_area("Texto", height=140, max_chars=600,
                     placeholder="Escribe lo que quieres que diga la voz…")

if "historial" not in st.session_state:
    st.session_state["historial"] = []

if st.button("Generar audio", type="primary", disabled=not texto.strip()):
    datos = {"texto": texto, "modelo": modelo, "voz": voz, "idioma": idioma, "exaggeration": exaggeration,
             "cfg_weight": cfg_weight, "temperature": temperature}
    archivos = {"voz_archivo": (voz_archivo.name, voz_archivo.getvalue())} if voz_archivo is not None else None
    with st.spinner("Generando… (si cambiaste de modelo, la primera vez tarda más)"):
        try:
            r = llamar("POST", "/tts", data=datos, files=archivos, timeout=600)
        except Exception as e:
            st.error(f"Error de conexión: {e}")
        else:
            if r.ok:
                st.session_state["historial"].insert(0, {
                    "texto": texto, "modelo": modelo, "audio": r.content,
                    "seg": r.headers.get("X-Segundos-Generacion"), "dur": r.headers.get("X-Duracion-Audio")})
                catalogo.clear()
            else:
                try:
                    st.error(r.json().get("detail", r.text))
                except Exception:
                    st.error(r.text)

for i, h in enumerate(st.session_state["historial"][:10]):
    with st.container(border=True):
        st.caption(f"**{h['modelo']}** · {h['dur']} s de audio · generado en {h['seg']} s")
        st.write(h["texto"])
        st.audio(h["audio"], format="audio/wav")
        st.download_button("Descargar .wav", h["audio"], file_name=f"tts_{h['modelo']}_{i + 1}.wav",
                           mime="audio/wav", key=f"dl{i}")
