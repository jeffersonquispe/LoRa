"""Expone la API (puerto 8000) y el Streamlit (8501) a internet con ngrok.

Lee de .env:  NGROK_AUTHTOKEN, NGROK_UI_USER, NGROK_UI_PASS  (y TTS_API_KEY para proteger la API).
El Streamlit queda detrás de usuario/contraseña de ngrok (no tiene login propio); la API se protege con TTS_API_KEY,
que debe estar definida al arrancar la API.

    python iniciar_ngrok.py
Ctrl+C cierra los túneles. Las URLs públicas se imprimen al arrancar (cambian en cada ejecución con el plan gratis).
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request

RAIZ = os.path.dirname(os.path.abspath(__file__))


def leer_env() -> dict:
    env = {}
    try:
        with open(os.path.join(RAIZ, ".env"), encoding="utf-8") as f:
            for linea in f:
                if "=" in linea and not linea.lstrip().startswith("#"):
                    k, v = linea.strip().split("=", 1)
                    env[k] = v
    except FileNotFoundError:
        pass
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith(("NGROK_", "TTS_"))}}


def main() -> None:
    env = leer_env()
    token, user, clave = env.get("NGROK_AUTHTOKEN"), env.get("NGROK_UI_USER"), env.get("NGROK_UI_PASS")
    if not token or not user or not clave:
        sys.exit("Falta NGROK_AUTHTOKEN, NGROK_UI_USER o NGROK_UI_PASS en .env (ver .env.example)")
    if not env.get("TTS_API_KEY"):
        print("AVISO: TTS_API_KEY no está definida: la API quedaría abierta a internet.", file=sys.stderr)

    config = f"""version: "3"
endpoints:
  - name: api
    upstream:
      url: 8000
  - name: ui
    upstream:
      url: 8501
    traffic_policy:
      on_http_request:
        - actions:
            - type: basic-auth
              config:
                credentials:
                  - "{user}:{clave}"
"""
    # El archivo lleva la contraseña: vive en una carpeta temporal y se borra al salir. El token va por entorno.
    with tempfile.NamedTemporaryFile("w", suffix=".yml", delete=False, encoding="utf-8") as f:
        f.write(config)
    proc = subprocess.Popen(["ngrok", "start", "--all", "--config", f.name, "--log", "stdout", "--log-level", "warn"],
                            env={**os.environ, "NGROK_AUTHTOKEN": token})
    try:
        for _ in range(20):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
                    t = json.load(r)["tunnels"]
                if len(t) >= 2:
                    for x in t:
                        print(f"{x['name']:>4}: {x['public_url']}")
                    print(f"\nStreamlit: usuario '{user}' y la contraseña de NGROK_UI_PASS. "
                          f"API: cabecera X-API-Key (TTS_API_KEY) y ngrok-skip-browser-warning: 1")
                    break
            except Exception:
                pass
        proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        proc.terminate()
        os.remove(f.name)


if __name__ == "__main__":
    main()
