"""Arranca la API escuchando en IPv4 e IPv6 a la vez.

El gateway de Salad se conecta por IPv6; `--host ::` de uvicorn deja el socket solo en IPv6 (asyncio activa
IPV6_V6ONLY) y `--host 0.0.0.0` solo en IPv4. Aquí se crea un socket de doble pila. Si el contenedor no tiene IPv6, cae a IPv4.
"""
import os
import socket
import sys

import uvicorn

sys.path.insert(0, os.path.join(os.getcwd(), "tts_app"))  # equivale a --app-dir tts_app
PUERTO = int(os.environ.get("PORT", "8000"))


def socket_doble_pila() -> socket.socket:
    s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    s.bind(("::", PUERTO))
    return s


if __name__ == "__main__":
    config = uvicorn.Config("api:app", host="0.0.0.0", port=PUERTO)
    try:
        uvicorn.Server(config).run(sockets=[socket_doble_pila()])
    except OSError as e:  # sin IPv6 en el contenedor
        print(f"IPv6 no disponible ({e}); escuchando solo en IPv4")
        uvicorn.Server(config).run()
