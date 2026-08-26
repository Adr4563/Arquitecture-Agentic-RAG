"""
Cliente HTTP hacia el carrito mecanum (firmware en
../carrito-mecanum-esp32/2-l298n-mecanum/mecanum_car_esp32s3.ino).

Protocolo del firmware: GET /cmd?move=<código>, uno de F/B/SL/SR/RL/RR/FL/
FR/BL/BR — el propio ESP32 tiene un watchdog de 500ms: si no le llega OTRO
comando antes de que pasen esos 500ms, frena los motores solo. Por eso una
sola llamada ya es un gesto seguro y breve — no hace falta mandar un STOP
explícito después.

El ESP32 no tiene IP fija ni mDNS (ver el propio .ino: la IP se imprime por
Serial al conectar y puede cambiar con el DHCP del router) — hay que
exportar CARRITO_HOST a mano con la IP del momento:
    export CARRITO_HOST=http://192.168.1.50

Sin esa variable configurada, mover()/mover_360() no lanzan excepción — solo
loguean y no hacen nada, para no cortar el flujo de trivia por un carrito
que no está prendido/conectado en este momento.
"""

import os
import time

import requests

CARRITO_HOST = os.environ.get("CARRITO_HOST", "")  # ej. http://192.168.1.50
TIMEOUT = 2.0  # está en la misma LAN; si no responde rápido, no está conectado.

# Traduce el texto en español de la columna 'desplazamiento' del dataset al
# código que entiende server.arg("move") en el firmware. SL/SR (strafe) y no
# RL/RR (rotar) para izquierda/derecha: las ruedas mecanum se pueden mover de
# costado sin girar, que es lo que "desplazamiento" sugiere frente a "girar".
_DIRECCION_A_COMANDO = {
    "adelante": "F",
    "atrás": "B",
    "atras": "B",
    "izquierda": "SL",
    "derecha": "SR",
}


def mover(direccion):
    """Manda UN comando de movimiento — el watchdog del firmware lo frena
    solo a los 500ms si no llega nada más detrás. Devuelve True si el
    carrito respondió, False si no se pudo mandar (sin CARRITO_HOST, carrito
    apagado/fuera de red, lo que sea) — nunca lanza."""
    comando = _DIRECCION_A_COMANDO.get((direccion or "").strip().lower())
    if not comando:
        print(f"    [carrito] dirección sin comando mapeado: {direccion!r}, se ignora")
        return False
    if not CARRITO_HOST:
        print(f"    [carrito] CARRITO_HOST no configurado, no se manda {direccion!r}")
        return False
    try:
        requests.get(f"{CARRITO_HOST}/cmd", params={"move": comando}, timeout=TIMEOUT)
        return True
    except requests.RequestException as e:
        print(f"    [carrito] no se pudo mandar {direccion!r} ({comando}): {e}")
        return False


def mover_360(repeticiones=6, pausa=0.4):
    """'Girar 360°' no es un comando único en el firmware (solo RL/RR
    puntuales) — se aproxima mandando 'rotar' varias veces seguidas.

    ⚠️ repeticiones/pausa son un punto de partida SIN CALIBRAR contra el
    carrito real: no hay forma de saber cuántos grados gira por pulso sin
    probarlo con el hardware delante. Ajustalos viendo el carrito girar.
    """
    if not CARRITO_HOST:
        print("    [carrito] CARRITO_HOST no configurado, no se manda 'Girar 360°'")
        return False
    ok = True
    for _ in range(repeticiones):
        try:
            requests.get(f"{CARRITO_HOST}/cmd", params={"move": "RR"}, timeout=TIMEOUT)
        except requests.RequestException as e:
            print(f"    [carrito] no se pudo mandar 'Girar 360°': {e}")
            ok = False
            break
        time.sleep(pausa)
    return ok
