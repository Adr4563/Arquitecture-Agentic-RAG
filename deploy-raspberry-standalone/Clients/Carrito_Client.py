"""
Cliente Serial hacia el carrito mecanum (firmware en
../carrito-mecanum-esp32/2-l298n-mecanum/mecanum_car_esp32s3.ino).

El ESP32 va conectado por cable USB directo a esta misma Raspberry Pi (antes
era WiFi/HTTP — ver el propio .ino, que documenta el cambio) — más simple y
sin depender de la LAN ni de que el ESP32 tenga IP. Protocolo: un comando de
texto por línea (F/B/SL/SR/RL/RR/FL/FR/BL/BR/S), igual que antes viajaba como
?move=<código> por HTTP. El ESP32 tiene el mismo watchdog de 500ms que ya
tenía: si no le llega OTRO comando antes de que pasen esos 500ms, frena los
motores solo — una sola llamada ya es un gesto seguro y breve.

CARRITO_PORT configurable por env var si el puerto no es el default:
    export CARRITO_PORT=/dev/ttyUSB0

Sin el puerto disponible (cable desconectado, ESP32 apagado, etc.),
mover()/mover_360() no lanzan excepción — solo loguean y no hacen nada, para
no cortar el flujo de trivia por un carrito que no está conectado.
"""

import os
import time

import serial

CARRITO_PORT = os.environ.get("CARRITO_PORT", "/dev/ttyACM0")
BAUDRATE = 115200
TIMEOUT = 2.0  # cable directo: si no responde rápido, no está conectado/andando.

# Traduce el texto en español de la columna 'desplazamiento' del dataset al
# código que entiende el firmware. SL/SR (strafe) y no RL/RR (rotar) para
# izquierda/derecha: las ruedas mecanum se pueden mover de costado sin girar,
# que es lo que "desplazamiento" sugiere frente a "girar".
_DIRECCION_A_COMANDO = {
    "adelante": "F",
    "atrás": "B",
    "atras": "B",
    "izquierda": "SL",
    "derecha": "SR",
}

# Conexión persistente a nivel de módulo: abrir el puerto reinicia el ESP32
# (toggle de DTR, comportamiento normal de Arduino/ESP32 por USB), así que no
# conviene abrir/cerrar en cada mover() — se abre una sola vez y se reutiliza.
# Si se desconecta el cable, _conexion queda con un objeto muerto y el
# próximo intento de escribir falla y dispara una reconexión (ver _puerto()).
_conexion = None


def _puerto():
    """Devuelve la conexión serial abierta, reconectando si hace falta (primera
    vez, o si la anterior murió por desconexión). None si no se pudo abrir."""
    global _conexion
    if _conexion is not None and _conexion.is_open:
        return _conexion
    try:
        _conexion = serial.Serial(CARRITO_PORT, BAUDRATE, timeout=TIMEOUT)
        # Abrir el puerto reinicia el ESP32 (DTR); setup() tarda un instante
        # en volver a dejarlo listo para recibir comandos — sin esta espera,
        # el primer mover() de la sesión se puede perder en el reinicio.
        time.sleep(2.0)
        return _conexion
    except serial.SerialException as e:
        print(f"    [carrito] no se pudo abrir {CARRITO_PORT}: {e}")
        _conexion = None
        return None


def _mandar(comando):
    conexion = _puerto()
    if conexion is None:
        return False
    try:
        conexion.write(f"{comando}\n".encode())
        return True
    except serial.SerialException as e:
        print(f"    [carrito] no se pudo mandar {comando!r}: {e}")
        global _conexion
        _conexion = None  # forzar reconexión en el próximo intento
        return False


def mover(direccion):
    """Manda UN comando de movimiento — el watchdog del firmware lo frena
    solo a los 500ms si no llega nada más detrás. Devuelve True si el
    comando se pudo escribir al puerto, False si no (sin ESP32
    conectado/apagado, cable desconectado, lo que sea) — nunca lanza."""
    comando = _DIRECCION_A_COMANDO.get((direccion or "").strip().lower())
    if not comando:
        print(f"    [carrito] dirección sin comando mapeado: {direccion!r}, se ignora")
        return False
    if not _mandar(comando):
        print(f"    [carrito] no se pudo mandar {direccion!r} ({comando})")
        return False
    return True


def mover_360(repeticiones=6, pausa=0.4):
    """'Girar 360°' no es un comando único en el firmware (solo RL/RR
    puntuales) — se aproxima mandando 'rotar' varias veces seguidas.

    ⚠️ repeticiones/pausa son un punto de partida SIN CALIBRAR contra el
    carrito real: no hay forma de saber cuántos grados gira por pulso sin
    probarlo con el hardware delante. Ajustalos viendo el carrito girar.
    """
    ok = True
    for _ in range(repeticiones):
        if not _mandar("RR"):
            print("    [carrito] no se pudo mandar 'Girar 360°'")
            ok = False
            break
        time.sleep(pausa)
    return ok
