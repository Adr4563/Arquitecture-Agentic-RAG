"""
Controla la carita que se muestra en pantalla (faces/{happy,sad,angry,
content,speaking}.{mp4,gif} — mp4 para el backend DRM, con hwdec por
hardware; gif para el backend Tkinter, que no lee mp4. Ver mostrar_cara()).

La pantalla es "hardware conectado a la máquina donde corre chat.py" (el
Manager), por eso vive junto a él en vez de junto a preguntas.py.

Dos backends, elegidos automáticamente según dónde se corre (mismo
mostrar_cara/detener para los dos, así que el resto de chat.py no se entera
de cuál está activo):

- **DRM (Raspberry Pi con LCD/HDMI directo, sin sesión gráfica):** usa `mpv`
  con salida `--vo=drm`, que dibuja directo sobre el framebuffer sin
  necesitar X11/Wayland. Un solo proceso `mpv --idle` se lanza una vez y
  queda vivo toda la sesión; cambiar de cara es un `loadfile` por su socket
  IPC (`--input-ipc-server`), no un relanzo — evita el parpadeo de
  reabrir ventana/reconectar a la pantalla en cada cambio.
  Requiere `sudo apt install mpv` y que el usuario esté en el grupo `video`
  (`sudo usermod -aG video,render $USER`, y volver a iniciar sesión para que
  el grupo aplique — un `newgrp video` alcanza para probarlo en la sesión
  actual sin cerrar sesión).
- **Tkinter (demo en PC con escritorio, ej. Windows):** usa face_viewer.py,
  un visor propio en Tkinter (viene incluido con Python, no depende de
  instalar nada aparte ni de tener algo en el PATH). Igual que con mpv, se
  lanza UNA sola vez y se queda vivo durante toda la sesión: cambiar de cara
  no relanza la ventana (eso competía por foco/z-order con la terminal cada
  vez, y a veces se perdía esa carrera y la ventana nueva quedaba tapada) —
  solo se reescribe qué gif tiene que mostrar, vía un archivo de señal que
  el visor revisa periódicamente.

La detección es automática: Linux sin $DISPLAY (headless, HDMI/LCD directo)
usa mpv/DRM; cualquier otro caso (Windows, o Linux con sesión gráfica) usa
Tkinter.
"""

import json
import os
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FACES_DIR = os.path.join(HERE, "faces")
VIEWER = os.path.join(HERE, "face_viewer.py")
ARCHIVO_SENAL = os.path.join(HERE, ".current_face")
MPV_SOCKET = "/tmp/arquitecture_rag_face.sock"

CARAS_VALIDAS = {"happy", "sad", "angry", "content", "speaking"}

# Headless en Linux (sin sesión gráfica) == pantalla conectada directo por
# HDMI/LCD a la Pi, sin X11/Wayland de por medio -> DRM. Con $DISPLAY seteado
# (o en Windows) hay un escritorio real -> Tkinter.
USA_DRM = sys.platform.startswith("linux") and not os.environ.get("DISPLAY")

_proceso = None


# ─── Backend DRM (mpv, Raspberry Pi headless) ────────────────────────────

def _mpv_enviar(comando):
    """Manda un comando IPC a mpv (conexión corta, una por llamada: no hace
    falta mantener el socket abierto entre cambios de cara, que son
    esporádicos)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(MPV_SOCKET)
        s.sendall((json.dumps({"command": comando}) + "\n").encode())


def _asegurar_visor_drm(ruta_gif_inicial):
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        return
    try:
        os.remove(MPV_SOCKET)
    except FileNotFoundError:
        pass
    _proceso = subprocess.Popen(
        [
            "mpv", "--fs", "--vo=drm", "--idle=yes", "--loop-file=inf",
            "--no-osc", "--no-input-default-bindings", "--really-quiet",
            # Las caras acá son .mp4 (H.264), no .gif: con --hwdec=auto mpv
            # decodifica con el v4l2m2m de la Pi (drmprime, zero-copy al
            # framebuffer) en vez de por software — se midió ~100-137% CPU
            # (más de un núcleo) por software vs ~10-23% por hardware para
            # estas mismas animaciones. GIF no tiene decodificador de
            # hardware en ningún lado, por eso se convirtieron a video.
            "--hwdec=auto",
            f"--input-ipc-server={MPV_SOCKET}", ruta_gif_inicial,
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # El socket IPC tarda un instante en existir tras el Popen; sin esta
    # espera, el primer mostrar_cara() puede pisarse con la creación del
    # socket y perder el primer loadfile.
    for _ in range(50):  # hasta 5s
        if os.path.exists(MPV_SOCKET):
            break
        time.sleep(0.1)


def _mostrar_cara_drm(ruta_gif):
    _asegurar_visor_drm(ruta_gif)
    _mpv_enviar(["loadfile", ruta_gif, "replace"])


def _detener_drm():
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        try:
            _mpv_enviar(["quit"])
            _proceso.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            _proceso.terminate()
    _proceso = None
    try:
        os.remove(MPV_SOCKET)
    except FileNotFoundError:
        pass


# ─── Backend Tkinter (demo en PC con escritorio) ─────────────────────────

def _asegurar_visor_tk(ruta_gif_inicial):
    """Lanza la ventana del visor si todavía no hay una viva. No hace nada
    si ya hay una corriendo — ahí el cambio de cara va por el archivo de
    señal, no por relanzar el proceso."""
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        return
    _proceso = subprocess.Popen(
        [sys.executable, VIEWER, ruta_gif_inicial, ARCHIVO_SENAL],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _mostrar_cara_tk(ruta_gif):
    _asegurar_visor_tk(ruta_gif)
    with open(ARCHIVO_SENAL, "w", encoding="utf-8") as f:
        f.write(ruta_gif)


def _detener_tk():
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        _proceso.terminate()
        try:
            _proceso.wait(timeout=2)  # sin esto, Windows todavía tiene el archivo
        except subprocess.TimeoutExpired:  # abierto un instante y el remove() de abajo falla
            pass
    _proceso = None
    try:
        os.remove(ARCHIVO_SENAL)
    except (FileNotFoundError, PermissionError):
        pass


# ─── Interfaz pública (no cambia según el backend) ───────────────────────

def mostrar_cara(nombre):
    """Muestra faces/<nombre> en loop, en la ventana/pantalla ya abierta
    (o recién abierta si es la primera vez).

    Extensión según backend: DRM usa faces/<nombre>.mp4 (H.264 + --hwdec=auto
    en _asegurar_visor_drm, decodificado por el v4l2m2m de la Pi — ver la nota
    ahí sobre por qué se convirtió desde .gif, que no tiene hwdec en ningún
    lado). Tkinter (demo en PC) sigue con faces/<nombre>.gif, porque
    face_viewer.py anima con tk.PhotoImage(format="gif ..."), que no lee mp4."""
    if nombre not in CARAS_VALIDAS:
        print(f"[display] Cara desconocida: {nombre!r} (válidas: {sorted(CARAS_VALIDAS)})")
        return

    extension = "mp4" if USA_DRM else "gif"
    ruta = os.path.join(FACES_DIR, f"{nombre}.{extension}")
    if not os.path.isfile(ruta):
        print(f"[display] No existe {ruta}")
        return

    if USA_DRM:
        _mostrar_cara_drm(ruta)
    else:
        _mostrar_cara_tk(ruta)


def detener():
    """Cierra la ventana/proceso del visor, si hay uno corriendo (ej. al
    salir del chat)."""
    if USA_DRM:
        _detener_drm()
    else:
        _detener_tk()
