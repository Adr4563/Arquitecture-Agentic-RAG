import os
import subprocess

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUSICA_DIR = os.path.join(HERE, "musica")
REPRODUCCION_MAX_SEG = 20


# Margen sobre REPRODUCCION_MAX_SEG para el modo bloqueante: mpv tarda un
# poco en arrancar y en cerrar, y si el timeout fuera exacto cortaria el
# final de la cancion. Solo aplica como red de seguridad -- si mpv se
# colgara, el turno no se queda esperando para siempre.
_TIMEOUT_ESPERA_SEG = REPRODUCCION_MAX_SEG + 10


def reproducir(nombre_archivo, esperar=False):
    """Reproduce un archivo de musica/ recortado a REPRODUCCION_MAX_SEG.

    `esperar=False` (default): lanza mpv y vuelve enseguida. Es lo que
    quieren las reacciones -- la musica suena de fondo mientras el turno
    sigue.

    `esperar=True`: BLOQUEA hasta que termina de sonar. Lo usan las
    preguntas de Reconocimiento Musical, donde el usuario tiene que escuchar
    la cancion ANTES de poder responder: si no bloqueara, se le pediria la
    respuesta encima de la musica recien arrancada.

    Devuelve True si se llego a reproducir, False si no habia archivo o no
    esta mpv."""
    ruta = os.path.join(MUSICA_DIR, nombre_archivo)
    if not os.path.isfile(ruta):
        print(f"    [música] no existe {ruta}, no se reproduce")
        return False
    cmd = ["mpv", "--no-video", f"--length={REPRODUCCION_MAX_SEG}",
           "--really-quiet", ruta]
    try:
        if not esperar:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=_TIMEOUT_ESPERA_SEG)
        return True
    except FileNotFoundError:
        print("    [música] mpv no está instalado, no se puede reproducir")
        return False
    except subprocess.TimeoutExpired:
        # mpv colgado: se reporta y se sigue. Nunca dejar el turno trabado
        # por un problema de audio -- mismo criterio que el resto de los
        # clientes de hardware (ver la nota en Agent_Behavior.py).
        print(f"    [música] mpv no termino en {_TIMEOUT_ESPERA_SEG}s, se sigue igual")
        return True
