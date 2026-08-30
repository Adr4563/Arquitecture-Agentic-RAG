"""
Medición de tiempo y recursos por componente -- para responder "qué parte
del pipeline tarda más" y "cuánto CPU/memoria usa el proceso a lo largo de
una sesión". Todo vive en este único archivo, en Python puro (+ psutil
opcional): no hace falta un runtime aparte (Node.js, un APM externo) para
esto -- el stack entero corre como UN proceso (Orchestrator_Management.py;
voz_server.py y el resto son threads, no procesos, ver start-all.sh), así
que ni siquiera hace falta psutil multi-PID.

Dos piezas independientes:

1. @medir("nombre_componente") -- decorador para poner sobre funciones que
   ya existen (Agent_Router.enrutar, Llama_Client.generar_respuesta,
   Camara_Client.detectar_emocion, etc.). Cada llamada queda anotada en
   logs/tiempos.csv con cuánto tardó. No cambia el valor de retorno ni el
   comportamiento de la función -- si algo revienta adentro, la excepción
   sigue subiendo tal cual, solo que la fila del log queda con error=1.
   Pensado para varios threads llamando a la vez (voz_server.py corre en su
   propio thread): la escritura al CSV va con lock.

2. iniciar_muestreo_recursos() -- un hilo daemon que cada
   INTERVALO_MUESTREO_SEG segundos (default 5, configurable con la env var
   PERF_MUESTREO_SEG) anota en logs/recursos.csv el CPU% y la memoria RSS
   del proceso entero. Si psutil no está instalado, se desactiva solo (un
   aviso una vez) -- @medir no depende de psutil, solo de
   time.perf_counter(), así que el timing por componente sigue funcionando
   igual.

Para ver el resumen ("quién tarda más", "cómo vino el CPU/memoria"):
    python perf_report.py
"""

import csv
import functools
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(HERE, "logs")
RUTA_TIEMPOS = os.path.join(LOGS_DIR, "tiempos.csv")
RUTA_RECURSOS = os.path.join(LOGS_DIR, "recursos.csv")

INTERVALO_MUESTREO_SEG = float(os.environ.get("PERF_MUESTREO_SEG", "5"))

_ENCABEZADOS_TIEMPOS = ["timestamp", "componente", "duracion_ms", "error"]
_ENCABEZADOS_RECURSOS = ["timestamp", "cpu_percent", "memoria_rss_mb", "hilos_activos"]

# Un solo lock alcanza para los dos archivos: son escrituras cortas
# (append de una fila) y no hay contención real entre ellas -- no vale la
# pena un lock por archivo.
_lock = threading.Lock()


def _escribir_fila(ruta, encabezados, fila):
    os.makedirs(LOGS_DIR, exist_ok=True)
    with _lock:
        nuevo = not os.path.exists(ruta)
        with open(ruta, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if nuevo:
                writer.writerow(encabezados)
            writer.writerow(fila)


@contextmanager
def medir_bloque(componente):
    """Como @medir, pero para envolver un tramo de código en vez de una
    función entera -- hace falta cuando el nombre del componente depende de
    un argumento de la llamada (ej. Llama_Client.generar_respuesta():
    "llama_generar:lora-chat" vs "llama_generar:lora-trivia", para no
    mezclar en un mismo promedio dos modelos con pesos/latencias muy
    distintas, ver la nota de CHAT_MODEL/TRIVIA_MODEL ahí)."""
    inicio = time.perf_counter()
    error = 0
    try:
        yield
    except Exception:
        error = 1
        raise
    finally:
        duracion_ms = (time.perf_counter() - inicio) * 1000
        _escribir_fila(
            RUTA_TIEMPOS, _ENCABEZADOS_TIEMPOS,
            [datetime.now().isoformat(timespec="seconds"), componente,
             f"{duracion_ms:.1f}", error],
        )


def medir(componente):
    """Decorador: cronometra cada llamada a la función decorada y la anota
    en logs/tiempos.csv bajo el nombre `componente` -- no el nombre de la
    función, así varias funciones pueden compartir un componente (o una
    función puede loguearse con un nombre más legible que el real). Para
    cuando `componente` no es fijo (depende de un argumento), ver
    medir_bloque() más abajo."""
    def decorador(func):
        @functools.wraps(func)
        def envoltorio(*args, **kwargs):
            with medir_bloque(componente):
                return func(*args, **kwargs)
        return envoltorio
    return decorador


_muestreo_iniciado = False


def iniciar_muestreo_recursos():
    """Arranca (una sola vez) el hilo que muestrea CPU/memoria del proceso
    cada INTERVALO_MUESTREO_SEG segundos. Se llama una sola vez desde main()
    de Orchestrator_Management.py, igual que voz_output.cargar() o
    _precargar_modelos() -- no bloquea, el hilo queda como daemon."""
    global _muestreo_iniciado
    if _muestreo_iniciado:
        return
    _muestreo_iniciado = True

    try:
        import psutil
    except ImportError:
        print("[perf_monitor] psutil no instalado -- sin muestreo de CPU/memoria "
              "(agregar 'psutil' a requirements.txt). El timing por componente "
              "(@medir) sigue funcionando igual, no depende de psutil.")
        return

    proceso = psutil.Process(os.getpid())
    proceso.cpu_percent()  # la primera llamada siempre da 0.0 -- se descarta acá,
    # así la primera fila real ya viene con un intervalo de referencia.

    def _loop():
        while True:
            time.sleep(INTERVALO_MUESTREO_SEG)
            try:
                cpu = proceso.cpu_percent()
                mem_mb = proceso.memory_info().rss / (1024 * 1024)
                hilos = proceso.num_threads()
            except Exception:
                continue
            _escribir_fila(
                RUTA_RECURSOS, _ENCABEZADOS_RECURSOS,
                [datetime.now().isoformat(timespec="seconds"),
                 f"{cpu:.1f}", f"{mem_mb:.1f}", hilos],
            )

    threading.Thread(target=_loop, daemon=True).start()
    print(f"[perf_monitor] muestreo de recursos activo cada {INTERVALO_MUESTREO_SEG:.0f}s -> {RUTA_RECURSOS}")
