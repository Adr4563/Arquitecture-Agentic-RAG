"""
Voz de salida: Lora dice en voz alta cada respuesta. DOS motores: uno local
(el default) y uno en la nube como respaldo. Se eligen con VOZ_MOTOR:

    export VOZ_MOTOR=piper   # (default) Piper, 100% local
    export VOZ_MOTOR=edge    # edge-tts, NECESITA INTERNET

Medido en la Raspberry Pi (aarch64), frase de 85 caracteres:

    | motor / voz                | sintesis | vs tiempo real | peso  |
    |----------------------------|----------|----------------|-------|
    | espeak-ng (descartado)     |  0.05s   |  x128          |   --  |
    | piper es_MX-claude-high    |  1.60s   |  x2.86         |  60MB |
    | piper es_ES-sharvard-medium|  1.42s   |  x2.72         |  73MB |
    | piper es_ES-davefx-medium  |  1.39s   |  x2.63         |  60MB |
    | piper es_MX-ald-x_low      |  1.35s   |  x3.52         |  20MB |
    | piper es_AR-daniela-high   |  7.34s   |  x0.45   (!)   | 109MB |
    | Kokoro-82M v1.0            | 12.16s   |  x0.31   (!)   | 310MB |
    | Coqui XTTS-v2              |   --     |  no probado    | ~1.8GB|

hablar() BLOQUEA, asi que la sintesis se le suma a cada turno: ~1.4s de
media sobre las preguntas reales del dataset (p90 2.4s).

Se probo reproducir en streaming para adelantar el primer sonido y se
revirtio por un chasquido al final de cada frase -- ver _hablar_piper().

Se mantienen DOS motores a proposito, uno local y uno de respaldo en la
nube -- espeak-ng quedo afuera: anda y es instantaneo, pero es sintesis por
reglas (formantes), no una red neuronal, y suena claramente robotico al
lado de Piper. Como Piper ya cubre el caso sin internet, no aportaba nada.

Descartados: Kokoro corre a 0.31x tiempo real (12s por frase) pese a tener
solo 82M parametros, y XTTS-v2 es varias veces mas grande -- ni se probo,
arrastra torch/transformers/librosa y un modelo de ~1.8GB. es_AR-daniela-high
tambien queda afuera: 7.3s por frase.

Ojo con el nombre de las voces Piper: el tier ("high"/"medium"/"x_low") NO
predice la velocidad. es_MX-claude-high es "high" y corre a x2.86 con 60MB;
es_AR-daniela-high es el mismo tier, pesa 109MB y va a x0.45. Lo que manda
es el tamano del .onnx.

Default de Piper: es_MX-claude-high -- el mejor tier de calidad entre los
usables, a 0.2s de los "medium", y con acento latinoamericano (los es_ES
son de Espana).

edge-tts (es-AR-ElenaNeural, de Microsoft) sigue siendo el default: se
eligio a mano tras escuchar 46 voces en espanol. Su problema es que NO es
local -- si la Pi se queda sin WiFi, falla. Los otros dos son la respuesta
a eso.

Cualquiera sea el motor, si falla (sin internet, binario ausente, modelo
sin descargar) se loguea y se sigue sin romper el chat -- mismo criterio
que Carrito_Client.py/Musica_Client.py con hardware que no responde.

Reproduce con mpv (mismo binario que Musica_Client) y bloquea hasta que
termina de hablar: el usuario tiene que escuchar antes de que la
conversacion siga.
"""

import asyncio
import os
import subprocess
import tempfile
import wave

# Piper (onnxruntime) agarra los 4 nucleos de la Pi por default. Medido: la
# sintesis tarda lo mismo con 1, 2 o 4 hilos (1.92s / 1.92s / 1.96s), asi que
# limitarlo a 1 es gratis en velocidad y deja 3 nucleos libres para Ollama.
# Tiene que ir ANTES de importar piper/onnxruntime, que leen esto al cargar.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Default: piper (local). Se eligio sobre edge-tts porque no depende de
# internet -- si la Pi se queda sin WiFi, con edge Lora se queda muda. Es
# ademas mas rapido de punta a punta (8.70s vs 11.54s en la misma frase).
#
# Si piper no puede cargar (modelo sin descargar, paquete sin instalar),
# cargar() degrada solo a _MOTOR_RESPALDO en vez de dejar al robot mudo.
MOTOR = os.environ.get("VOZ_MOTOR", "piper").strip().lower()

# A donde caer si el motor pedido no esta disponible. edge-tts es la mejor
# red: no necesita nada instalado en la Pi. Si tampoco hay internet, hablar()
# loguea y el chat sigue por texto.
_MOTOR_RESPALDO = "edge"

# ── edge-tts ──────────────────────────────────────────────────────────
VOZ = os.environ.get("VOZ_EDGE", "es-AR-ElenaNeural")

# ── Piper ─────────────────────────────────────────────────────────────
# Ruta al .onnx. Ver README para descargarlo:
#   python -m piper.download_voices es_MX-ald-x_low
MODELO_PIPER = os.environ.get(
    "VOZ_PIPER_MODELO",
    os.path.expanduser("~/piper-voces/es_MX-claude-high.onnx"),
)
_voz_piper = None   # se carga una sola vez en cargar(), pesa ~5s



def cargar():
    """Precarga lo que el motor activo necesite. La llama
    Orchestrator_Management.py al arrancar, para que el costo no caiga en
    medio de la primera frase.

    edge-tts no tiene nada que cargar (es un servicio en la nube). Piper si: cargar el .onnx tarda ~5s en la Pi, y sin esto ese
    tiempo se lo comeria el primer turno."""
    global _voz_piper, MOTOR
    if MOTOR != "piper":
        return
    try:
        from piper import PiperVoice
        _voz_piper = PiperVoice.load(MODELO_PIPER)
        print(f"[voz] piper listo ({os.path.basename(MODELO_PIPER)})")
    except Exception as e:
        # Puede fallar por dos motivos: falta el paquete (pip install
        # piper-tts) o falta el .onnx (ver README, ollama-style:
        # python -m piper.download_voices es_MX-claude-high). En los dos
        # casos preferimos hablar peor a no hablar, asi que se degrada al
        # motor de respaldo en vez de dejar mudo al robot.
        MOTOR = _MOTOR_RESPALDO
        print(f"[voz] no se pudo cargar piper ({e})")
        print(f"[voz] se usa {MOTOR} como respaldo -- "
              f"para voz local: python3 -m piper.download_voices "
              f"{os.path.basename(MODELO_PIPER).replace('.onnx', '')}")


async def _sintetizar_edge(texto, ruta):
    import edge_tts
    comunicador = edge_tts.Communicate(texto, VOZ)
    await comunicador.save(ruta)


def _generar(texto, ruta):
    """Deja en `ruta` el audio de `texto` segun el motor activo. Devuelve
    False si este motor no puede generar nada ahora."""
    # piper no pasa por aca: usa streaming, sin archivo intermedio.
    asyncio.run(_sintetizar_edge(texto, ruta))
    return True


def _hablar_piper(texto):
    """Sintetiza el texto entero a un .wav temporal y lo reproduce con mpv.

    Se probo antes con streaming (un chunk por oracion mandado a mpv por
    stdin en PCM crudo) y se revirtio: hacia un chasquido audible al final
    de cada frase -- mpv cerraba el dispositivo de audio al terminar el
    stream, y ni 150ms de silencio de cola lo tapaban. La ganancia no lo
    justificaba: el 80% de las preguntas del dataset son de UNA oracion, y
    ahi el streaming ahorraba 0.05s (con 2+ oraciones ahorraba ~0.5s, pero
    es el 20%). Con archivo mpv conoce la duracion de antemano y cierra
    limpio."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        ruta = tmp.name
    try:
        with wave.open(ruta, "wb") as w:
            _voz_piper.synthesize_wav(texto, w)
        subprocess.run(
            ["mpv", "--no-video", "--really-quiet", ruta],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    finally:
        try:
            os.unlink(ruta)
        except OSError:
            pass


def hablar(texto):
    """Sintetiza y reproduce `texto` en voz alta, bloqueando hasta que
    termina. No hace nada si texto viene vacio. Ante cualquier fallo del
    motor loguea y sigue: el chat nunca se cae por un problema de audio."""
    if not texto or not texto.strip():
        return

    if MOTOR == "piper":
        if _voz_piper is None:
            print("[voz] piper no esta cargado (ver cargar()), no se habla")
            return
        try:
            _hablar_piper(texto)
        except FileNotFoundError as e:
            print(f"[voz] falta un binario ({e}) -- no se puede hablar")
        except Exception as e:
            print(f"[voz] error al hablar con piper ({e})")
        return

    # Solo llega edge-tts aca, que devuelve mp3.
    sufijo = ".mp3"
    with tempfile.NamedTemporaryFile(suffix=sufijo, delete=False) as tmp:
        ruta = tmp.name
    try:
        if not _generar(texto, ruta):
            return
        subprocess.run(
            ["mpv", "--no-video", "--really-quiet", ruta],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as e:
        print(f"[voz] falta mpv ({e}) -- no se puede hablar")
    except Exception as e:
        pista = " -- ¿sin internet?" if MOTOR == "edge" else ""
        print(f"[voz] error al hablar con motor={MOTOR} ({e}){pista}")
    finally:
        # El archivo temporal se crea con delete=False (hace falta: mpv lo
        # abre por ruta, en otro proceso), asi que hay que borrarlo a mano o
        # se acumulan en /tmp durante toda la sesion.
        try:
            os.unlink(ruta)
        except OSError:
            pass
