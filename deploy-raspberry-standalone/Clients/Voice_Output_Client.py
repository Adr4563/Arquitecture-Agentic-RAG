"""
Voz de salida: Lora dice en voz alta cada respuesta. Tres motores, elegidos
con VOZ_MOTOR:

    export VOZ_MOTOR=edge     # (default) edge-tts, NECESITA INTERNET
    export VOZ_MOTOR=espeak   # eSpeak NG, 100% local
    export VOZ_MOTOR=piper    # Piper, 100% local, mejor calidad

Medido en la Raspberry Pi (aarch64), frase de 85 caracteres:

    | motor            | sintesis | audio | vs tiempo real |
    |------------------|----------|-------|----------------|
    | espeak-ng        |  0.05s   | 6.1s  |  x128          |
    | piper es_MX x_low|  1.5s    | 5.5s  |  x3.5          |
    | piper es_AR high |  9.1s    | 4.1s  |  x0.44  (!)    |

hablar() BLOQUEA, asi que ese tiempo se le suma a cada turno. Por eso la
voz "high" de Piper esta descartada: 9 segundos de silencio por frase hacen
al robot inusable, aunque suene mejor. x_low es el punto medio razonable si
espeak-ng suena demasiado robotico.

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

MOTOR = os.environ.get("VOZ_MOTOR", "edge").strip().lower()

# ── edge-tts ──────────────────────────────────────────────────────────
VOZ = os.environ.get("VOZ_EDGE", "es-AR-ElenaNeural")

# ── eSpeak NG ─────────────────────────────────────────────────────────
# es-la = espanol latinoamericano (mas cercano al de los chicos que lo usan
# que es-es). 150 ppm: el default (175) suena atropellado para leer una
# pregunta de trivia en voz alta.
VOZ_ESPEAK = os.environ.get("VOZ_ESPEAK", "es-la")
VELOCIDAD_ESPEAK = os.environ.get("VOZ_ESPEAK_VELOCIDAD", "150")

# ── Piper ─────────────────────────────────────────────────────────────
# Ruta al .onnx. Ver README para descargarlo:
#   python -m piper.download_voices es_MX-ald-x_low
MODELO_PIPER = os.environ.get(
    "VOZ_PIPER_MODELO",
    os.path.expanduser("~/piper-voces/es_MX-ald-x_low.onnx"),
)
_voz_piper = None   # se carga una sola vez en cargar(), pesa ~5s


def cargar():
    """Precarga lo que el motor activo necesite. La llama
    Orchestrator_Management.py al arrancar, para que el costo no caiga en
    medio de la primera frase.

    edge/espeak no tienen nada que cargar (servicio en la nube y binario del
    sistema). Piper si: cargar el .onnx tarda ~5s en la Pi, y sin esto ese
    tiempo se lo comeria el primer turno."""
    global _voz_piper
    if MOTOR != "piper":
        return
    try:
        from piper import PiperVoice
        _voz_piper = PiperVoice.load(MODELO_PIPER)
        print(f"[voz] piper listo ({os.path.basename(MODELO_PIPER)})")
    except Exception as e:
        print(f"[voz] no se pudo cargar piper ({e}) -- se sigue sin voz")


async def _sintetizar_edge(texto, ruta):
    import edge_tts
    comunicador = edge_tts.Communicate(texto, VOZ)
    await comunicador.save(ruta)


def _generar(texto, ruta):
    """Deja en `ruta` el audio de `texto` segun el motor activo. Devuelve
    False si este motor no puede generar nada ahora."""
    if MOTOR == "espeak":
        subprocess.run(
            ["espeak-ng", "-v", VOZ_ESPEAK, "-s", VELOCIDAD_ESPEAK, "-w", ruta, texto],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        return True
    if MOTOR == "piper":
        if _voz_piper is None:
            print("[voz] piper no esta cargado (ver cargar()), no se habla")
            return False
        import wave
        with wave.open(ruta, "wb") as w:
            _voz_piper.synthesize_wav(texto, w)
        return True
    asyncio.run(_sintetizar_edge(texto, ruta))
    return True


def hablar(texto):
    """Sintetiza y reproduce `texto` en voz alta, bloqueando hasta que
    termina. No hace nada si texto viene vacio. Ante cualquier fallo del
    motor loguea y sigue: el chat nunca se cae por un problema de audio."""
    if not texto or not texto.strip():
        return

    # .wav para los motores locales, .mp3 para edge-tts (es lo que devuelve).
    sufijo = ".wav" if MOTOR in ("espeak", "piper") else ".mp3"
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
        # Falta mpv, o falta espeak-ng con VOZ_MOTOR=espeak.
        print(f"[voz] falta un binario ({e}) -- no se puede hablar")
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
