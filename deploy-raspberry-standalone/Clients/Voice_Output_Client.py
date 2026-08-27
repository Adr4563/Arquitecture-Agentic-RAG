"""
Voz de salida: Lora habla en voz alta cada respuesta con edge-tts (voz
es-AR-ElenaNeural, de Microsoft Edge) — elegida a mano tras escuchar 46
voces en español de edge-tts, más las de Piper/Kokoro/eSpeak NG antes.

⚠️ A diferencia de Piper (lo que usaba antes), edge-tts NECESITA INTERNET —
usa el servicio en la nube de Microsoft (gratis, sin cuenta ni API key),
pero no es local. Si la Pi se queda sin WiFi, esto falla gracioso (log, no
excepción) y el chat sigue funcionando solo con texto — mismo criterio que
Carrito_Client.py/Musica_Client.py con hardware que no responde. La Pi ya
necesita WiFi de todos modos para Ollama y la página de voz_server.py, así
que no es un requisito nuevo del despliegue.

Reproduce con mpv (mismo binario que el resto) y BLOQUEA hasta que termina
de hablar — mismo criterio que los time.sleep(PAUSA_CAMBIO_CARA) que ya
tiene Orchestrator_Management.py: el usuario tiene que terminar de escuchar
antes de que la conversación siga.
"""

import asyncio
import os
import subprocess
import tempfile

VOZ = "es-AR-ElenaNeural"


def cargar():
    """No hay ningún modelo que cargar acá (edge-tts es un servicio en la
    nube, no un modelo local) — la función existe solo para que
    Orchestrator_Management.py no tenga que distinguir qué motor de voz está
    activo al arrancar."""
    pass


async def _sintetizar(texto, ruta):
    import edge_tts
    comunicador = edge_tts.Communicate(texto, VOZ)
    await comunicador.save(ruta)


def hablar(texto):
    """Sintetiza y reproduce `texto` en voz alta, bloqueando hasta que
    termina. No hace nada si texto viene vacío. Si no hay internet o mpv no
    está instalado, loguea y sigue sin romper el chat."""
    if not texto or not texto.strip():
        return

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        ruta = tmp.name
    try:
        asyncio.run(_sintetizar(texto, ruta))
        subprocess.run(
            ["mpv", "--no-video", "--really-quiet", ruta],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("[voz] mpv no está instalado, no se puede reproducir")
    except Exception as e:
        print(f"[voz] error al hablar ({e}) — ¿sin internet?")
    finally:
        try:
            os.remove(ruta)
        except OSError:
            pass
