"""
Cliente hacia la cámara IMX500 para el veredicto real del "Juego de
emociones"/"Juego de imitación" (ver ai-camera/TODO-emociones-imx500.txt
para el porqué del enfoque CPU+ONNX en vez del sensor NPU, y
ai-camera/reconocer_emocion.py para el detalle de los dos modelos).

El código de detección vive en ai-camera/ -- carpeta HERMANA de esta, no una
subcarpeta -- porque también lo usan snapshot_deteccion.py/live_stream.py de
ese proyecto para detección de objetos; no se duplica acá, se importa vía
sys.path. El import es perezoso (adentro de detectar_emocion(), no al tope
del módulo) para que Orchestrator_Management.py arranque igual en una Pi sin
la cámara conectada o sin onnxruntime/opencv/picamera2 instalados -- mismo
criterio que Carrito_Client.py/Musica_Client.py con su hardware: si algo
falla, se loguea y se sigue sin veredicto de cámara, nunca una excepción que
corte el turno de trivia.
"""
import os
import sys

import perf_monitor

HERE = os.path.dirname(os.path.abspath(__file__))
AI_CAMERA_DIR = os.path.join(HERE, "..", "..", "ai-camera")


@perf_monitor.medir("camara")
def detectar_emocion():
    """Captura un frame con la cámara y devuelve (emocion, confianza) --
    emocion ya traducida a Feliz/Triste/Enojado/Neutral (las mismas 4 que
    usa la columna 'cara' del dataset de trivia). Devuelve (None, None) si
    no se pudo evaluar por CUALQUIER motivo: dependencias no instaladas,
    cámara no conectada, o cámara conectada pero no se detectó ninguna cara
    en varios intentos (ver reconocer_emocion._capturar_y_detectar)."""
    if AI_CAMERA_DIR not in sys.path:
        sys.path.insert(0, AI_CAMERA_DIR)
    try:
        from reconocer_emocion import ETIQUETAS_RELEVANTES, _capturar_y_detectar
    except ImportError as e:
        print(f"    [cámara] no disponible ({e}) -- sigue sin veredicto de cámara")
        return None, None

    try:
        emocion, confianza = _capturar_y_detectar()
    except Exception as e:
        print(f"    [cámara] falló la captura/detección ({e})")
        return None, None

    if emocion is None:
        print("    [cámara] no se detectó ninguna cara -- sigue sin veredicto")
        return None, None
    return ETIQUETAS_RELEVANTES.get(emocion), confianza
