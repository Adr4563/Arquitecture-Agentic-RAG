"""
Reconocimiento de emoción facial (feliz / triste / enojado / etc.) para
"Juego de emociones" y "Juego de imitación" -- ver TODO-emociones-imx500.txt
para el porqué de este enfoque en particular.

Camino elegido: CPU (no el sensor IMX500). Se investigó el model zoo oficial
de Sony/Raspberry Pi y NO hay ningún modelo de emociones publicado para el
IMX500 (solo detección de objetos, clasificación genérica, segmentación,
pose) -- entrenar uno propio y convertirlo a .rpk necesita el "IMX500
Converter" de Sony, que corre en Linux x86_64, no en esta Raspberry Pi (ARM).

En vez de eso: dos modelos ONNX chicos y YA ENTRENADOS (nada que entrenar
acá), corriendo con `onnxruntime` (mucho más liviano que TensorFlow/Keras,
la alternativa CPU que se había descartado antes por pesada):

- `modelos/face_detection_yunet_2023mar.onnx` (~230KB): detector de caras
  YuNet, de OpenCV Zoo. Reemplaza a los Haar cascades clásicos
  (`cv2.CascadeClassifier`), que OpenCV 5.x sacó del build por default.
- `modelos/emotion-ferplus-8.onnx` (~35MB): clasificador de emoción,
  entrenado sobre el dataset FER+ (ONNX Model Zoo, modelo público de
  referencia). Entrada: cara recortada, 64x64, escala de grises. Salida:
  8 clases -- neutral/felicidad/sorpresa/tristeza/enojo/asco/miedo/desprecio.

Probado con imágenes de archivo (ver `detectar_emocion_en_imagen`) contra
fotos de referencia conocidas -- todavía NO probado con la cámara real de
esta Raspberry Pi (no hay una conectada donde se escribió esto). Antes de
usarlo en producción, correr `python reconocer_emocion.py` acá con la
cámara conectada y confirmar que detecta bien caras reales, con buena luz.

NO integrado todavía al flujo de Trivia (Orchestrator_Management.py) --
eso es diseño nuevo aparte (cómo mapear la emoción detectada al veredicto
acierto/error), ver la sección correspondiente en TODO-emociones-imx500.txt
antes de tocar ese archivo.
"""
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODELO_EMOCION = os.path.join(HERE, "modelos", "emotion-ferplus-8.onnx")
MODELO_CARA = os.path.join(HERE, "modelos", "face_detection_yunet_2023mar.onnx")

ETIQUETAS = ["neutral", "felicidad", "sorpresa", "tristeza", "enojo", "asco", "miedo", "desprecio"]
# Las 3 que importan para el dataset de trivia (columna 'cara': Feliz/Triste/Enojado).
ETIQUETAS_RELEVANTES = {"felicidad": "Feliz", "tristeza": "Triste", "enojo": "Enojado"}

_sess_emocion = ort.InferenceSession(MODELO_EMOCION)
_input_name = _sess_emocion.get_inputs()[0].name
# OpenCV 5.x sacó cv2.CascadeClassifier (Haar) del build por default --
# FaceDetectorYN es el reemplazo moderno (también ONNX, más preciso).
_detector_cara = cv2.FaceDetectorYN_create(MODELO_CARA, "", (320, 320), score_threshold=0.7)


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


def detectar_emocion_en_frame(frame_bgr):
    """Recibe un frame BGR (numpy array, como los que da OpenCV/Picamera2) y
    devuelve (emocion, confianza) -- emocion es una de ETIQUETAS, o
    (None, None) si no se detectó ninguna cara."""
    alto, ancho = frame_bgr.shape[:2]
    _detector_cara.setInputSize((ancho, alto))
    _, caras = _detector_cara.detect(frame_bgr)
    if caras is None or len(caras) == 0:
        return None, None

    cara = max(caras, key=lambda c: c[14])  # la de mayor confianza si hay varias
    x, y, w, h = [max(0, int(v)) for v in cara[:4]]
    gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    recorte = gris[y:y + h, x:x + w]
    if recorte.size == 0:
        return None, None
    recorte = cv2.resize(recorte, (64, 64)).astype(np.float32)
    entrada = recorte.reshape(1, 1, 64, 64)

    salida = _sess_emocion.run(None, {_input_name: entrada})[0][0]
    probs = _softmax(salida)
    idx = int(np.argmax(probs))
    return ETIQUETAS[idx], float(probs[idx])


def detectar_emocion_en_imagen(ruta_imagen):
    """Como detectar_emocion_en_frame, pero desde un archivo -- útil para
    probar sin cámara (ver el bloque __main__ para el modo con cámara)."""
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError(f"no se pudo leer {ruta_imagen}")
    return detectar_emocion_en_frame(img)


def _capturar_y_detectar():
    """Toma una foto con la Raspberry Pi AI Camera (picamera2, no el modo
    NPU del IMX500 -- acá se usa como cámara normal, la inferencia corre en
    la CPU con los modelos ONNX de arriba) y detecta la emoción."""
    from picamera2 import Picamera2

    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.start(config, show_preview=False)
    time.sleep(2)  # deja asentar AE/AWB, mismo criterio que snapshot_deteccion.py

    emocion, confianza = None, None
    for _ in range(15):
        frame = picam2.capture_array("main")
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        emocion, confianza = detectar_emocion_en_frame(frame_bgr)
        if emocion is not None:
            break
        time.sleep(0.2)

    picam2.stop()
    return emocion, confianza


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Modo prueba sin cámara: python reconocer_emocion.py <ruta_imagen>
        emocion, confianza = detectar_emocion_en_imagen(sys.argv[1])
    else:
        # Modo real: captura con la cámara conectada a esta Raspberry Pi.
        emocion, confianza = _capturar_y_detectar()

    if emocion is None:
        print("No se detectó ninguna cara.")
    else:
        relevante = ETIQUETAS_RELEVANTES.get(emocion)
        extra = f" (== '{relevante}' del dataset de trivia)" if relevante else " (no es una de las 3 que usa el dataset de trivia)"
        print(f"Emoción detectada: {emocion} (confianza: {confianza:.2%}){extra}")
