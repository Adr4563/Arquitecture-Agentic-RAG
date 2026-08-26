# ai-camera

Scripts de detección de objetos en tiempo real usando la **Raspberry Pi AI Camera**
(sensor Sony IMX500 con NPU integrada) a través de `picamera2`. La inferencia corre
dentro del propio sensor de la cámara, no en la CPU de la Pi.

También incluye reconocimiento de emoción facial (`reconocer_emocion.py`), que
a diferencia de los otros 3 scripts corre en la **CPU** de la Pi, no en el sensor
— ver la sección propia más abajo para el porqué.

## Archivos

| Archivo | Qué hace |
|---|---|
| `hdmi_live.py` | Detección en vivo con preview directo por HDMI (DRM), sin necesidad de escritorio/X11. Pensado para un monitor conectado directo a la Pi. |
| `live_stream.py` | Servidor HTTP (puerto `5001`) que sirve un stream MJPEG con las detecciones dibujadas, para verlo desde otra máquina de la red. |
| `snapshot_deteccion.py` | Toma una sola foto (hasta 15 intentos buscando una detección), dibuja las cajas y la guarda en `output/`. |
| `reconocer_emocion.py` | Detecta cara + clasifica emoción (feliz/triste/enojado/etc.), corriendo en CPU con `onnxruntime` -- ver "Reconocimiento de emoción facial" más abajo. |

## Requisitos de hardware

- Raspberry Pi con la **AI Camera** (sensor IMX500) conectada al puerto CSI.
- Raspberry Pi OS / Ubuntu con soporte de `libcamera`.

## Instalación

```bash
cd ai-camera
./install.sh
```

Esto instala:

- **Paquetes de sistema (apt):** `python3-libcamera`, `python3-kms++`, `libcap-dev`
  e **`imx500-all`** (firmware + modelos oficiales de la AI Camera).
- **Paquetes de Python (pip):** `opencv-python`, `numpy`, `picamera2`
  (vía `sudo pip3 install --break-system-packages -r requirements.txt`).

## Modelos: dónde van y cómo se instalan

**Los modelos NO se descargan a mano ni se guardan dentro de este repo.** Vienen
con el paquete apt `imx500-all`, que los deja a nivel de sistema:

- Modelos (`.rpk`): `/usr/share/imx500-models/`
  Los 3 scripts usan el mismo por defecto:
  `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk`
  (detección de objetos, SSD MobileNetV2 FPN-Lite 320×320).
- Etiquetas (labels): vienen con el paquete `picamera2`, en
  `/usr/lib/python3/dist-packages/picamera2/examples/imx500/assets/coco_labels.txt`
  (dataset COCO, clases genéricas: persona, auto, perro, etc.).

Para comprobar qué modelos quedaron instalados:

```bash
ls /usr/share/imx500-models/
```

Si querés usar otro modelo (por ejemplo uno de clasificación en vez de detección):

1. Confirmá que el `.rpk` está en `/usr/share/imx500-models/` (si no, instalalo con
   el paquete apt correspondiente, ej. `imx500-models-classification` si existe para tu versión).
2. Cambiá la constante `MODEL` al inicio del script que quieras usar.
3. Si el modelo nuevo no es de detección de objetos COCO, actualizá también
   `LABELS_FILE` con el archivo de etiquetas que corresponda a ese modelo.

## Estructura de carpetas

```
Arquitecture-RAG/
└── ai-camera/
    ├── hdmi_live.py
    ├── live_stream.py
    ├── snapshot_deteccion.py
    ├── reconocer_emocion.py
    ├── modelos/                          <- .onnx de reconocer_emocion.py, sí van en git (ver nota abajo)
    │   ├── face_detection_yunet_2023mar.onnx
    │   └── emotion-ferplus-8.onnx
    ├── requirements.txt
    ├── install.sh
    ├── README.md
    ├── TODO-emociones-imx500.txt
    └── output/              <- se crea sola al correr snapshot_deteccion.py
```

No hace falta crear ninguna carpeta `models/` para los `.rpk` de la AI Camera:
viven en `/usr/share/imx500-models/` a nivel de sistema (los instala apt, no
pip ni git). Los `.onnx` de `reconocer_emocion.py` son distintos -- no tienen
un paquete de sistema que los instale, así que esos SÍ están versionados acá
en `modelos/` (~35MB en total).

## Uso

```bash
# Preview en vivo por HDMI (Ctrl+C para salir)
python3 hdmi_live.py

# Stream MJPEG por red — abrir http://<ip-raspberry>:5001/ en el navegador
python3 live_stream.py

# Foto única con detecciones dibujadas -> output/imx500_snapshot.jpg
python3 snapshot_deteccion.py
```

## Reconocimiento de emoción facial (CPU, no el sensor)

`reconocer_emocion.py` es para "Juego de emociones"/"Juego de imitación" del
dataset de trivia (ver `TODO-emociones-imx500.txt` para el contexto completo
de por qué se eligió este camino). A diferencia de los otros 3 scripts, NO
usa el modo NPU del IMX500 -- se investigó el model zoo oficial de Sony/
Raspberry Pi (`github.com/raspberrypi/imx500-models`) y no hay ningún modelo
de emociones publicado ahí; entrenar uno propio y convertirlo al formato
`.rpk` necesita el "IMX500 Converter" de Sony, que corre en Linux x86_64, no
en esta Raspberry Pi (ARM).

En cambio usa la cámara como cámara normal (una foto con `picamera2`) y
corre la inferencia en la CPU con dos modelos ONNX chicos y **ya
entrenados** (nada que entrenar en este repo), vía `onnxruntime` -- mucho
más liviano que la alternativa de TensorFlow/Keras que se había descartado
antes:

| Modelo | Peso | Qué hace |
|---|---|---|
| `modelos/face_detection_yunet_2023mar.onnx` | ~230KB | Detecta la cara en la foto (YuNet, OpenCV Zoo). Reemplaza a `cv2.CascadeClassifier` (Haar), que OpenCV 5.x sacó del build por default. |
| `modelos/emotion-ferplus-8.onnx` | ~35MB | Clasifica la emoción de la cara recortada (64×64, escala de grises) en 8 clases -- modelo de referencia público, entrenado sobre el dataset FER+ (ONNX Model Zoo). |

Solo 3 de las 8 clases importan para el dataset de trivia (columna `cara`:
Feliz/Triste/Enojado) -- `ETIQUETAS_RELEVANTES` en el script mapea
`felicidad`→`Feliz`, `tristeza`→`Triste`, `enojo`→`Enojado`; las otras 5
(neutral, sorpresa, asco, miedo, desprecio) se detectan pero no tienen
equivalente en el dataset.

```bash
# Con la cámara conectada: toma una foto y detecta la emoción
python3 reconocer_emocion.py

# Sin cámara, para probar contra una foto existente
python3 reconocer_emocion.py ruta/a/una/foto.jpg
```

⚠️ **Probado contra fotos de archivo, todavía NO contra la cámara real de
esta Raspberry Pi** (no había una conectada donde se escribió esto) --
correr el modo con cámara y confirmar que detecta bien caras reales, con
la luz e iluminación reales del lugar donde va a estar el robot, antes de
confiar en el resultado.

⚠️ **Todavía NO está conectado al flujo de Trivia** (`Orchestrator_
Management.py`) -- hoy "Juego de emociones"/"Juego de imitación" siguen
yendo por `reaccionar_libre()` sin veredicto real (no tienen
`respuesta_esperada` en el dataset). Conectarlo (que el robot pida una
emoción, la cámara la verifique, y eso decida acierto/error como hace
`Agent_Corrector` con las respuestas de texto) es diseño nuevo, no solo
enchufar el modelo -- ver la sección "DÓNDE ENGANCHARÍA EN EL CÓDIGO" de
`TODO-emociones-imx500.txt` antes de encararlo.

## Notas

- Los 3 scripts comparten el mismo umbral de confianza (`THRESHOLD = 0.45`),
  ajustable al inicio de cada archivo.
- `snapshot_deteccion.py` guarda siempre en `ai-camera/output/imx500_snapshot.jpg`
  (ruta relativa al propio script, se crea la carpeta si no existe). Antes apuntaba
  a una ruta temporal de una sesión de trabajo anterior que ya no existía — corregido.
