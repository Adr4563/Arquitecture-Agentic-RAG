# ai-camera

Scripts de detección de objetos en tiempo real usando la **Raspberry Pi AI Camera**
(sensor Sony IMX500 con NPU integrada) a través de `picamera2`. La inferencia corre
dentro del propio sensor de la cámara, no en la CPU de la Pi.

## Archivos

| Archivo | Qué hace |
|---|---|
| `hdmi_live.py` | Detección en vivo con preview directo por HDMI (DRM), sin necesidad de escritorio/X11. Pensado para un monitor conectado directo a la Pi. |
| `live_stream.py` | Servidor HTTP (puerto `5001`) que sirve un stream MJPEG con las detecciones dibujadas, para verlo desde otra máquina de la red. |
| `snapshot_deteccion.py` | Toma una sola foto (hasta 15 intentos buscando una detección), dibuja las cajas y la guarda en `output/`. |

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
    ├── requirements.txt
    ├── install.sh
    ├── README.md
    └── output/              <- se crea sola al correr snapshot_deteccion.py
```

No hace falta crear ninguna carpeta `models/` dentro del repo: los `.rpk` de la
AI Camera viven en `/usr/share/imx500-models/` a nivel de sistema (los instala apt,
no pip ni git).

## Uso

```bash
# Preview en vivo por HDMI (Ctrl+C para salir)
python3 hdmi_live.py

# Stream MJPEG por red — abrir http://<ip-raspberry>:5001/ en el navegador
python3 live_stream.py

# Foto única con detecciones dibujadas -> output/imx500_snapshot.jpg
python3 snapshot_deteccion.py
```

## Notas

- Los 3 scripts comparten el mismo umbral de confianza (`THRESHOLD = 0.45`),
  ajustable al inicio de cada archivo.
- `snapshot_deteccion.py` guarda siempre en `ai-camera/output/imx500_snapshot.jpg`
  (ruta relativa al propio script, se crea la carpeta si no existe). Antes apuntaba
  a una ruta temporal de una sesión de trabajo anterior que ya no existía — corregido.
