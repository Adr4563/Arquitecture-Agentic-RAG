import sys
import time
from functools import lru_cache

import cv2

from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
LABELS_FILE = "/usr/lib/python3/dist-packages/picamera2/examples/imx500/assets/coco_labels.txt"
THRESHOLD = 0.45

last_detections = []


class Detection:
    def __init__(self, coords, category, conf, metadata):
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)


def parse_detections(metadata):
    global last_detections
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is None:
        return last_detections
    input_w, input_h = imx500.get_input_size()
    boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
    if intrinsics.bbox_normalization:
        boxes = boxes / input_h
    if intrinsics.bbox_order == "xy":
        boxes = boxes[:, [1, 0, 3, 2]]
    last_detections = [
        Detection(box, category, score, metadata)
        for box, score, category in zip(boxes, scores, classes) if score > THRESHOLD
    ]
    return last_detections


@lru_cache
def get_labels():
    return intrinsics.labels


def draw_detections(request, stream="main"):
    detections = last_results
    if not detections:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for det in detections:
            x, y, w, h = det.box
            label = f"{labels[int(det.category)]} ({det.conf:.2f})"
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
            cv2.putText(m.array, label, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


print("Cargando IMX500...", flush=True)
imx500 = IMX500(MODEL)
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"
if intrinsics.labels is None:
    with open(LABELS_FILE) as f:
        intrinsics.labels = f.read().splitlines()
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate or 15}, buffer_count=12)

print("Abriendo preview en HDMI (DRM)...", flush=True)
picam2.start(config, show_preview=True)
print("Preview activo. Corriendo deteccion en vivo...", flush=True)

last_results = None
picam2.pre_callback = draw_detections
try:
    while True:
        last_results = parse_detections(picam2.capture_metadata())
except KeyboardInterrupt:
    pass
