import os
import time
import sys
import cv2
import numpy as np

from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
LABELS_FILE = "/usr/lib/python3/dist-packages/picamera2/examples/imx500/assets/coco_labels.txt"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
OUT_IMG = os.path.join(OUT_DIR, "imx500_snapshot.jpg")

THRESHOLD = 0.45
IOU = 0.65
MAX_DET = 10

imx500 = IMX500(MODEL)
intrinsics = imx500.network_intrinsics
if not intrinsics:
    intrinsics = NetworkIntrinsics()
    intrinsics.task = "object detection"

if intrinsics.labels is None:
    try:
        with open(LABELS_FILE) as f:
            intrinsics.labels = f.read().splitlines()
    except FileNotFoundError:
        intrinsics.labels = [str(i) for i in range(91)]
intrinsics.update_with_defaults()

picam2 = Picamera2(imx500.camera_num)
config = picam2.create_preview_configuration(controls={"FrameRate": intrinsics.inference_rate}, buffer_count=12)
picam2.start(config, show_preview=False)

labels = intrinsics.labels
bbox_normalization = intrinsics.bbox_normalization
bbox_order = intrinsics.bbox_order

# give the network firmware a moment to load + let AE/AWB settle
time.sleep(3)

detections = []
frame = None
for i in range(15):
    metadata = picam2.capture_metadata()
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    if np_outputs is not None:
        input_w, input_h = imx500.get_input_size()
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h
        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        dets = [
            (imx500.convert_inference_coords(box, metadata, picam2), labels[int(c)], float(s))
            for box, s, c in zip(boxes, scores, classes) if s > THRESHOLD
        ]
        if dets:
            detections = dets
            frame = picam2.capture_array("main")
            break
    time.sleep(0.2)

if frame is None:
    frame = picam2.capture_array("main")

picam2.stop()

print("Detecciones:", len(detections))
for box, label, conf in detections:
    x, y, w, h = box
    print(f"  - {label}: {conf:.2f} @ ({x},{y},{w},{h})")

img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
for box, label, conf in detections:
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"{label} {conf:.2f}", (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

os.makedirs(OUT_DIR, exist_ok=True)
cv2.imwrite(OUT_IMG, img)
print("Snapshot guardado en:", OUT_IMG)
