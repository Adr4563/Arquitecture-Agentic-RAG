import io
import time
import threading
import socketserver
from http import server

import cv2

from picamera2 import Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
LABELS_FILE = "/usr/lib/python3/dist-packages/picamera2/examples/imx500/assets/coco_labels.txt"
THRESHOLD = 0.45
PORT = 5001

PAGE = """\
<html>
<head><title>IMX500 - deteccion en vivo</title></head>
<body style="margin:0;background:#111;">
<img src="stream.mjpg" style="width:100%;height:auto;display:block;" />
</body>
</html>
"""


class StreamingOutput:
    def __init__(self):
        self.frame = None
        self.condition = threading.Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


output = StreamingOutput()


class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            content = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass
        else:
            self.send_error(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
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
    config = picam2.create_preview_configuration(
        main={"size": (640, 480)},
        controls={"FrameRate": intrinsics.inference_rate or 15},
        buffer_count=8,
    )
    picam2.start(config, show_preview=False)
    time.sleep(2)

    labels = intrinsics.labels
    bbox_normalization = intrinsics.bbox_normalization
    bbox_order = intrinsics.bbox_order
    last_dets = []

    server_thread = None

    def capture_loop():
        nonlocal last_dets
        while True:
            metadata = picam2.capture_metadata()
            np_outputs = imx500.get_outputs(metadata, add_batch=True)
            if np_outputs is not None:
                input_w, input_h = imx500.get_input_size()
                boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
                if bbox_normalization:
                    boxes = boxes / input_h
                if bbox_order == "xy":
                    boxes = boxes[:, [1, 0, 3, 2]]
                dets = []
                for box, score, cat in zip(boxes, scores, classes):
                    if score > THRESHOLD:
                        coords = imx500.convert_inference_coords(box, metadata, picam2)
                        dets.append((coords, labels[int(cat)], float(score)))
                last_dets = dets

            frame = picam2.capture_array("main")
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.shape[-1] == 3 else cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            for coords, label, score in last_dets:
                x, y, w, h = coords
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {score:.2f}", (x, max(y - 5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                output.write(jpg.tobytes())

    t = threading.Thread(target=capture_loop, daemon=True)
    t.start()

    addr = ("0.0.0.0", PORT)
    srv = StreamingServer(addr, StreamingHandler)
    print(f"READY on port {PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
