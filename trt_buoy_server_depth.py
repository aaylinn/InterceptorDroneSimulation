import json
import struct
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from infer_trt import load_engine, preprocess, infer, postprocess


HOST = "0.0.0.0"
PORT = 5000

MAX_REQUEST_SIZE = 10 * 1024 * 1024
HEADER_SIZE = 8

MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 50.0


print("TensorRT motoru yükleniyor...")
ENGINE = load_engine()
print("TensorRT motoru hazır.")


def calculate_target_depth(depth_image, x, y, width, height):
    image_height, image_width = depth_image.shape[:2]

    # Kutunun kenarlarında arka plan bulunabileceği için
    # yalnızca merkezdeki yüzde 50'lik alan kullanılır.
    x1 = int(x + width * 0.25)
    y1 = int(y + height * 0.25)
    x2 = int(x + width * 0.75)
    y2 = int(y + height * 0.75)

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))
    x2 = max(x1 + 1, min(image_width, x2))
    y2 = max(y1 + 1, min(image_height, y2))

    roi = depth_image[y1:y2, x1:x2]

    valid_mask = (
        np.isfinite(roi)
        & (roi >= MIN_DEPTH_M)
        & (roi <= MAX_DEPTH_M)
    )

    valid = roi[valid_mask]
    valid_ratio = (
        float(valid.size) / float(roi.size)
        if roi.size
        else 0.0
    )

    if valid.size < 5:
        return 0.0, valid_ratio

    # Tek piksel yerine medyan kullanmak gürültüye
    # ve hatalı derinlik değerlerine karşı dayanıklıdır.
    depth_m = float(np.median(valid))

    return depth_m, valid_ratio


class DetectionHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self):
        self.send_json({
            "status": "ready",
            "service": "Jetson TensorRT RGB-D buoy detector",
        })

    def do_POST(self):
        if self.path != "/detect":
            self.send_json({"error": "Geçersiz adres"}, 404)
            return

        try:
            content_length = int(
                self.headers.get("Content-Length", "0")
            )

            if (
                content_length <= HEADER_SIZE
                or content_length > MAX_REQUEST_SIZE
            ):
                self.send_json(
                    {"error": "Geçersiz RGB-D veri boyutu"},
                    400,
                )
                return

            body = self.rfile.read(content_length)

            rgb_size, depth_size = struct.unpack(
                "!II",
                body[:HEADER_SIZE],
            )

            expected_size = HEADER_SIZE + rgb_size + depth_size

            if expected_size != len(body):
                self.send_json(
                    {"error": "RGB-D paket boyutu uyuşmuyor"},
                    400,
                )
                return

            rgb_start = HEADER_SIZE
            rgb_end = rgb_start + rgb_size
            depth_end = rgb_end + depth_size

            rgb_data = body[rgb_start:rgb_end]
            depth_data = body[rgb_end:depth_end]

            image = cv2.imdecode(
                np.frombuffer(rgb_data, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )

            depth_mm = cv2.imdecode(
                np.frombuffer(depth_data, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:
                self.send_json(
                    {"error": "RGB görüntüsü okunamadı"},
                    400,
                )
                return

            if depth_mm is None or depth_mm.dtype != np.uint16:
                self.send_json(
                    {"error": "Depth görüntüsü okunamadı"},
                    400,
                )
                return

            original_height, original_width = image.shape[:2]

            if depth_mm.shape[:2] != image.shape[:2]:
                depth_mm = cv2.resize(
                    depth_mm,
                    (original_width, original_height),
                    interpolation=cv2.INTER_NEAREST,
                )

            depth_image = depth_mm.astype(np.float32) / 1000.0

            input_tensor, scale, pad_x, pad_y = preprocess(image)
            outputs, inference_ms = infer(ENGINE, input_tensor)

            detections = postprocess(
                outputs["output0"],
                image,
                scale,
                pad_x,
                pad_y,
            )

            if not detections:
                result = {
                    "visible": 0.0,
                    "center_x_norm": 0.0,
                    "center_y_norm": 0.0,
                    "width_ratio": 0.0,
                    "height_ratio": 0.0,
                    "area_ratio": 0.0,
                    "confidence": 0.0,
                    "source_code": 0.0,
                    "depth_m": 0.0,
                    "depth_valid_ratio": 0.0,
                    "inference_ms": round(inference_ms, 2),
                }

            else:
                x, y, width, height, confidence = max(
                    detections,
                    key=lambda detection: detection[4],
                )

                depth_m, depth_valid_ratio = (
                    calculate_target_depth(
                        depth_image,
                        x,
                        y,
                        width,
                        height,
                    )
                )

                result = {
                    "visible": 1.0,
                    "center_x_norm": (
                        x + width / 2
                    ) / original_width,
                    "center_y_norm": (
                        y + height / 2
                    ) / original_height,
                    "width_ratio": width / original_width,
                    "height_ratio": height / original_height,
                    "area_ratio": (
                        width * height
                    ) / (
                        original_width * original_height
                    ),
                    "confidence": float(confidence),
                    "source_code": 1.0,
                    "depth_m": depth_m,
                    "depth_valid_ratio": depth_valid_ratio,
                    "inference_ms": round(inference_ms, 2),
                }

            print(
                f"visible={result['visible']:.0f} "
                f"confidence={result['confidence']:.3f} "
                f"depth={result['depth_m']:.3f}m "
                f"valid={result['depth_valid_ratio']:.2f} "
                f"inference={result['inference_ms']:.2f}ms"
            )

            self.send_json(result)

        except Exception as error:
            print(f"Hata: {error}")
            self.send_json({"error": str(error)}, 500)

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = HTTPServer((HOST, PORT), DetectionHandler)

    print(f"Sunucu hazır: http://{HOST}:{PORT}")
    print("Kapatmak için Ctrl+C")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatılıyor.")
    finally:
        server.server_close()
