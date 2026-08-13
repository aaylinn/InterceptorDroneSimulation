#!/usr/bin/env python3

import json
import struct
import time
import urllib.request

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


RGB_TOPIC = (
    "/world/interceptor_world/model/x500_depth_0/"
    "link/camera_link/sensor/IMX214/image"
)

DEPTH_TOPIC = "/depth_camera"
TARGET_TOPIC = "/buoy/target"

JETSON_URL = "http://192.168.55.1:5000/detect"

JPEG_QUALITY = 75
MAX_DEPTH_AGE_S = 0.30
MAX_DEPTH_M = 65.0


class JetsonBuoyDepthClient(Node):

    def __init__(self):
        super().__init__("jetson_buoy_depth_client")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()

        self.latest_depth = None
        self.latest_depth_time = 0.0

        self.busy = False
        self.frame_count = 0
        self.last_log_time = 0.0

        self.publisher = self.create_publisher(
            Float32MultiArray,
            TARGET_TOPIC,
            10,
        )

        self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            qos,
        )

        self.create_subscription(
            Image,
            RGB_TOPIC,
            self.rgb_callback,
            qos,
        )

        cv2.namedWindow(
            "Jetson RGB-D Buoy Detection",
            cv2.WINDOW_NORMAL,
        )

        cv2.resizeWindow(
            "Jetson RGB-D Buoy Detection",
            960,
            540,
        )

        cv2.startWindowThread()

        self.get_logger().info(
            "Jetson RGB-D hedef istemcisi hazır"
        )
        self.get_logger().info(
            f"Jetson adresi: {JETSON_URL}"
        )
        self.get_logger().info(
            f"RGB topic: {RGB_TOPIC}"
        )
        self.get_logger().info(
            f"Depth topic: {DEPTH_TOPIC}"
        )

    def publish_empty(self):
        message = Float32MultiArray()
        message.data = [0.0] * 10
        self.publisher.publish(message)

    def depth_callback(self, message):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="passthrough",
            )

            if depth.ndim != 2:
                return

            self.latest_depth = np.array(
                depth,
                dtype=np.float32,
                copy=True,
            )

            self.latest_depth_time = time.monotonic()

        except Exception as error:
            self.get_logger().warning(
                f"Depth dönüşüm hatası: {error}",
                throttle_duration_sec=2.0,
            )

    @staticmethod
    def encode_depth(depth):
        depth_mm = np.zeros(
            depth.shape,
            dtype=np.uint16,
        )

        valid = (
            np.isfinite(depth)
            & (depth > 0.0)
            & (depth <= MAX_DEPTH_M)
        )

        depth_mm[valid] = np.clip(
            depth[valid] * 1000.0,
            1.0,
            65535.0,
        ).astype(np.uint16)

        success, encoded = cv2.imencode(
            ".png",
            depth_mm,
        )

        if not success:
            raise RuntimeError(
                "Depth PNG kodlama başarısız"
            )

        return encoded.tobytes()

    def rgb_callback(self, message):
        if self.busy:
            return

        now = time.monotonic()

        if self.latest_depth is None:
            self.publish_empty()
            return

        depth_age = now - self.latest_depth_time

        if depth_age > MAX_DEPTH_AGE_S:
            self.publish_empty()

            self.get_logger().warning(
                f"Depth görüntüsü eski: {depth_age:.3f}s",
                throttle_duration_sec=2.0,
            )
            return

        self.busy = True

        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )

            depth = self.latest_depth.copy()

            if depth.shape[:2] != image.shape[:2]:
                depth = cv2.resize(
                    depth,
                    (
                        image.shape[1],
                        image.shape[0],
                    ),
                    interpolation=cv2.INTER_NEAREST,
                )

            rgb_success, rgb_encoded = cv2.imencode(
                ".jpg",
                image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    JPEG_QUALITY,
                ],
            )

            if not rgb_success:
                raise RuntimeError(
                    "RGB JPEG kodlama başarısız"
                )

            rgb_data = rgb_encoded.tobytes()
            depth_data = self.encode_depth(depth)

            header = struct.pack(
                "!II",
                len(rgb_data),
                len(depth_data),
            )

            request_body = (
                header
                + rgb_data
                + depth_data
            )

            request = urllib.request.Request(
                JETSON_URL,
                data=request_body,
                headers={
                    "Content-Type":
                    "application/x-rgb-depth"
                },
                method="POST",
            )

            start = time.perf_counter()

            with urllib.request.urlopen(
                request,
                timeout=2.0,
            ) as response:
                result = json.loads(
                    response.read().decode("utf-8")
                )

            round_trip_ms = (
                time.perf_counter() - start
            ) * 1000.0

            target = Float32MultiArray()

            target.data = [
                float(result.get("visible", 0.0)),
                float(result.get("center_x_norm", 0.0)),
                float(result.get("center_y_norm", 0.0)),
                float(result.get("width_ratio", 0.0)),
                float(result.get("height_ratio", 0.0)),
                float(result.get("area_ratio", 0.0)),
                float(result.get("confidence", 0.0)),
                float(result.get("source_code", 0.0)),
                float(result.get("depth_m", 0.0)),
                float(
                    result.get(
                        "depth_valid_ratio",
                        0.0,
                    )
                ),
            ]

            self.publisher.publish(target)

            display = image.copy()
            image_height, image_width = (
                display.shape[:2]
            )

            if target.data[0] >= 0.5:
                center_x = int(
                    target.data[1] * image_width
                )
                center_y = int(
                    target.data[2] * image_height
                )
                box_width = int(
                    target.data[3] * image_width
                )
                box_height = int(
                    target.data[4] * image_height
                )

                x1 = center_x - box_width // 2
                y1 = center_y - box_height // 2
                x2 = center_x + box_width // 2
                y2 = center_y + box_height // 2

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    3,
                )

                depth_m = target.data[8]
                valid_ratio = target.data[9]

                if depth_m > 0.0:
                    label = (
                        f"Red {target.data[6]:.2f} "
                        f"{depth_m:.2f} m"
                    )
                else:
                    label = (
                        f"Red {target.data[6]:.2f} "
                        "depth yok"
                    )

                cv2.putText(
                    display,
                    label,
                    (
                        max(0, x1),
                        max(25, y1 - 10),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    display,
                    f"Depth valid: {valid_ratio:.2f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 0),
                    2,
                )

            else:
                cv2.putText(
                    display,
                    "KIRMIZI HEDEF YOK",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            cv2.drawMarker(
                display,
                (
                    image_width // 2,
                    image_height // 2,
                ),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                30,
                2,
            )

            cv2.imshow(
                "Jetson RGB-D Buoy Detection",
                display,
            )

            cv2.waitKey(10)

            self.frame_count += 1

            log_now = time.monotonic()

            if log_now - self.last_log_time >= 1.0:
                self.get_logger().info(
                    f"frame={self.frame_count} "
                    f"visible={target.data[0]:.0f} "
                    f"cx={target.data[1]:.3f} "
                    f"cy={target.data[2]:.3f} "
                    f"conf={target.data[6]:.3f} "
                    f"depth={target.data[8]:.3f}m "
                    f"valid={target.data[9]:.2f} "
                    f"Jetson="
                    f"{result.get('inference_ms', 0):.1f}ms "
                    f"toplam={round_trip_ms:.1f}ms"
                )

                self.last_log_time = log_now

        except Exception as error:
            self.publish_empty()

            self.get_logger().warning(
                f"Jetson RGB-D hatası: {error}",
                throttle_duration_sec=2.0,
            )

        finally:
            self.busy = False


def main():
    rclpy.init()
    node = JetsonBuoyDepthClient()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
