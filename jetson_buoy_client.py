#!/usr/bin/env python3

import json
import time
import urllib.request

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


CAMERA_TOPIC = (
    "/world/interceptor_world/model/x500_depth_0/"
    "link/camera_link/sensor/IMX214/image"
)

TARGET_TOPIC = "/buoy/target"
JETSON_URL = "http://192.168.55.1:5000/detect"
JPEG_QUALITY = 75


class JetsonBuoyClient(Node):

    def __init__(self):
        super().__init__("jetson_buoy_client")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.bridge = CvBridge()

        cv2.namedWindow(
            "Jetson Buoy Detection",
            cv2.WINDOW_NORMAL,
        )
        cv2.resizeWindow(
            "Jetson Buoy Detection",
            960,
            540,
        )
        cv2.startWindowThread()

        self.publisher = self.create_publisher(
            Float32MultiArray,
            TARGET_TOPIC,
            10,
        )

        self.subscription = self.create_subscription(
            Image,
            CAMERA_TOPIC,
            self.image_callback,
            qos,
        )

        self.busy = False
        self.frame_count = 0
        self.last_log_time = 0.0

        self.get_logger().info("Jetson hedef istemcisi hazır")
        self.get_logger().info(f"Jetson adresi: {JETSON_URL}")

    def publish_empty(self):
        message = Float32MultiArray()
        message.data = [0.0] * 8
        self.publisher.publish(message)

    def image_callback(self, message):
        if self.busy:
            return

        self.busy = True

        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )

            success, encoded = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            if not success:
                raise RuntimeError("JPEG kodlama başarısız")

            request = urllib.request.Request(
                JETSON_URL,
                data=encoded.tobytes(),
                headers={"Content-Type": "image/jpeg"},
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
            ]

            self.publisher.publish(target)
            display = image.copy()
            image_height, image_width = display.shape[:2]

            if target.data[0] >= 0.5:
                center_x = int(target.data[1] * image_width)
                center_y = int(target.data[2] * image_height)
                box_width = int(target.data[3] * image_width)
                box_height = int(target.data[4] * image_height)

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

                cv2.putText(
                    display,
                    f"Red {target.data[6]:.2f}",
                    (max(0, x1), max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                cv2.putText(
                    display,
                    "HEDEF YOK",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            cv2.drawMarker(
                display,
                (image_width // 2, image_height // 2),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                30,
                2,
            )

            cv2.imshow("Jetson Buoy Detection", display)
            cv2.waitKey(10)
            self.frame_count += 1

            now = time.monotonic()
            if now - self.last_log_time >= 1.0:
                self.get_logger().info(
                    f"frame={self.frame_count} "
                    f"visible={target.data[0]:.0f} "
                    f"cx={target.data[1]:.3f} "
                    f"cy={target.data[2]:.3f} "
                    f"conf={target.data[6]:.3f} "
                    f"Jetson={result.get('inference_ms', 0):.1f}ms "
                    f"toplam={round_trip_ms:.1f}ms"
                )
                self.last_log_time = now

        except Exception as error:
            self.publish_empty()
            self.get_logger().warning(
                f"Jetson bağlantı/tespit hatası: {error}",
                throttle_duration_sec=2.0,
            )

        finally:
            self.busy = False


def main():
    rclpy.init()
    node = JetsonBuoyClient()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
