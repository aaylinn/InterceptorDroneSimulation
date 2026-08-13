#!/usr/bin/env python3

import math
import time

import rclpy

from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


RAW_TOPIC = "/buoy/target"
KALMAN_TOPIC = "/buoy/target_kf"
CONTROL_TOPIC = "/buoy/target_control"

IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 360.0
HORIZONTAL_FOV_RAD = 1.274

FX = IMAGE_WIDTH / (
    2.0 * math.tan(HORIZONTAL_FOV_RAD / 2.0)
)
FY = FX

MAX_PREDICTION_AGE_S = 0.40

# Hedefin gelecekteki konumuna yönelme süresi.
LEAD_TIME_S = 0.30
MAX_UNCERTAINTY_M = 1.50
MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 30.0


class KalmanControlAdapter(Node):

    def __init__(self):
        super().__init__(
            "kalman_control_adapter"
        )

        self.publisher = self.create_publisher(
            Float32MultiArray,
            CONTROL_TOPIC,
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            RAW_TOPIC,
            self.raw_callback,
            10,
        )

        self.create_subscription(
            Float32MultiArray,
            KALMAN_TOPIC,
            self.kalman_callback,
            10,
        )

        self.last_raw_visible = None
        self.last_raw_received_at = None
        self.last_log_time = 0.0

        self.get_logger().info(
            "Kalman kontrol adaptörü hazır"
        )
        self.get_logger().info(
            f"Çıkış: {CONTROL_TOPIC}"
        )

    def raw_callback(self, message):
        data = list(message.data)

        if len(data) < 10:
            return

        if data[0] >= 0.5:
            self.last_raw_visible = data[:10]
            self.last_raw_received_at = (
                time.monotonic()
            )

    def publish_empty(self):
        message = Float32MultiArray()
        message.data = [0.0] * 10
        self.publisher.publish(message)

    def kalman_callback(self, message):
        data = list(message.data)

        if len(data) < 15:
            self.publish_empty()
            return

        filter_active = data[0] >= 0.5
        real_measurement = data[1] >= 0.5

        lateral_x = float(data[2])
        vertical_y = float(data[3])
        depth_m = float(data[4])

        velocity_x = float(data[5])
        velocity_y = float(data[6])

        acceleration_x = float(data[12])
        acceleration_y = float(data[13])

        measurement_age = float(data[10])
        uncertainty = float(data[11])

        valid = (
            filter_active
            and measurement_age
            <= MAX_PREDICTION_AGE_S
            and uncertainty
            <= MAX_UNCERTAINTY_M
            and MIN_DEPTH_M
            <= depth_m
            <= MAX_DEPTH_M
        )

        if not valid:
            self.publish_empty()
            return

        # Constant-Acceleration modeli ile hedefin
        # LEAD_TIME_S saniye sonraki yatay/dikey
        # konumunu hesapla.
        lead_t = LEAD_TIME_S

        lateral_x = (
            lateral_x
            + velocity_x * lead_t
            + 0.5 * acceleration_x
            * lead_t * lead_t
        )

        vertical_y = (
            vertical_y
            + velocity_y * lead_t
            + 0.5 * acceleration_y
            * lead_t * lead_t
        )

        center_x = (
            0.5
            + lateral_x * FX
            / (depth_m * IMAGE_WIDTH)
        )

        center_y = (
            0.5
            + vertical_y * FY
            / (depth_m * IMAGE_HEIGHT)
        )

        center_x = max(
            0.0,
            min(1.0, center_x),
        )
        center_y = max(
            0.0,
            min(1.0, center_y),
        )

        if self.last_raw_visible is not None:
            width_ratio = float(
                self.last_raw_visible[3]
            )
            height_ratio = float(
                self.last_raw_visible[4]
            )
            area_ratio = float(
                self.last_raw_visible[5]
            )
            confidence = float(
                self.last_raw_visible[6]
            )
            depth_valid_ratio = float(
                self.last_raw_visible[9]
            )
        else:
            width_ratio = 0.0
            height_ratio = 0.0
            area_ratio = 0.0
            confidence = 0.30
            depth_valid_ratio = 0.10

        if not real_measurement:
            # Kısa Kalman tahmini sırasında kontrolcünün
            # veriyi hemen reddetmesini önler.
            confidence = max(
                confidence,
                0.30,
            )
            depth_valid_ratio = max(
                depth_valid_ratio,
                0.10,
            )

        output = Float32MultiArray()
        output.data = [
            1.0,
            center_x,
            center_y,
            width_ratio,
            height_ratio,
            area_ratio,
            confidence,
            6.0,
            depth_m,
            depth_valid_ratio,
        ]

        self.publisher.publish(output)

        now = time.monotonic()

        if now - self.last_log_time >= 0.50:
            mode = (
                "KF_ÖLÇÜM"
                if real_measurement
                else "KF_TAHMİN"
            )

            self.get_logger().info(
                f"{mode} | "
                f"cx={center_x:.3f} "
                f"cy={center_y:.3f} "
                f"depth={depth_m:.2f}m "
                f"age={measurement_age:.2f}s "
                f"unc={uncertainty:.2f}"
            )

            self.last_log_time = now


def main():
    rclpy.init()
    node = KalmanControlAdapter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(
            "\nKalman kontrol adaptörü durduruldu."
        )
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
