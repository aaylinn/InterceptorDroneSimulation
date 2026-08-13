#!/usr/bin/env python3

import math
import time

import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


RAW_TOPIC = "/buoy/target"
FILTERED_TOPIC = "/buoy/target_kf"

# OAK-D Lite RGB kamera özellikleri
IMAGE_WIDTH = 640.0
IMAGE_HEIGHT = 360.0
HORIZONTAL_FOV_RAD = 1.274

FX = IMAGE_WIDTH / (
    2.0 * math.tan(HORIZONTAL_FOV_RAD / 2.0)
)
FY = FX

MIN_CONFIDENCE = 0.25
MIN_DEPTH_VALID_RATIO = 0.05
MIN_DEPTH_M = 0.20
MAX_DEPTH_M = 30.0

# Ölçüm uzun süre kesilirse filtre sıfırlanır.
RESET_TIMEOUT_S = 1.50

# Ölçüm gürültüsü.
# Küçük değer kameraya daha fazla güvenmek demektir.
MEASUREMENT_NOISE = 0.12

# Hareket modeli belirsizliği.
# Büyük değer ani hız değişimlerine daha hızlı uyum sağlar.
PROCESS_NOISE = 1.20


class KalmanBuoyTracker(Node):

    def __init__(self):
        super().__init__("kalman_buoy_tracker")

        self.subscription = self.create_subscription(
            Float32MultiArray,
            RAW_TOPIC,
            self.target_callback,
            10,
        )

        self.publisher = self.create_publisher(
            Float32MultiArray,
            FILTERED_TOPIC,
            10,
        )

        # Durum:
        # [x, y, z, vx, vy, vz]
        #
        # x: sağ-sol konumu
        # y: aşağı-yukarı konumu
        # z: kameradan ileri mesafe
        self.state = np.zeros((9, 1), dtype=np.float64)

        self.covariance = np.eye(
            9,
            dtype=np.float64,
        ) * 10.0

        self.measurement_matrix = np.zeros(
            (3, 9),
            dtype=np.float64,
        )

        self.measurement_matrix[0, 0] = 1.0
        self.measurement_matrix[1, 1] = 1.0
        self.measurement_matrix[2, 2] = 1.0

        self.measurement_covariance = (
            np.eye(3, dtype=np.float64)
            * MEASUREMENT_NOISE
        )

        self.identity = np.eye(9, dtype=np.float64)

        self.initialized = False
        self.last_update_time = None
        self.last_measurement_time = None
        self.last_log_time = 0.0

        self.get_logger().info(
            "Kalman duba takip düğümü hazır"
        )

        self.get_logger().info(
            f"Giriş: {RAW_TOPIC}"
        )

        self.get_logger().info(
            f"Çıkış: {FILTERED_TOPIC}"
        )

    def build_transition_matrix(self, dt):
        transition = np.eye(9, dtype=np.float64)

        dt2 = dt * dt

        # x,y,z + vx,vy,vz + ax,ay,az
        for p_i, v_i, a_i in (
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
        ):
            transition[p_i, v_i] = dt
            transition[p_i, a_i] = 0.5 * dt2
            transition[v_i, a_i] = dt

        return transition

    def build_process_covariance(self, dt):
        q = np.zeros((9, 9), dtype=np.float64)

        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5
        dt6 = dt ** 6

        for p_i, v_i, a_i in (
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
        ):
            block = np.array([
                [dt6 / 36.0, dt5 / 12.0, dt4 / 6.0],
                [dt5 / 12.0, dt4 / 4.0, dt3 / 2.0],
                [dt4 / 6.0, dt3 / 2.0, dt2],
            ]) * PROCESS_NOISE

            idx = (p_i, v_i, a_i)

            for r in range(3):
                for c in range(3):
                    q[idx[r], idx[c]] = block[r, c]

        return q

    def predict(self, dt):
        transition = self.build_transition_matrix(dt)
        process_covariance = (
            self.build_process_covariance(dt)
        )

        self.state = transition @ self.state

        self.covariance = (
            transition
            @ self.covariance
            @ transition.T
            + process_covariance
        )

    def update(self, measurement):
        innovation = (
            measurement
            - self.measurement_matrix @ self.state
        )

        predicted_depth = float(self.state[2, 0])
        measured_depth = float(measurement[2, 0])

        gap = (
            time.monotonic() - self.last_measurement_time
            if self.last_measurement_time is not None
            else 0.0
        )

        # Fiziksel olarak mantıksız depth sıçramasını ele.
        max_depth_jump = max(
            1.5,
            0.25 * abs(predicted_depth),
            12.0 * gap + 0.5,
        )

        if (
            predicted_depth > 0.2
            and abs(measured_depth - predicted_depth)
            > max_depth_jump
        ):
            self.get_logger().warning(
                "OUTLIER_DEPTH | "
                f"tahmin={predicted_depth:.2f}m "
                f"olcum={measured_depth:.2f}m "
                "-> REDDEDILDI"
            )
            return False

        innovation_covariance = (
            self.measurement_matrix
            @ self.covariance
            @ self.measurement_matrix.T
            + self.measurement_covariance
        )

        inverse_s = np.linalg.inv(
            innovation_covariance
        )

        nis = float(
            innovation.T
            @ inverse_s
            @ innovation
        )

        # Mahalanobis/innovation kapısı
        if nis > 25.0:
            self.get_logger().warning(
                f"OUTLIER_NIS | {nis:.2f} "
                "-> REDDEDILDI"
            )
            return False

        kalman_gain = (
            self.covariance
            @ self.measurement_matrix.T
            @ inverse_s
        )

        self.state = (
            self.state
            + kalman_gain @ innovation
        )

        self.covariance = (
            self.identity
            - kalman_gain
            @ self.measurement_matrix
        ) @ self.covariance

        return True

    def target_is_valid(self, data):
        return (
            len(data) >= 10
            and data[0] >= 0.5
            and data[6] >= MIN_CONFIDENCE
            and MIN_DEPTH_M <= data[8] <= MAX_DEPTH_M
            and data[9] >= MIN_DEPTH_VALID_RATIO
        )

    def make_measurement(self, data):
        center_x_norm = float(data[1])
        center_y_norm = float(data[2])
        depth_m = float(data[8])

        pixel_x = center_x_norm * IMAGE_WIDTH
        pixel_y = center_y_norm * IMAGE_HEIGHT

        lateral_x = (
            (pixel_x - IMAGE_WIDTH / 2.0)
            * depth_m
            / FX
        )

        vertical_y = (
            (pixel_y - IMAGE_HEIGHT / 2.0)
            * depth_m
            / FY
        )

        forward_z = depth_m

        return np.array(
            [
                [lateral_x],
                [vertical_y],
                [forward_z],
            ],
            dtype=np.float64,
        )

    def initialize_filter(self, measurement, now):
        self.state[:] = 0.0
        self.state[0:3] = measurement

        self.covariance = np.eye(
            9,
            dtype=np.float64,
        )

        self.covariance[0:3, 0:3] *= 0.20
        self.covariance[3:6, 3:6] *= 5.00
        self.covariance[6:9, 6:9] *= 8.00

        self.initialized = True
        self.last_update_time = now
        self.last_measurement_time = now

        self.get_logger().info(
            "Kalman filtresi ilk ölçümle başlatıldı"
        )

    def publish_state(
        self,
        measurement_available,
        confidence,
        depth_valid_ratio,
        now,
    ):
        message = Float32MultiArray()

        time_since_measurement = (
            now - self.last_measurement_time
            if self.last_measurement_time is not None
            else 999.0
        )

        position_uncertainty = math.sqrt(
            max(
                0.0,
                float(
                    self.covariance[0, 0]
                    + self.covariance[1, 1]
                    + self.covariance[2, 2]
                ),
            )
        )

        message.data = [
            1.0 if self.initialized else 0.0,
            1.0 if measurement_available else 0.0,
            float(self.state[0, 0]),
            float(self.state[1, 0]),
            float(self.state[2, 0]),
            float(self.state[3, 0]),
            float(self.state[4, 0]),
            float(self.state[5, 0]),
            float(confidence),
            float(depth_valid_ratio),
            float(time_since_measurement),
            float(position_uncertainty),
            float(self.state[6, 0]),
            float(self.state[7, 0]),
            float(self.state[8, 0]),
        ]

        self.publisher.publish(message)

        if now - self.last_log_time >= 0.25:
            source = (
                "ÖLÇÜM"
                if measurement_available
                else "TAHMİN"
            )

            self.get_logger().info(
                f"{source} | "
                f"konum=({self.state[0, 0]:+.2f}, "
                f"{self.state[1, 0]:+.2f}, "
                f"{self.state[2, 0]:.2f}) m | "
                f"hız=({self.state[3, 0]:+.2f}, "
                f"{self.state[4, 0]:+.2f}, "
                f"{self.state[5, 0]:+.2f}) m/s | "
                f"kayıp={time_since_measurement:.2f}s "
                f"belirsizlik={position_uncertainty:.2f}"
            )

            self.last_log_time = now

    def target_callback(self, message):
        now = time.monotonic()
        data = list(message.data)

        measurement_available = (
            self.target_is_valid(data)
        )

        confidence = (
            float(data[6])
            if len(data) > 6
            else 0.0
        )

        depth_valid_ratio = (
            float(data[9])
            if len(data) > 9
            else 0.0
        )

        if (
            self.initialized
            and self.last_update_time is not None
        ):
            dt = max(
                0.001,
                min(
                    0.20,
                    now - self.last_update_time,
                ),
            )

            self.predict(dt)
            self.last_update_time = now

        if measurement_available:
            measurement = self.make_measurement(data)

            if not self.initialized:
                self.initialize_filter(
                    measurement,
                    now,
                )
            else:
                measurement_available = self.update(
                    measurement
                )

                if measurement_available:
                    self.last_measurement_time = now

        elif (
            self.initialized
            and self.last_measurement_time is not None
            and now - self.last_measurement_time
            > RESET_TIMEOUT_S
        ):
            self.get_logger().warning(
                "Hedef uzun süre kayıp; "
                "Kalman filtresi sıfırlandı"
            )

            self.initialized = False
            self.last_update_time = None
            self.last_measurement_time = None
            self.state[:] = 0.0

        self.publish_state(
            measurement_available,
            confidence,
            depth_valid_ratio,
            now,
        )


def main():
    rclpy.init()
    node = KalmanBuoyTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nKalman takip durduruldu.")
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
