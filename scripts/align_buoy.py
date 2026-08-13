#!/usr/bin/env python3

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray


TARGET_TOPIC = "/buoy/target"

TARGET_TIMEOUT_S = 0.60
CONTROL_HZ = 10.0
MAX_TEST_TIME_S = 60.0

X_DEADBAND = 0.055
Y_DEADBAND = 0.070

YAW_GAIN = 60.0
VERTICAL_GAIN = 0.75

MAX_YAW_RATE = 15.0
MAX_VERTICAL_SPEED = 0.18

STABLE_FRAMES_REQUIRED = 20


@dataclass
class Target:
    visible: bool
    center_x: float
    center_y: float
    confidence: float
    source_code: int
    received_at: float


class TargetReceiver(Node):
    def __init__(self):
        super().__init__("buoy_align_controller")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.target: Optional[Target] = None

        self.create_subscription(
            Float32MultiArray,
            TARGET_TOPIC,
            self.target_callback,
            qos,
        )

    def target_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 8:
            return

        self.target = Target(
            visible=msg.data[0] >= 0.5,
            center_x=float(msg.data[1]),
            center_y=float(msg.data[2]),
            confidence=float(msg.data[6]),
            source_code=int(round(msg.data[7])),
            received_at=time.monotonic(),
        )


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


async def stop_motion(drone: System):
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(
            0.0,
            0.0,
            0.0,
            0.0,
        )
    )


async def wait_for_connection(drone: System):
    print("PX4 baglantisi bekleniyor...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 baglandi.")
            return


async def vehicle_is_in_air(drone: System) -> bool:
    async for in_air in drone.telemetry.in_air():
        return bool(in_air)

    return False


async def run():
    rclpy.init()

    node = TargetReceiver()
    drone = System()
    offboard_started = False

    try:
        await drone.connect(
            system_address="udpin://0.0.0.0:14540"
        )

        await wait_for_connection(drone)

        if not await vehicle_is_in_air(drone):
            print(
                "DRONE HAVADA DEGIL: "
                "QGroundControl ile 2 m takeoff yap."
            )
            return

        # Offboard başlamadan önce sıfır setpoint gönder.
        await stop_motion(drone)

        print("Offboard baslatiliyor...")

        try:
            await drone.offboard.start()
            offboard_started = True
        except OffboardError as error:
            print(
                "Offboard baslatilamadi: "
                f"{error._result.result}"
            )
            return

        print("HIZALAMA TESTI BASLADI.")
        print("Ileri hiz her zaman 0.00 m/s.")
        print(
            "Acil durdurma: Ctrl+C veya "
            "QGroundControl'dan Hold/Land"
        )

        start_time = time.monotonic()
        stable_frames = 0
        last_log_time = 0.0

        while (
            time.monotonic() - start_time
            < MAX_TEST_TIME_S
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.0,
            )

            now = time.monotonic()
            target = node.target

            target_is_fresh = (
                target is not None
                and target.visible
                and now - target.received_at
                <= TARGET_TIMEOUT_S
            )

            if not target_is_fresh:
                stable_frames = 0

                await stop_motion(drone)

                if now - last_log_time >= 1.0:
                    print(
                        "HEDEF YOK/ESKI -> "
                        "tum hizlar 0"
                    )
                    last_log_time = now

                await asyncio.sleep(
                    1.0 / CONTROL_HZ
                )
                continue

            error_x = target.center_x - 0.5
            error_y = target.center_y - 0.5

            yaw_rate = 0.0
            down_speed = 0.0

            if abs(error_x) > X_DEADBAND:
                yaw_rate = clamp(
                    error_x * YAW_GAIN,
                    MAX_YAW_RATE,
                )

            if abs(error_y) > Y_DEADBAND:
                down_speed = clamp(
                    error_y * VERTICAL_GAIN,
                    MAX_VERTICAL_SPEED,
                )

            aligned = (
                abs(error_x) <= X_DEADBAND
                and abs(error_y) <= Y_DEADBAND
            )

            if aligned:
                stable_frames += 1
            else:
                stable_frames = 0

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(
                    0.0,        # ileri
                    0.0,        # sağ
                    down_speed, # aşağı/yukarı
                    yaw_rate,   # dönüş
                )
            )

            if now - last_log_time >= 0.5:
                print(
                    f"src={target.source_code} "
                    f"conf={target.confidence:.2f} "
                    f"x={target.center_x:.3f} "
                    f"y={target.center_y:.3f} | "
                    f"ileri=0.00 "
                    f"asagi={down_speed:.2f} "
                    f"yaw={yaw_rate:.1f} "
                    f"stable="
                    f"{stable_frames}/"
                    f"{STABLE_FRAMES_REQUIRED}"
                )

                last_log_time = now

            if (
                stable_frames
                >= STABLE_FRAMES_REQUIRED
            ):
                await stop_motion(drone)

                print(
                    "HIZALAMA BASARILI: "
                    "hedef 2 saniye merkezde kaldi."
                )
                break

            await asyncio.sleep(
                1.0 / CONTROL_HZ
            )

        else:
            print(
                "60 saniyelik hizalama "
                "suresi doldu."
            )

    finally:
        if offboard_started:
            try:
                await stop_motion(drone)
                await asyncio.sleep(0.3)
                await drone.offboard.stop()

                print(
                    "Offboard durduruldu; "
                    "PX4 Hold moduna donmeli."
                )

            except Exception as error:
                print(
                    f"Durdurma uyarisi: {error}"
                )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nKullanici durdurdu.")
