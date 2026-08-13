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

CONTROL_HZ = 20.0
SEARCH_YAW_RATE = 25.0
SEARCH_TIMEOUT_S = 20.0

TARGET_TIMEOUT_S = 0.60
MIN_CONFIDENCE = 0.35
CONFIRM_FRAMES = 3


@dataclass
class Target:
    visible: bool
    center_x: float
    center_y: float
    confidence: float
    source_code: int
    received_at: float
    frame_id: int


class TargetReceiver(Node):

    def __init__(self):
        super().__init__("buoy_search_controller")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.target: Optional[Target] = None
        self.frame_counter = 0

        self.create_subscription(
            Float32MultiArray,
            TARGET_TOPIC,
            self.target_callback,
            qos,
        )

    def target_callback(self, msg):
        if len(msg.data) < 8:
            return

        self.frame_counter += 1

        self.target = Target(
            visible=msg.data[0] >= 0.5,
            center_x=float(msg.data[1]),
            center_y=float(msg.data[2]),
            confidence=float(msg.data[6]),
            source_code=int(round(msg.data[7])),
            received_at=time.monotonic(),
            frame_id=self.frame_counter,
        )


async def send_motion(drone, yaw_rate):
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(
            0.0,
            0.0,
            0.0,
            yaw_rate,
        )
    )


async def wait_for_connection(drone):
    print("PX4 baglantisi bekleniyor...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 baglandi.")
            return


async def vehicle_is_in_air(drone):
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
                "QGroundControl ile Takeoff yap ve Hold'a al."
            )
            return

        # Offboard oncesi ilk setpoint.
        await send_motion(drone, 0.0)
        await asyncio.sleep(0.2)

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

        print("")
        print("TARAMA TESTI BASLADI")
        print(f"Yaw tarama hizi: {SEARCH_YAW_RATE:.1f} deg/s")
        print(f"Gerekli ardisik tespit: {CONFIRM_FRAMES}")
        print("Ileri/sag/dikey hiz = 0")
        print("Acil durdurma: Ctrl+C veya QGroundControl Hold/Land")
        print("")

        search_started = time.monotonic()

        confirm_count = 0
        last_processed_frame = -1
        last_log = 0.0
        found = False

        while (
            time.monotonic() - search_started
            < SEARCH_TIMEOUT_S
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.0,
            )

            now = time.monotonic()
            target = node.target

            fresh_target = (
                target is not None
                and target.visible
                and target.confidence >= MIN_CONFIDENCE
                and now - target.received_at <= TARGET_TIMEOUT_S
            )

            if fresh_target:
                # Hedef goruldugu anda donusu kes.
                await send_motion(drone, 0.0)

                if target.frame_id != last_processed_frame:
                    last_processed_frame = target.frame_id
                    confirm_count += 1

                    print(
                        f"KIRMIZI ADAY | "
                        f"conf={target.confidence:.2f} "
                        f"x={target.center_x:.3f} "
                        f"y={target.center_y:.3f} "
                        f"dogrulama={confirm_count}/{CONFIRM_FRAMES}"
                    )

                if confirm_count >= CONFIRM_FRAMES:
                    found = True

                    print("")
                    print("KIRMIZI HEDEF BULUNDU.")
                    print("TARAMA DURDURULDU.")
                    print(
                        f"Son konum: "
                        f"x={target.center_x:.3f}, "
                        f"y={target.center_y:.3f}"
                    )
                    print(
                        f"Confidence: "
                        f"{target.confidence:.2f}"
                    )
                    break

            else:
                confirm_count = 0

                await send_motion(
                    drone,
                    SEARCH_YAW_RATE,
                )

                if now - last_log >= 1.0:
                    elapsed = now - search_started

                    print(
                        f"SEARCH | "
                        f"hedef yok -> yaw="
                        f"{SEARCH_YAW_RATE:.1f} deg/s "
                        f"sure={elapsed:.1f}s"
                    )

                    last_log = now

            await asyncio.sleep(
                1.0 / CONTROL_HZ
            )

        await send_motion(drone, 0.0)

        if not found:
            print("")
            print(
                "TARAMA SURESI DOLDU: "
                "kirmizi hedef bulunamadi."
            )

    finally:
        if offboard_started:
            try:
                await send_motion(drone, 0.0)
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
        print("\nKullanici taramayi durdurdu.")
