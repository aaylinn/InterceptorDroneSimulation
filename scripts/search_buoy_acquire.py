#!/usr/bin/env python3

import asyncio
import time

import rclpy
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32MultiArray


TARGET_TOPIC = "/buoy/target"

CONTROL_HZ = 20.0

SEARCH_YAW_RATE = 25.0
SEARCH_TIMEOUT_S = 30.0

MIN_CONFIDENCE = 0.35
TARGET_TIMEOUT_S = 0.60

SEARCH_CONFIRM_FRAMES = 3

# Kirmiziyi tam ortaya degil biraz sola aliyoruz.
# Boylece yesil icin goruntunun saginda daha fazla alan kalir.
ACQUIRE_TARGET_X = 0.40
ACQUIRE_X_TOLERANCE = 0.07

ACQUIRE_YAW_GAIN = 60.0
MAX_ACQUIRE_YAW_RATE = 15.0

ACQUIRE_STABLE_FRAMES = 4
ACQUIRE_LOST_TIMEOUT_S = 1.0


def clamp(value, limit):
    return max(-limit, min(limit, value))


class TargetReceiver(Node):

    def __init__(self):
        super().__init__("buoy_search_acquire")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.visible = False
        self.center_x = 0.0
        self.center_y = 0.0
        self.confidence = 0.0
        self.received_at = 0.0

        self.frame_id = 0

        self.create_subscription(
            Float32MultiArray,
            TARGET_TOPIC,
            self.callback,
            qos,
        )

    def callback(self, msg):
        if len(msg.data) < 8:
            return

        self.frame_id += 1

        self.visible = msg.data[0] >= 0.5
        self.center_x = float(msg.data[1])
        self.center_y = float(msg.data[2])
        self.confidence = float(msg.data[6])
        self.received_at = time.monotonic()


async def send_yaw(drone, yaw_rate):
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(
            0.0,
            0.0,
            0.0,
            yaw_rate,
        )
    )


async def wait_connection(drone):
    print("PX4 baglantisi bekleniyor...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 baglandi.")
            return


async def vehicle_in_air(drone):
    async for value in drone.telemetry.in_air():
        return bool(value)

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

        await wait_connection(drone)

        if not await vehicle_in_air(drone):
            print(
                "DRONE HAVADA DEGIL: "
                "QGroundControl ile Takeoff yap."
            )
            return

        await send_yaw(drone, 0.0)
        await asyncio.sleep(0.2)

        try:
            await drone.offboard.start()
            offboard_started = True

        except OffboardError as error:
            print(
                "Offboard baslatilamadi:",
                error._result.result,
            )
            return

        state = "SEARCH"

        confirmation = 0
        stable_frames = 0

        last_frame = -1
        last_seen_at = 0.0
        last_log = 0.0

        started_at = time.monotonic()

        print("SEARCH + ACQUIRE TESTI BASLADI")
        print(
            f"Tarama hizi: "
            f"{SEARCH_YAW_RATE:.1f} deg/s"
        )
        print(
            f"Acquire hedef x: "
            f"{ACQUIRE_TARGET_X:.2f}"
        )

        while (
            time.monotonic() - started_at
            < SEARCH_TIMEOUT_S
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.0,
            )

            now = time.monotonic()

            fresh = (
                node.visible
                and node.confidence >= MIN_CONFIDENCE
                and now - node.received_at
                <= TARGET_TIMEOUT_S
            )

            if state == "SEARCH":

                if fresh:
                    await send_yaw(drone, 0.0)

                    if node.frame_id != last_frame:
                        last_frame = node.frame_id
                        confirmation += 1

                        print(
                            f"KIRMIZI ADAY | "
                            f"conf={node.confidence:.2f} "
                            f"x={node.center_x:.3f} "
                            f"{confirmation}/"
                            f"{SEARCH_CONFIRM_FRAMES}"
                        )

                    if (
                        confirmation
                        >= SEARCH_CONFIRM_FRAMES
                    ):
                        state = "ACQUIRE"
                        stable_frames = 0
                        last_seen_at = now

                        print(
                            "KIRMIZI DOGRULANDI "
                            "-> ACQUIRE"
                        )

                else:
                    confirmation = 0

                    await send_yaw(
                        drone,
                        SEARCH_YAW_RATE,
                    )

                    if now - last_log >= 1.0:
                        print(
                            "SEARCH | "
                            "kirmizi araniyor"
                        )
                        last_log = now

            elif state == "ACQUIRE":

                if fresh:
                    last_seen_at = now

                    error_x = (
                        node.center_x
                        - ACQUIRE_TARGET_X
                    )

                    yaw_rate = clamp(
                        error_x
                        * ACQUIRE_YAW_GAIN,
                        MAX_ACQUIRE_YAW_RATE,
                    )

                    if (
                        abs(error_x)
                        <= ACQUIRE_X_TOLERANCE
                    ):
                        yaw_rate = 0.0

                        if node.frame_id != last_frame:
                            last_frame = node.frame_id
                            stable_frames += 1
                    else:
                        stable_frames = 0

                    await send_yaw(
                        drone,
                        yaw_rate,
                    )

                    if now - last_log >= 0.5:
                        print(
                            f"ACQUIRE | "
                            f"x={node.center_x:.3f} "
                            f"hata={error_x:+.3f} "
                            f"yaw={yaw_rate:+.1f} "
                            f"stable="
                            f"{stable_frames}/"
                            f"{ACQUIRE_STABLE_FRAMES}"
                        )
                        last_log = now

                    if (
                        stable_frames
                        >= ACQUIRE_STABLE_FRAMES
                    ):
                        await send_yaw(
                            drone,
                            0.0,
                        )

                        print("")
                        print(
                            "HEDEF ACQUIRE TAMAMLANDI."
                        )
                        print(
                            f"Kirmizi x="
                            f"{node.center_x:.3f}"
                        )
                        print(
                            "Tarama durduruldu."
                        )
                        break

                else:
                    await send_yaw(drone, 0.0)

                    if (
                        now - last_seen_at
                        > ACQUIRE_LOST_TIMEOUT_S
                    ):
                        print(
                            "ACQUIRE hedefi kaybetti "
                            "-> SEARCH"
                        )

                        state = "SEARCH"
                        confirmation = 0
                        stable_frames = 0

            await asyncio.sleep(
                1.0 / CONTROL_HZ
            )

        else:
            print(
                "SEARCH/ACQUIRE "
                "zaman asimina ugradi."
            )

    finally:
        if offboard_started:
            try:
                await send_yaw(drone, 0.0)
                await asyncio.sleep(0.3)
                await drone.offboard.stop()

                print(
                    "Offboard durduruldu; "
                    "PX4 Hold'a donmeli."
                )

            except Exception as error:
                print(
                    "Durdurma uyarisi:",
                    error,
                )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nKullanici durdurdu.")
