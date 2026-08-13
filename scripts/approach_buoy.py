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
CONTROL_HZ = 10.0
TARGET_TIMEOUT_S = 0.50
MAX_TEST_TIME_S = 120.0

X_DEADBAND = 0.065
Y_DEADBAND = 0.085
YAW_GAIN = 45.0
VERTICAL_GAIN = 0.55
MAX_YAW_RATE = 10.0
MAX_VERTICAL_SPEED = 0.10

TRUSTED_FORWARD_SOURCE = 1
MIN_FORWARD_CONFIDENCE = 0.40
TRUST_RISE_YOLO = 0.25
TRUST_FALL_UNTRUSTED = 0.10
TRUST_FALL_MISALIGNED = 0.35
FORWARD_TRUST_THRESHOLD = 0.70

FAR_FORWARD_SPEED = 0.12
NEAR_FORWARD_SPEED = 0.05
SLOW_HEIGHT_RATIO = 0.16
STOP_HEIGHT_RATIO = 0.19
STOP_YOLO_FRAMES_REQUIRED = 3
TRUSTED_HEIGHT_TIMEOUT_S = 1.20
NEAR_LOCK_HEIGHT_RATIO = 0.18
NEAR_UNTRUSTED_FRAMES_REQUIRED = 5

POSITION_EMA_ALPHA = 0.35
MAX_POSITION_STEP = 0.10
HEIGHT_EMA_ALPHA = 0.35


@dataclass
class Target:
    visible: bool
    center_x: float
    center_y: float
    width_ratio: float
    height_ratio: float
    area_ratio: float
    confidence: float
    source_code: int
    received_at: float


class TargetReceiver(Node):
    def __init__(self):
        super().__init__("buoy_approach_controller_safe")
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
            width_ratio=float(msg.data[3]),
            height_ratio=float(msg.data[4]),
            area_ratio=float(msg.data[5]),
            confidence=float(msg.data[6]),
            source_code=int(round(msg.data[7])),
            received_at=time.monotonic(),
        )


class GuidanceFilter:
    def __init__(self):
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.yolo_height: Optional[float] = None
        self.last_yolo_height_time: Optional[float] = None

    @staticmethod
    def bounded_measurement(previous: float, measurement: float, max_step: float) -> float:
        delta = max(-max_step, min(max_step, measurement - previous))
        return previous + delta

    def update_position(self, target: Target):
        if self.x is None or self.y is None:
            self.x = target.center_x
            self.y = target.center_y
            return
        bounded_x = self.bounded_measurement(self.x, target.center_x, MAX_POSITION_STEP)
        bounded_y = self.bounded_measurement(self.y, target.center_y, MAX_POSITION_STEP)
        self.x = POSITION_EMA_ALPHA * bounded_x + (1.0 - POSITION_EMA_ALPHA) * self.x
        self.y = POSITION_EMA_ALPHA * bounded_y + (1.0 - POSITION_EMA_ALPHA) * self.y

    def update_trusted_height(self, target: Target, now: float):
        if target.source_code != TRUSTED_FORWARD_SOURCE:
            return
        if target.confidence < MIN_FORWARD_CONFIDENCE:
            return
        if self.yolo_height is None:
            self.yolo_height = target.height_ratio
        else:
            self.yolo_height = (
                HEIGHT_EMA_ALPHA * target.height_ratio
                + (1.0 - HEIGHT_EMA_ALPHA) * self.yolo_height
            )
        self.last_yolo_height_time = now

    def trusted_height_is_fresh(self, now: float) -> bool:
        return (
            self.yolo_height is not None
            and self.last_yolo_height_time is not None
            and now - self.last_yolo_height_time <= TRUSTED_HEIGHT_TIMEOUT_S
        )

    def reset(self):
        self.x = None
        self.y = None
        self.yolo_height = None
        self.last_yolo_height_time = None


def clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def source_name(source_code: int) -> str:
    return {1: "YOLO", 2: "ROI", 3: "FALLBACK", 4: "HSV"}.get(source_code, "UNKNOWN")


async def send_velocity(drone: System, forward_speed: float, down_speed: float, yaw_rate: float):
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(forward_speed, 0.0, down_speed, yaw_rate)
    )


async def stop_motion(drone: System):
    await send_velocity(drone, 0.0, 0.0, 0.0)


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


def calculate_forward_speed(height_ratio: float) -> float:
    if height_ratio >= STOP_HEIGHT_RATIO:
        return 0.0
    if height_ratio <= SLOW_HEIGHT_RATIO:
        return FAR_FORWARD_SPEED
    span = STOP_HEIGHT_RATIO - SLOW_HEIGHT_RATIO
    progress = clamp01((height_ratio - SLOW_HEIGHT_RATIO) / max(span, 1e-6))
    return FAR_FORWARD_SPEED + (NEAR_FORWARD_SPEED - FAR_FORWARD_SPEED) * progress


async def run():
    rclpy.init()
    node = TargetReceiver()
    drone = System()
    offboard_started = False
    guidance = GuidanceFilter()

    try:
        await drone.connect(system_address="udpin://0.0.0.0:14540")
        await wait_for_connection(drone)

        if not await vehicle_is_in_air(drone):
            print("DRONE HAVADA DEGIL: QGroundControl ile yaklasik 2 metre takeoff yap.")
            return

        await stop_motion(drone)
        print("Offboard baslatiliyor...")

        try:
            await drone.offboard.start()
            offboard_started = True
        except OffboardError as error:
            print(f"Offboard baslatilamadi: {error._result.result}")
            return

        print("YAKLASMA-DURMA TESTI (GUVENLI OPTIMIZE) BASLADI.")
        print("ROI/HSV hizalamaya yardim eder; ileri hareket ettiremez.")
        print(f"Yavaslama h={SLOW_HEIGHT_RATIO:.2f}, durma h={STOP_HEIGHT_RATIO:.2f}")
        print("Acil durdurma: Ctrl+C veya QGroundControl Hold/Land")

        start_time = time.monotonic()
        trust = 0.0
        stop_yolo_frames = 0
        near_untrusted_frames = 0
        last_log_time = 0.0
        success = False

        while time.monotonic() - start_time < MAX_TEST_TIME_S:
            rclpy.spin_once(node, timeout_sec=0.0)
            now = time.monotonic()
            target = node.target

            target_is_fresh = (
                target is not None
                and target.visible
                and now - target.received_at <= TARGET_TIMEOUT_S
            )

            if not target_is_fresh:
                trust = 0.0
                stop_yolo_frames = 0
                near_untrusted_frames = 0
                guidance.reset()
                await stop_motion(drone)
                if now - last_log_time >= 0.5:
                    print("HEDEF YOK/ESKI -> ileri=0.00, tum hareketler durduruldu")
                    last_log_time = now
                await asyncio.sleep(1.0 / CONTROL_HZ)
                continue

            guidance.update_position(target)
            guidance.update_trusted_height(target, now)

            error_x = guidance.x - 0.5
            error_y = guidance.y - 0.5
            yaw_rate = 0.0
            down_speed = 0.0
            forward_speed = 0.0

            if abs(error_x) > X_DEADBAND:
                yaw_rate = clamp(error_x * YAW_GAIN, MAX_YAW_RATE)
            if abs(error_y) > Y_DEADBAND:
                down_speed = clamp(error_y * VERTICAL_GAIN, MAX_VERTICAL_SPEED)

            aligned = abs(error_x) <= X_DEADBAND and abs(error_y) <= Y_DEADBAND
            current_is_trusted_yolo = (
                target.source_code == TRUSTED_FORWARD_SOURCE
                and target.confidence >= MIN_FORWARD_CONFIDENCE
            )

            if not aligned:
                trust -= TRUST_FALL_MISALIGNED
                stop_yolo_frames = 0
                state = "ALIGN"
            elif not current_is_trusted_yolo:
                trust -= TRUST_FALL_UNTRUSTED
                stop_yolo_frames = max(0, stop_yolo_frames - 1)
                state = "VERIFY"
            else:
                trust += TRUST_RISE_YOLO
                state = "STABLE"

            trust = clamp01(trust)
            trusted_height_fresh = guidance.trusted_height_is_fresh(now)
            near_lock_available = (
                aligned
                and trusted_height_fresh
                and guidance.yolo_height is not None
                and guidance.yolo_height >= NEAR_LOCK_HEIGHT_RATIO
            )
            if near_lock_available and not current_is_trusted_yolo:
                near_untrusted_frames += 1
            elif current_is_trusted_yolo:
                near_untrusted_frames = 0

            if (
                aligned
                and current_is_trusted_yolo
                and trusted_height_fresh
                and guidance.yolo_height >= STOP_HEIGHT_RATIO
            ):
                forward_speed = 0.0
                stop_yolo_frames += 1
                state = "STOP"
            elif (
                aligned
                and current_is_trusted_yolo
                and trusted_height_fresh
                and trust >= FORWARD_TRUST_THRESHOLD
            ):
                forward_speed = calculate_forward_speed(guidance.yolo_height)
                state = "SLOW" if guidance.yolo_height >= SLOW_HEIGHT_RATIO else "APPROACH"
            else:
                forward_speed = 0.0

            await send_velocity(drone, forward_speed, down_speed, yaw_rate)

            if now - last_log_time >= 0.4:
                height_text = (
                    f"{guidance.yolo_height:.3f}"
                    if guidance.yolo_height is not None
                    else "None"
                )
                print(
                    f"{state} | src={source_name(target.source_code)} "
                    f"conf={target.confidence:.2f} x={guidance.x:.3f} "
                    f"y={guidance.y:.3f} yolo_h={height_text} | "
                    f"trust={trust:.2f} ileri={forward_speed:.2f} "
                    f"asagi={down_speed:.2f} yaw={yaw_rate:.1f} "
                    f"stop={stop_yolo_frames}/{STOP_YOLO_FRAMES_REQUIRED} "
                    f"near_roi={near_untrusted_frames}/"
                    f"{NEAR_UNTRUSTED_FRAMES_REQUIRED}"
                )
                last_log_time = now

            if near_untrusted_frames >= NEAR_UNTRUSTED_FRAMES_REQUIRED:
                await stop_motion(drone)
                print(
                    "YAKLASMA BASARILI: yakin mesafede "
                    "YOLO-ROI/HSV gecisi algilandi ve duruldu."
                )
                print(
                    f"Son guvenilir YOLO height_ratio="
                    f"{guidance.yolo_height:.3f}"
                )
                success = True
                break

            if stop_yolo_frames >= STOP_YOLO_FRAMES_REQUIRED:
                await stop_motion(drone)
                print("YAKLASMA BASARILI: guvenilir YOLO mesafe esiginde duruldu.")
                print(
                    f"Son YOLO height_ratio={guidance.yolo_height:.3f}, "
                    f"ham area_ratio={target.area_ratio:.3f}"
                )
                success = True
                break

            await asyncio.sleep(1.0 / CONTROL_HZ)

        if not success:
            print("Test suresi doldu veya guvenilir YOLO durma esigine ulasilamadi.")

    finally:
        if offboard_started:
            try:
                await stop_motion(drone)
                await asyncio.sleep(0.3)
                await drone.offboard.stop()
                print("Offboard durduruldu; PX4 Hold moduna donmeli.")
            except Exception as error:
                print(f"Durdurma uyarisi: {error}")

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nKullanici durdurdu; ileri hareket kesildi.")
