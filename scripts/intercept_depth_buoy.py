#!/usr/bin/env python3

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import rclpy

from mavsdk import System
from mavsdk.offboard import (
    OffboardError,
    VelocityBodyYawspeed,
)
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Float32MultiArray


TARGET_TOPIC = "/buoy/target_control"

CONTROL_HZ = 20.0
TARGET_TIMEOUT_S = 1.20
MAX_TEST_TIME_S = 180.0

MIN_CONFIDENCE = 0.25
MIN_DEPTH_VALID_RATIO = 0.05
MIN_VALID_DEPTH_M = 0.10
MAX_VALID_DEPTH_M = 30.0

X_DEADBAND = 0.080
Y_DEADBAND = 0.20

YAW_GAIN = 100.0
VERTICAL_GAIN = 0.25

MAX_YAW_RATE = 60.0
MAX_VERTICAL_SPEED = 0.15

# Depth tabanlı yaklaşma hızları
FAR_DEPTH_M = 5.0
MID_DEPTH_M = 2.0
NEAR_DEPTH_M = 0.65

FAR_SPEED = 10.0
MID_SPEED = 4.0
NEAR_SPEED = 0.5
VERY_NEAR_SPEED = 0.5

# Bu mesafede süre tabanlı kör yaklaşma yerine,
# hizalamayı sürdürerek temas hamlesi başlar.
COMMIT_DEPTH_M = 1.40
COMMIT_SPEED = 0.80

# Kalman merkez tahmininden üretilecek yanal hareket
COMMIT_LATERAL_GAIN = 3.0
MAX_COMMIT_LATERAL_SPEED = 1.5

# Hedef görüntünün kenarındayken ileri hareket tamamen kesilmez
COMMIT_EDGE_FORWARD_SPEED = 0.20

# Temas moduna geçiş için normal hizalamadan daha geniş pencere
COMMIT_X_WINDOW = 0.18
COMMIT_Y_WINDOW = 0.20

# Temas hamlesi için güvenlik üst sınırı.
# Başlangıç kararı mesafeden verilir.
MAX_COMMIT_TIME_S = 8.0

# Çok yakında YOLO/depth kaybolursa hedef kamerayı
# doldurmuş olabilir. Son doğrultuda kısa süre devam edilir.
BLIND_COMMIT_TIME_S = 2.0

POSITION_ALPHA = 0.40
DEPTH_ALPHA = 0.35
MAX_DEPTH_STEP_M = 0.75


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
    depth_m: float
    depth_valid_ratio: float
    received_at: float


class TargetReceiver(Node):

    def __init__(self):
        super().__init__(
            "depth_buoy_intercept_controller"
        )

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

    def target_callback(
        self,
        message: Float32MultiArray,
    ):
        if len(message.data) < 10:
            return

        # Tek karelik YOLO kaybında son güvenilir
        # kırmızı hedef hemen silinmez. received_at
        # değişmediği için en fazla TARGET_TIMEOUT_S
        # boyunca korunur.
        if message.data[0] < 0.5:
            return

        self.target = Target(
            visible=message.data[0] >= 0.5,
            center_x=float(message.data[1]),
            center_y=float(message.data[2]),
            width_ratio=float(message.data[3]),
            height_ratio=float(message.data[4]),
            area_ratio=float(message.data[5]),
            confidence=float(message.data[6]),
            source_code=int(
                round(message.data[7])
            ),
            depth_m=float(message.data[8]),
            depth_valid_ratio=float(
                message.data[9]
            ),
            received_at=time.monotonic(),
        )


def clamp(value, limit):
    return max(-limit, min(limit, value))


async def send_velocity(
    drone,
    forward_speed,
    down_speed,
    yaw_rate,
    right_speed=0.0,
):
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(
            forward_speed,
            right_speed,
            down_speed,
            yaw_rate,
        )
    )


async def stop_motion(drone):
    await send_velocity(
        drone,
        0.0,
        0.0,
        0.0,
    )


async def wait_for_connection(drone):
    print("PX4 bağlantısı bekleniyor...")

    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 bağlandı.")
            return


async def vehicle_is_in_air(drone):
    async for in_air in drone.telemetry.in_air():
        return bool(in_air)

    return False


def approach_speed(depth_m):
    if depth_m > FAR_DEPTH_M:
        return FAR_SPEED

    if depth_m > MID_DEPTH_M:
        return MID_SPEED

    if depth_m > NEAR_DEPTH_M:
        return NEAR_SPEED

    return VERY_NEAR_SPEED


def depth_is_valid(target):
    return (
        MIN_VALID_DEPTH_M
        <= target.depth_m
        <= MAX_VALID_DEPTH_M
        and target.depth_valid_ratio
        >= MIN_DEPTH_VALID_RATIO
    )


async def run():
    rclpy.init()

    node = TargetReceiver()
    drone = System()

    offboard_started = False

    filtered_x = None
    filtered_y = None
    filtered_depth = None

    commit_started_at = None
    last_target_seen_at = None
    last_log_time = 0.0

    try:
        await drone.connect(
            system_address="udpin://0.0.0.0:14540"
        )

        await wait_for_connection(drone)

        if not await vehicle_is_in_air(drone):
            print(
                "DRONE HAVADA DEĞİL: "
                "QGroundControl ile yaklaşık 2 metre "
                "Takeoff yap ve Hold moduna al."
            )
            return

        await stop_motion(drone)

        print("Offboard başlatılıyor...")

        try:
            await drone.offboard.start()
            offboard_started = True

        except OffboardError as error:
            print(
                "Offboard başlatılamadı: "
                f"{error._result.result}"
            )
            return

        print(
            "DEPTH TABANLI KIRMIZI HEDEFE "
            "TEMAS TESTİ BAŞLADI."
        )
        print(
            f"Temas hamlesi başlangıcı: "
            f"{COMMIT_DEPTH_M:.2f} m"
        )
        print(
            f"Temas hızı: "
            f"{COMMIT_SPEED:.2f} m/s"
        )
        print(
            "Acil durdurma: Ctrl+C veya "
            "QGroundControl Hold/Land"
        )

        test_started_at = time.monotonic()

        while (
            time.monotonic() - test_started_at
            < MAX_TEST_TIME_S
        ):
            rclpy.spin_once(
                node,
                timeout_sec=0.0,
            )

            now = time.monotonic()
            target = node.target

            target_fresh = (
                target is not None
                and target.visible
                and target.confidence
                >= MIN_CONFIDENCE
                and now - target.received_at
                <= TARGET_TIMEOUT_S
            )

            # Temas hamlesi başladıktan sonraki kontrol
            if commit_started_at is not None:
                commit_elapsed = (
                    now - commit_started_at
                )

                if (
                    commit_elapsed
                    >= MAX_COMMIT_TIME_S
                ):
                    await stop_motion(drone)

                    print(
                        "TEMAS HAMLESİ TAMAMLANDI: "
                        "ileri hareket kesildi."
                    )
                    break

                if target_fresh:
                    last_target_seen_at = now

                    error_x = (
                        target.center_x - 0.5
                    )
                    error_y = (
                        target.center_y - 0.5
                    )

                    yaw_rate = clamp(
                        error_x * YAW_GAIN,
                        MAX_YAW_RATE,
                    )

                    down_speed = clamp(
                        error_y * VERTICAL_GAIN,
                        MAX_VERTICAL_SPEED,
                    )

                    # Kalman tarafından süzülmüş/tahmin edilmiş
                    # hedef merkezinden sağ-sol gövde hızı üretilir.
                    right_speed = clamp(
                        error_x * COMMIT_LATERAL_GAIN,
                        MAX_COMMIT_LATERAL_SPEED,
                    )

                    # Hedef görüntünün kenarındayken tamamen
                    # durmak yerine yavaş ileri hareket korunur.
                    if abs(error_x) > 0.30:
                        forward_speed = (
                            COMMIT_EDGE_FORWARD_SPEED
                        )
                        state = "COMMIT_INTERCEPT"
                    else:
                        forward_speed = COMMIT_SPEED
                        state = "COMMIT_TRACK"

                    current_depth = (
                        target.depth_m
                        if depth_is_valid(target)
                        else 0.0
                    )

                else:
                    time_since_seen = (
                        now - last_target_seen_at
                        if last_target_seen_at
                        is not None
                        else 999.0
                    )

                    if (
                        time_since_seen
                        <= BLIND_COMMIT_TIME_S
                    ):
                        # Hedef çok yakında kamerayı
                        # doldurduysa kısa temas devamı.
                        forward_speed = COMMIT_SPEED
                        right_speed = 0.0
                        down_speed = 0.0
                        yaw_rate = 0.0
                        current_depth = 0.0
                        state = "COMMIT_BLIND"

                    else:
                        await stop_motion(drone)

                        print(
                            "TEMAS İPTAL: hedef temas "
                            "aşamasında uzun süre kayboldu."
                        )
                        break

                await send_velocity(
                    drone,
                    forward_speed,
                    down_speed,
                    yaw_rate,
                    right_speed=right_speed,
                )

                if now - last_log_time >= 0.25:
                    print(
                        f"{state} | "
                        f"depth={current_depth:.3f}m "
                        f"ileri={forward_speed:.2f} "
                        f"sag={right_speed:.2f} "
                        f"yaw={yaw_rate:.1f} "
                        f"asagi={down_speed:.2f} "
                        f"sure={commit_elapsed:.2f}/"
                        f"{MAX_COMMIT_TIME_S:.2f}"
                    )

                    last_log_time = now

                await asyncio.sleep(
                    1.0 / CONTROL_HZ
                )
                continue

            # Normal yaklaşma aşaması
            if not target_fresh:
                await stop_motion(drone)

                filtered_x = None
                filtered_y = None
                filtered_depth = None

                if now - last_log_time >= 0.50:
                    print(
                        "KIRMIZI HEDEF YOK/ESKİ "
                        "-> tüm hareketler durdu"
                    )
                    last_log_time = now

                await asyncio.sleep(
                    1.0 / CONTROL_HZ
                )
                continue

            last_target_seen_at = now

            if filtered_x is None:
                filtered_x = target.center_x
                filtered_y = target.center_y
            else:
                filtered_x = (
                    POSITION_ALPHA
                    * target.center_x
                    + (1.0 - POSITION_ALPHA)
                    * filtered_x
                )

                filtered_y = (
                    POSITION_ALPHA
                    * target.center_y
                    + (1.0 - POSITION_ALPHA)
                    * filtered_y
                )

            if depth_is_valid(target):
                if filtered_depth is None:
                    filtered_depth = target.depth_m
                else:
                    bounded_depth = max(
                        filtered_depth
                        - MAX_DEPTH_STEP_M,
                        min(
                            filtered_depth
                            + MAX_DEPTH_STEP_M,
                            target.depth_m,
                        ),
                    )

                    filtered_depth = (
                        DEPTH_ALPHA
                        * bounded_depth
                        + (1.0 - DEPTH_ALPHA)
                        * filtered_depth
                    )
            else:
                filtered_depth = None

            error_x = filtered_x - 0.5
            error_y = filtered_y - 0.5

            yaw_rate = 0.0
            down_speed = 0.0
            forward_speed = 0.0

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

            if filtered_depth is None:
                state = "DEPTH_BEKLE"
                forward_speed = 0.0

            elif (
                abs(error_x) <= COMMIT_X_WINDOW
                and abs(error_y) <= COMMIT_Y_WINDOW
                and filtered_depth <= COMMIT_DEPTH_M
            ):
                await stop_motion(drone)
                await asyncio.sleep(0.15)

                commit_started_at = (
                    time.monotonic()
                )

                print(
                    "TEMAS HAMLESİ BAŞLIYOR | "
                    f"depth={filtered_depth:.3f}m "
                    f"conf={target.confidence:.2f}"
                )

                await asyncio.sleep(
                    1.0 / CONTROL_HZ
                )
                continue

            elif not aligned:
                state = "ALIGN"
                forward_speed = 0.0

            else:
                forward_speed = approach_speed(
                    filtered_depth
                )
                state = "APPROACH_DEPTH"

            await send_velocity(
                drone,
                forward_speed,
                down_speed,
                yaw_rate,
            )

            if now - last_log_time >= 0.40:
                depth_text = (
                    f"{filtered_depth:.3f}"
                    if filtered_depth is not None
                    else "None"
                )

                print(
                    f"{state} | "
                    f"conf={target.confidence:.2f} "
                    f"x={filtered_x:.3f} "
                    f"y={filtered_y:.3f} "
                    f"depth={depth_text}m "
                    f"valid="
                    f"{target.depth_valid_ratio:.2f} "
                    f"ileri={forward_speed:.2f} "
                    f"yaw={yaw_rate:.1f} "
                    f"asagi={down_speed:.2f}"
                )

                last_log_time = now

            await asyncio.sleep(
                1.0 / CONTROL_HZ
            )

        else:
            print(
                "Maksimum test süresi doldu."
            )

    finally:
        if offboard_started:
            try:
                await stop_motion(drone)
                await asyncio.sleep(0.30)
                await drone.offboard.stop()

                print(
                    "Offboard durduruldu; "
                    "PX4 Hold moduna dönmeli."
                )

            except Exception as error:
                print(
                    f"Durdurma uyarısı: {error}"
                )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(run())

    except KeyboardInterrupt:
        print(
            "\nKullanıcı durdurdu; "
            "ileri hareket kesildi."
        )
