#!/usr/bin/env python3

import math
import subprocess
import time


WORLD_NAME = "interceptor_world"

RED_MODEL = "moving_red_buoy"
GREEN_MODEL = "fixed_green_buoy"

RED_X = -8.0
GREEN_X = -8.0

# Kirmizi: soldaki A-B koridoru
RED_Y_MIN = -4.0
RED_Y_MAX = -0.6
RED_SPEED = 1.0

# Sinus hareketinin dikey bileseni
RED_Z_CENTER = 3.0
RED_Z_AMPLITUDE = 0.60
RED_WAVELENGTH = 1.70

# Yesil: sagdaki A-B koridoru
GREEN_Y_MIN = 0.6
GREEN_Y_MAX = 4.0
GREEN_Z = 3.0
GREEN_SPEED = 1.0

UPDATE_PERIOD = 0.10


def set_positions(red_y, red_z, green_y):
    request = (
        "pose { "
        f'name: "{RED_MODEL}" '
        "position { "
        f"x: {RED_X} "
        f"y: {red_y} "
        f"z: {red_z} "
        "} "
        "orientation { x: 0 y: 0 z: 0 w: 1 } "
        "} "
        "pose { "
        f'name: "{GREEN_MODEL}" '
        "position { "
        f"x: {GREEN_X} "
        f"y: {green_y} "
        f"z: {GREEN_Z} "
        "} "
        "orientation { x: 0 y: 0 z: 0 w: 1 } "
        "}"
    )

    result = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            f"/world/{WORLD_NAME}/set_pose_vector",
            "--reqtype",
            "gz.msgs.Pose_V",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            request,
        ],
        capture_output=True,
        text=True,
        timeout=3,
    )

    return (
        result.returncode == 0
        and "true" in result.stdout.lower()
    )


def main():
    red_y = RED_Y_MIN
    green_y = GREEN_Y_MAX

    red_direction = 1
    green_direction = -1

    previous_time = time.monotonic()
    last_log_time = 0.0

    print("KARMA YORUNGE TESTI BASLADI")
    print("Kirmizi: A-B + dikey sinus")
    print("Yesil: dogrusal A-B")
    print("Durdurmak icin Ctrl+C")

    try:
        while True:
            now = time.monotonic()
            dt = now - previous_time
            previous_time = now

            red_y += RED_SPEED * red_direction * dt
            green_y += GREEN_SPEED * green_direction * dt

            if red_y >= RED_Y_MAX:
                red_y = RED_Y_MAX
                red_direction = -1
            elif red_y <= RED_Y_MIN:
                red_y = RED_Y_MIN
                red_direction = 1

            if green_y >= GREEN_Y_MAX:
                green_y = GREEN_Y_MAX
                green_direction = -1
            elif green_y <= GREEN_Y_MIN:
                green_y = GREEN_Y_MIN
                green_direction = 1

            # Y boyunca ilerledikce Z sinus cizer.
            distance = red_y - RED_Y_MIN

            red_z = (
                RED_Z_CENTER
                + RED_Z_AMPLITUDE
                * math.sin(
                    2.0
                    * math.pi
                    * distance
                    / RED_WAVELENGTH
                )
            )

            successful = set_positions(
                red_y,
                red_z,
                green_y,
            )

            if not successful:
                print("set_pose_vector basarisiz.")

            if now - last_log_time >= 0.5:
                direction_text = (
                    "A->B"
                    if red_direction > 0
                    else "B->A"
                )

                print(
                    f"Kirmizi {direction_text}: "
                    f"y={red_y:+.2f} "
                    f"z={red_z:+.2f} | "
                    f"Yesil y={green_y:+.2f}"
                )

                last_log_time = now

            time.sleep(UPDATE_PERIOD)

    except KeyboardInterrupt:
        print("\nHareket durduruldu.")


if __name__ == "__main__":
    main()
