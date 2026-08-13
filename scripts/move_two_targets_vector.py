import subprocess
import time


WORLD_NAME = "interceptor_world"

RED_MODEL = "moving_red_buoy"
GREEN_MODEL = "fixed_green_buoy"

RED_X = 20.0
GREEN_X = 20.8
Z_POSITION = 3.0

Y_MIN = -0.8
Y_MAX = 0.8

STEP = 0.30
WAIT_TIME = 0.20


def set_both_positions(red_y, green_y):
    request = (
        "pose { "
        f'name: "{RED_MODEL}" '
        f"position {{ "
        f"x: {RED_X} "
        f"y: {red_y} "
        f"z: {Z_POSITION} "
        "} "
        "orientation { "
        "x: 0 y: 0 z: 0 w: 1 "
        "} "
        "} "
        "pose { "
        f'name: "{GREEN_MODEL}" '
        f"position {{ "
        f"x: {GREEN_X} "
        f"y: {green_y} "
        f"z: {Z_POSITION} "
        "} "
        "orientation { "
        "x: 0 y: 0 z: 0 w: 1 "
        "} "
        "}"
    )

    result = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            (
                f"/world/{WORLD_NAME}/"
                "set_pose_vector"
            ),
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

    successful = (
        result.returncode == 0
        and "true" in result.stdout.lower()
    )

    if not successful:
        print(
            "Eşzamanlı hareket hatası:",
            result.stdout.strip(),
            result.stderr.strip(),
        )

    return successful


def main():
    red_y = Y_MIN
    green_y = Y_MAX

    red_direction = 1
    green_direction = -1

    last_log_time = 0.0

    print("Eşzamanlı iki duba hareketi başladı.")
    print("Kırmızı: A -> B")
    print("Yeşil: B -> A")
    print("Tek set_pose_vector çağrısı kullanılıyor.")
    print("Durdurmak için Ctrl+C")

    try:
        while True:
            successful = set_both_positions(
                red_y,
                green_y,
            )

            if successful:
                red_y += STEP * red_direction
                green_y += (
                    STEP * green_direction
                )

                if red_y >= Y_MAX:
                    red_y = Y_MAX
                    red_direction = -1
                elif red_y <= Y_MIN:
                    red_y = Y_MIN
                    red_direction = 1

                if green_y >= Y_MAX:
                    green_y = Y_MAX
                    green_direction = -1
                elif green_y <= Y_MIN:
                    green_y = Y_MIN
                    green_direction = 1

            now = time.monotonic()

            if now - last_log_time >= 1.0:
                print(
                    f"kırmızı_y={red_y:+.2f} "
                    f"yeşil_y={green_y:+.2f}"
                )
                last_log_time = now

            time.sleep(WAIT_TIME)

    except KeyboardInterrupt:
        print(
            "\nİki dubanın hareketi durduruldu."
        )


if __name__ == "__main__":
    main()
