import subprocess
import time

WORLD_NAME = "interceptor_world"

RED_MODEL = "moving_red_buoy"
GREEN_MODEL = "fixed_green_buoy"

RED_X = 8.0
GREEN_X = 8.9
Z_POSITION = 3.0

Y_MIN = -0.8
Y_MAX = 0.8

STEP = 0.02
WAIT_TIME = 0.20


def set_pose(model_name, x, y, z):
    request = (
        f'name: "{model_name}", '
        f'position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{x: 0, y: 0, z: 0, w: 1}}'
    )

    result = subprocess.run(
        [
            "gz",
            "service",
            "-s",
            f"/world/{WORLD_NAME}/set_pose",
            "--reqtype",
            "gz.msgs.Pose",
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

    if result.returncode != 0 or "true" not in result.stdout.lower():
        print(
            f"{model_name} hareket hatası: "
            f"{result.stdout.strip()} {result.stderr.strip()}"
        )


def main():
    # Kırmızı A'dan B'ye başlar.
    red_y = Y_MIN
    red_direction = 1

    # Yeşil B'den A'ya başlar.
    green_y = Y_MAX
    green_direction = -1

    print("İki duba ters yönlerde hareket ediyor.")
    print("Kırmızı: A -> B")
    print("Yeşil: B -> A")
    print("Durdurmak için Ctrl+C")

    try:
        while True:
            set_pose(RED_MODEL, RED_X, red_y, Z_POSITION)
            set_pose(GREEN_MODEL, GREEN_X, green_y, Z_POSITION)

            red_y += STEP * red_direction
            green_y += STEP * green_direction

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

            time.sleep(WAIT_TIME)

    except KeyboardInterrupt:
        print("\nİki dubanın hareketi durduruldu.")


if __name__ == "__main__":
    main()
