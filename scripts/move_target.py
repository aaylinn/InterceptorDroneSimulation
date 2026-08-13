import subprocess
import time

WORLD_NAME = "interceptor_world"
MODEL_NAME = "moving_red_buoy"

X_POSITION = 20.0

Y_MIN = -5.0
Y_MAX = 5.0

Z_POSITION = 3.0

STEP = 0.30
WAIT_TIME = 0.20


def set_target_position(x, y, z):
    request = (
        f'name: "{MODEL_NAME}", '
        f'position: {{x: {x}, y: {y}, z: {z}}}, '
        f'orientation: {{x: 0, y: 0, z: 0, w: 1}}'
    )

    command = [
        "gz",
        "service",
        "-s",
        f"/world/{WORLD_NAME}/set_pose",
        "--reqtype",
        "gz.msgs.Pose",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "500",
        "--req",
        request
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=2
    )

    if result.returncode != 0:
        print("Hata:", result.stderr.strip())


def main():
    y = Y_MIN
    direction = 1

    print("Hareketli hedef baslatildi.")
    print(f"A noktasi: ({X_POSITION}, {Y_MIN}, {Z_POSITION})")
    print(f"B noktasi: ({X_POSITION}, {Y_MAX}, {Z_POSITION})")

    try:
        while True:
            set_target_position(X_POSITION, y, Z_POSITION)

            y += STEP * direction

            if y >= Y_MAX:
                y = Y_MAX
                direction = -1

            elif y <= Y_MIN:
                y = Y_MIN
                direction = 1

            time.sleep(WAIT_TIME)

    except KeyboardInterrupt:
        print("\nHedef hareketi durduruldu.")


if __name__ == "__main__":
    main()