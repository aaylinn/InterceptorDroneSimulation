#!/usr/bin/env bash
set -e

# Bu scriptin bulunduğu proje klasörünü otomatik bul
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Varsayılan PX4 yolu.
# İstenirse dışarıdan PX4_DIR değişkeniyle değiştirilebilir.
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"

if [ ! -d "$PX4_DIR" ]; then
    echo "HATA: PX4-Autopilot bulunamadı: $PX4_DIR"
    echo "Örnek:"
    echo "PX4_DIR=/path/to/PX4-Autopilot ./start_interceptor.sh"
    exit 1
fi

if [ ! -f "$PROJECT_DIR/worlds/interceptor_world.sdf" ]; then
    echo "HATA: interceptor_world.sdf bulunamadı."
    exit 1
fi

export GZ_SIM_RESOURCE_PATH="$PROJECT_DIR/models:$PROJECT_DIR/worlds:${GZ_SIM_RESOURCE_PATH:-}"

env -u LIBGL_ALWAYS_SOFTWARE \
    -u GALLIUM_DRIVER \
    -u MESA_D3D12_DEFAULT_ADAPTER_NAME \
    -u QT_QPA_PLATFORM \
    -u QT_X11_NO_MITSHM \
    bash -c "
        cd \"$PX4_DIR\"

        PX4_GZ_WORLD=interceptor_world \
        make px4_sitl gz_x500_depth
    "
