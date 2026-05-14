#!/bin/bash
# build.sh - 在 RK3588 板子上编译推理库
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
LIBS_DIR="${SCRIPT_DIR}/libs/rknn_api"

mkdir -p "${BUILD_DIR}"

# 检查 RKNN API 是否存在
if [ ! -d "${LIBS_DIR}/include" ]; then
    echo "RKNN API not found at ${LIBS_DIR}"
    echo "Trying system paths..."
fi

cd "${BUILD_DIR}"
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DRKNN_API_DIR="${LIBS_DIR}"

make -j$(nproc)

echo ""
echo "Build complete!"
echo "Library: ${BUILD_DIR}/librknn_infer.so"
ls -lh "${BUILD_DIR}/librknn_infer.so"
