#!/usr/bin/env bash

set -euo pipefail

readonly SDK_VERSION="2.58.3"
readonly SDK_COMMIT="dfd6aa91250f5c31521d72d627865417989bb4e7"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly RUNTIME_DIR="${PROJECT_ROOT}/.runtime"
readonly SOURCE_DIR="${RUNTIME_DIR}/librealsense-${SDK_VERSION}"
readonly BUILD_DIR="${RUNTIME_DIR}/librealsense-build"
readonly LIBUSB_DIR="${RUNTIME_DIR}/libusb-local"
readonly OUTPUT_DIR="${PROJECT_ROOT}/src"
readonly BUILD_JOBS="${PARCEL_POSE_BUILD_JOBS:-4}"

if [[ -n "${PARCEL_POSE_PYTHON:-}" ]]; then
    python_executable="${PARCEL_POSE_PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    python_executable="${CONDA_PREFIX}/bin/python3.12"
else
    echo "Activate the Python 3.12 conda environment first." >&2
    exit 2
fi

if [[ ! -x "${python_executable}" ]]; then
    echo "Python 3.12 executable not found: ${python_executable}" >&2
    exit 2
fi
if [[ ! "${BUILD_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PARCEL_POSE_BUILD_JOBS must be a positive integer." >&2
    exit 2
fi

python_version="$(${python_executable} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "${python_version}" != "3.12" ]]; then
    echo "Expected Python 3.12, got ${python_version}: ${python_executable}" >&2
    exit 2
fi

verify_repo_binding() {
    PYTHONPATH="${OUTPUT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
        "${python_executable}" -c '
import pathlib
import sys

import pyrealsense2 as rs

module_path = pathlib.Path(rs.__file__).resolve()
output_dir = pathlib.Path(sys.argv[1]).resolve()
expected_version = sys.argv[2]
actual_version = getattr(rs, "__version__", None)
if module_path.parent != output_dir:
    raise SystemExit(f"binding is not repo-local: {module_path}")
if actual_version != expected_version:
    raise SystemExit(
        f"binding version is {actual_version!r}, expected {expected_version}"
    )
print(f"pyrealsense2 {rs.__version__} ready: {module_path}")
' "${OUTPUT_DIR}" "${SDK_VERSION}"
}

if verify_repo_binding >/dev/null 2>&1; then
    echo "Compatible repo-local pyrealsense2 is already available."
    exit 0
fi

for command_name in cmake git; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Required build command is missing: ${command_name}" >&2
        exit 2
    fi
done

mkdir -p "${RUNTIME_DIR}"
if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
    git clone --depth 1 --branch "v${SDK_VERSION}" \
        https://github.com/realsenseai/librealsense.git \
        "${SOURCE_DIR}"
fi

actual_commit="$(git -C "${SOURCE_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${SDK_COMMIT}" ]]; then
    echo "Unexpected librealsense source commit: ${actual_commit}" >&2
    echo "Expected v${SDK_VERSION}: ${SDK_COMMIT}" >&2
    exit 2
fi
if [[ -n "$(git -C "${SOURCE_DIR}" status --porcelain)" ]]; then
    echo "librealsense source tree has local changes: ${SOURCE_DIR}" >&2
    exit 2
fi

cmake_args=(
    -S "${SOURCE_DIR}"
    -B "${BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_PYTHON_BINDINGS=true
    -DPYTHON_EXECUTABLE="${python_executable}"
    -DBUILD_SHARED_LIBS=false
    -DBUILD_EXAMPLES=false
    -DBUILD_GRAPHICAL_EXAMPLES=false
    -DBUILD_TOOLS=false
    -DBUILD_UNIT_TESTS=false
    -DBUILD_ROSBAG2=false
    -DBUILD_WITH_DDS=false
    -DBUILD_WITH_CUDA=false
    -DFORCE_RSUSB_BACKEND=false
)

cmake "${cmake_args[@]}"

# JetPack has the libusb runtime but may not include its development headers.
# librealsense already defines a repo-local libusb target in that case. Build it
# first, then point the final configure step at a stable path outside BUILD_DIR.
if grep -q 'LIBUSB_INC.*NOTFOUND' "${BUILD_DIR}/CMakeCache.txt"; then
    cmake --build "${BUILD_DIR}" --target libusb -j"${BUILD_JOBS}"
    mkdir -p "${LIBUSB_DIR}/include/libusb-1.0" "${LIBUSB_DIR}/lib"
    install -m 0644 \
        "${BUILD_DIR}/libusb_install/include/libusb-1.0/libusb.h" \
        "${LIBUSB_DIR}/include/libusb-1.0/libusb.h"
    install -m 0644 \
        "${BUILD_DIR}/libusb_install/lib/libusb-1.0.a" \
        "${LIBUSB_DIR}/lib/libusb-1.0.a"
    cmake "${cmake_args[@]}" \
        -DLIBUSB_INC="${LIBUSB_DIR}/include/libusb-1.0" \
        -DLIBUSB_LIB="${LIBUSB_DIR}/lib/libusb-1.0.a"
fi

cmake --build "${BUILD_DIR}" --target pyrealsense2 -j"${BUILD_JOBS}"

binding_path="$(
    find "${BUILD_DIR}" -name 'pyrealsense2*.so' \
        -not -path '*/CMakeFiles/*' -print -quit
)"
if [[ -z "${binding_path}" ]]; then
    echo "Build completed without a pyrealsense2 extension." >&2
    exit 1
fi

install -m 0644 "${binding_path}" "${OUTPUT_DIR}/$(basename "${binding_path}")"
verify_repo_binding
