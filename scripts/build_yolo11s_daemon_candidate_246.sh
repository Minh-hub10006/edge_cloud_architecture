#!/usr/bin/env bash
# New candidate binary only. Never install into edge_artifacts/bin.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
candidate_root="$PWD"
sdk=/home/ubuntu/sigm/qairt/2.43.0.260128
abi=aarch64-oe-linux-gcc11.2
dlc=/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/npu_run/edge_packet/yolo11s320_w8a8_qcs6490.dlc
test "$(uname -m)" = aarch64
command -v g++ >/dev/null
test -f "$sdk/include/SNPE/SNPE/SNPEBuilder.h"
test -f "$sdk/lib/$abi/libSNPE.so"
test ! -e "$candidate_root/build"
mkdir "$candidate_root/build"
export LD_LIBRARY_PATH="$sdk/lib/$abi:/home/ubuntu/miniconda3/envs/tracking/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ADSP_LIBRARY_PATH="$sdk/lib/hexagon-v68/unsigned;$sdk/lib/$abi;/usr/lib/rfsa/adsp;/dsp;/opt"
printf '1bb6a516a06e23bb9a6022c0b537574745aee603adc2539ec8a885d78231d8b3  %s\n' "$dlc" | sha256sum -c -
printf '8a1aa8926b3e6a99874d787d192175b74f4cccdf9a3be4cbc3f14f7478616ad2  %s\n' "$sdk/include/SNPE/SNPE/SNPEBuilder.h" | sha256sum -c -
g++ -std=c++17 -O2 -Wall -Wextra \
  -I"$sdk/include/SNPE" -I"$sdk/include/zdl" \
  native/snpe_daemon_candidate/snpe_daemon.cpp \
  -L"$sdk/lib/$abi" -Wl,-rpath,"$sdk/lib/$abi" -lSNPE -lpthread \
  -o "$candidate_root/build/snpe-daemon-yolo11s-v3" \
  > "$candidate_root/build/compile.log" 2>&1 || {
    tail -n 40 "$candidate_root/build/compile.log"
    exit 1
  }
sha256sum "$candidate_root/build/snpe-daemon-yolo11s-v3"
# Cache-off is the mode proven by the Edge246 same-binary A/B and60-frame smoke.
set +e
timeout 120 "$candidate_root/build/snpe-daemon-yolo11s-v3" --dlc "$dlc" \
  --graph yolo11s320_float --outputs boxes,scores --runtime dsp --perf burst \
  --no-init-cache --build-check > "$candidate_root/build/network_selector.log" 2>&1
network_exit=$?
set -e
printf 'NETWORK_SELECTOR_EXIT=%s\n' "$network_exit"
tail -n 20 "$candidate_root/build/network_selector.log"
test "$network_exit" -eq 0
