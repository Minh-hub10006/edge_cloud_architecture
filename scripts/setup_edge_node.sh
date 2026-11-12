#!/usr/bin/env bash
# Prepare a QCS6490 Edge for live.sh/bench.sh, using this host's scripts/hosts/<node>.sh.
#   bash scripts/setup_edge_node.sh check    # read-only: report what is present/missing
#   bash scripts/setup_edge_node.sh build    # build both SNPE daemons for this host's ABI
# Python env, DLCs and the clip are copied/created by the operator (see output hints);
# this script never installs system packages.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/models.sh

CLIP_SHA=59a29f8913fc91a1ae9a835ffec182967f34204d3dee89c00c98e050f97b4b91
LIB="$QAIRT_ROOT_DEFAULT/lib/$QAIRT_ABI"
missing=0

row() { printf '%-8s %-28s %s\n' "$1" "$2" "$3"; }
need() { row MISSING "$1" "$2"; missing=1; }
sha_ok() { [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }

check() {
  echo "== $NODE_ID ($(hostname)), ABI $QAIRT_ABI"
  [ "$(uname -m)" = aarch64 ] && row OK arch aarch64 || need arch "$(uname -m)"
  [ -f "$QAIRT_ROOT_DEFAULT/include/SNPE/SNPE/SNPEBuilder.h" ] && [ -f "$LIB/libSNPE.so" ] \
    && row OK qairt "$LIB" || need qairt "$LIB/libSNPE.so"
  command -v g++ >/dev/null && row OK g++ "$(g++ -dumpversion)" || need g++ "install build-essential"
  command -v ffmpeg >/dev/null && row OK ffmpeg "$(command -v ffmpeg)" || need ffmpeg "install ffmpeg"
  sha_ok "$YOLOV8N_DLC" "$YOLOV8N_DLC_SHA" && row OK yolov8n-dlc "$YOLOV8N_DLC" || need yolov8n-dlc "copy to $YOLOV8N_DLC"
  sha_ok "$YOLO11S_DLC" "$YOLO11S_DLC_SHA" && row OK yolo11s-dlc "$YOLO11S_DLC" || need yolo11s-dlc "copy to $YOLO11S_DLC"
  sha_ok "$VIDEO" "$CLIP_SHA" && row OK clip "$VIDEO" || need clip "copy to $VIDEO"
  [ -x "$SNPE_DAEMON_BIN_DEFAULT" ] && row OK daemon-yolov8n "$SNPE_DAEMON_BIN_DEFAULT" || need daemon-yolov8n "run: setup_edge_node.sh build"
  [ -x "$YOLO11S_DAEMON_BIN" ] && row OK daemon-yolo11s "$YOLO11S_DAEMON_BIN" || need daemon-yolo11s "run: setup_edge_node.sh build"
  if [ -x "$EDGE_PYTHON_DEFAULT" ]; then
    local mods
    mods=$("$EDGE_PYTHON_DEFAULT" - <<'PY'
import importlib.util
print(" ".join(m for m in ("numpy", "cv2", "grpc", "google.protobuf", "psutil", "prometheus_client", "aiokafka")
               if importlib.util.find_spec(m) is None))
PY
)
    [ -z "$mods" ] && row OK python "$EDGE_PYTHON_DEFAULT" || need python-modules "$mods"
    # Decoded pixels differ across OpenCV builds (cv2 5.0.0 vs 4.13.0 on the same clip), which
    # breaks cross-Edge parity; every Edge must match Edge246.
    local cv; cv=$("$EDGE_PYTHON_DEFAULT" -c "import cv2; print(cv2.__version__)" 2>/dev/null || true)
    [ "$cv" = "$OPENCV_VERSION" ] && row OK opencv "$cv" \
      || need opencv "$cv (pip install opencv-python==$OPENCV_PIP opencv-python-headless==$OPENCV_PIP)"
  else
    need python "$EDGE_PYTHON_DEFAULT (conda create -n tracking python=3.10)"
  fi
  for port in "$RTSP_PORT" 50071 50073; do
    port_open "$CLOUD_HOST" "$port" && row OK "cloud:$port" reachable || row INFO "cloud:$port" "closed (start the matching Cloud receiver first)"
  done
  [ "$missing" = 0 ] && echo "READY" || echo "NOT READY"
  return "$missing"
}

build() {
  [ -f "$LIB/libSNPE.so" ] || fail SETUP "QAIRT libs missing: $LIB"
  sha_ok "$YOLO11S_DLC" "$YOLO11S_DLC_SHA" || fail SETUP "copy the pinned YOLO11s DLC first: $YOLO11S_DLC"
  mkdir -p "$(dirname "$SNPE_DAEMON_BIN_DEFAULT")" "$(dirname "$YOLO11S_DAEMON_BIN")"
  local work; work="$(mktemp -d)"
  # Baseline daemon (YOLOv8n): native/snpe_daemon source, same flags as its Makefile, this host's ABI.
  if [ -e "$SNPE_DAEMON_BIN_DEFAULT" ]; then echo "keep existing $SNPE_DAEMON_BIN_DEFAULT"; else
  g++ -std=c++17 -O2 -Wall -Wextra -I"$QAIRT_ROOT_DEFAULT/include/zdl" -I"$QAIRT_ROOT_DEFAULT/include/SNPE" \
    native/snpe_daemon/snpe_daemon.cpp -L"$LIB" -Wl,-rpath,"$LIB" -lSNPE -lpthread \
    -o "$SNPE_DAEMON_BIN_DEFAULT" >"$work/baseline.log" 2>&1 || { tail -n 30 "$work/baseline.log"; fail SETUP "baseline daemon build"; }
  fi
  # YOLO11s cache-off V3 daemon: same flags as scripts/build_yolo11s_daemon_candidate_246.sh.
  if [ -e "$YOLO11S_DAEMON_BIN" ]; then echo "keep existing $YOLO11S_DAEMON_BIN"; else
  g++ -std=c++17 -O2 -Wall -Wextra -I"$QAIRT_ROOT_DEFAULT/include/SNPE" -I"$QAIRT_ROOT_DEFAULT/include/zdl" \
    native/snpe_daemon_candidate/snpe_daemon.cpp -L"$LIB" -Wl,-rpath,"$LIB" -lSNPE -lpthread \
    -o "$YOLO11S_DAEMON_BIN" >"$work/yolo11s.log" 2>&1 || { tail -n 30 "$work/yolo11s.log"; fail SETUP "yolo11s daemon build"; }
  fi
  sha256sum "$SNPE_DAEMON_BIN_DEFAULT" "$YOLO11S_DAEMON_BIN"
  # Load both pinned graphs on the DSP once (cache off), as proven on Edge246.
  export LD_LIBRARY_PATH="$LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export ADSP_LIBRARY_PATH="$QAIRT_ROOT_DEFAULT/lib/hexagon-v68/unsigned;$LIB;/usr/lib/rfsa/adsp;/dsp;/opt"
  set +e
  timeout 120 "$YOLO11S_DAEMON_BIN" --dlc "$YOLO11S_DLC" --graph yolo11s320_float --outputs boxes,scores \
    --runtime dsp --perf burst --no-init-cache --build-check >"$work/build_check.log" 2>&1
  local code=$?
  # Same SNPE build for the YOLOv8n DLC (graph yolov8n_320_split, outputs boxes,scores).
  if [ "$code" = 0 ]; then
    timeout 120 "$YOLO11S_DAEMON_BIN" --dlc "$YOLOV8N_DLC" --graph yolov8n_320_split --outputs boxes,scores \
      --runtime dsp --perf burst --no-init-cache --build-check >>"$work/build_check.log" 2>&1
    code=$?
  fi
  set -e
  tail -n 8 "$work/build_check.log"
  [ "$code" = 0 ] && gate SETUP "PASS daemons built; YOLO11s+YOLOv8n DSP build-check OK (logs $work)" \
    || fail SETUP "DSP build-check exit $code (log $work/build_check.log)"
}

case "${1:-check}" in
  check) check ;;
  build) build ;;
  *) echo "Usage: bash scripts/setup_edge_node.sh check|build" >&2; exit 2 ;;
esac
