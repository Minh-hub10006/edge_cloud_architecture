#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export CLOUD_BENCHMARK=1 CLOUD_CONTROL_V1=0
export CLOUD_TRANSPORT="${1:-grpc}"
export CLOUD_MODELS="${2:-${CLOUD_MODELS:-dfine}}"
export CLOUD_MAX_INFLIGHT="${CLOUD_MAX_INFLIGHT:-2}"
export CLOUD_GRPC_BIND=0.0.0.0:50071
export KAFKA_BOOTSTRAP_SERVERS=192.168.180.95:9092
unset CLOUD_KAFKA_NAMESPACE CLOUD_ENABLE_OVERLAY_SMOKE
case "$CLOUD_TRANSPORT" in
  grpc) ;;
  kafka) export CLOUD_KAFKA_NAMESPACE=benchmark ;;
  *) echo "Usage: bash $0 {grpc|kafka} [dfine,rtdetrv2_small,lwdetr_large]" >&2; exit 2 ;;
esac
export DFINE_FULL_ONNX="${DFINE_FULL_ONNX:-$HOME/dfine_cloud_cuts/dfine_full.onnx}"
export REQUIRE_CUDA=1 DFINE_TOP_K=20 DFINE_CONF_THR=0.10 DFINE_BOX_FORMAT=cxcywh
export CLOUD_RTSP_BASE=rtsp://127.0.0.1:8554
export CLOUD_FRAME_CACHE_TTL_S=30 CLOUD_PREVIEW_FPS=30
exec "$HOME/miniconda3/envs/qnn/bin/python" -m uvicorn \
  web.cloud_app:create_app --factory --host 0.0.0.0 --port 8004
