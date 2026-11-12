#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
export CLOUD_TRANSPORT=kafka
export KAFKA_BOOTSTRAP_SERVERS="${KAFKA_BOOTSTRAP_SERVERS:-192.168.180.95:9092}"
export DFINE_FULL_ONNX="${DFINE_FULL_ONNX:-$HOME/dfine_cloud_cuts/dfine_full.onnx}"
export CLOUD_RTSP_BASE="${CLOUD_RTSP_BASE:-rtsp://127.0.0.1:8554}"
export CLOUD_MEDIAMTX_WEBRTC_BASE="${CLOUD_MEDIAMTX_WEBRTC_BASE:-http://192.168.180.249:8889}"
export CLOUD_PREVIEW_FPS="${CLOUD_PREVIEW_FPS:-15}"
export DFINE_TOP_K="${DFINE_TOP_K:-20}"
export DFINE_CONF_THR="${DFINE_CONF_THR:-0.10}"
export DFINE_BOX_FORMAT="${DFINE_BOX_FORMAT:-cxcywh}"
CLOUD_PYTHON="${CLOUD_PYTHON:-$HOME/miniconda3/envs/qnn/bin/python}"
"$CLOUD_PYTHON" -c 'import socket,sys; from urllib.parse import urlparse; import os; u=urlparse(os.environ["CLOUD_RTSP_BASE"]); socket.create_connection((u.hostname,u.port or 554),timeout=3).close(); print("Cloud Python:",sys.executable,"RTSP port reachable",flush=True)'
exec "$CLOUD_PYTHON" -m uvicorn web.cloud_app:create_app --factory --host 0.0.0.0 --port "${CLOUD_HTTP_PORT:-8003}"
