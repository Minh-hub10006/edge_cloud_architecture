#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD"
EDGE_PYTHON="${EDGE_PYTHON:-python}"
TEST_EPOCH="${TEST_EPOCH:-$(date +%s)}"
"$EDGE_PYTHON" -c 'import socket; socket.create_connection(("192.168.180.249",8554),timeout=3).close(); print("RTSP port reachable",flush=True)'
echo "Live URL: http://192.168.180.249:8003/v1/streams/edge-246/traffic-01/kafka-smoke/$TEST_EPOCH/live?fps=15"
exec "$EDGE_PYTHON" -m edge_agent.phase7 \
  --video="${TEST_VIDEO:-/home/ubuntu/edge_multimodel/videos/clips_1min/traffic_1min.mp4}" \
  --frames=900 --mode=dfine_cloud --node-id=edge-246 --camera-id=traffic-01 \
  --stream-id=kafka-smoke --stream-epoch="$TEST_EPOCH" --transport=kafka \
  --kafka-bootstrap=192.168.180.95:9092 --rtsp-base=rtsp://192.168.180.249:8554 \
  --width=1280 --height=720 --preview-fps=15 "$@"
