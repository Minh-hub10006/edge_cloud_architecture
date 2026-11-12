#!/usr/bin/env bash
# Candidate only; no baseline overwrite or Cloud/CUDA dependency.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
integration_root="$PWD"
integration_python=/home/ubuntu/miniconda3/envs/tracking/bin/python
"$integration_python" -B scripts/verify_release_snapshot.py
export QAIRT_ROOT=/home/ubuntu/sigm/qairt/2.43.0.260128
export SNPE_PERSISTENT=1 SNPE_PERSISTENT_MODE=daemon
export ADSP_LIBRARY_PATH="$QAIRT_ROOT/lib/hexagon-v68/unsigned;$QAIRT_ROOT/lib/aarch64-oe-linux-gcc11.2;/usr/lib/rfsa/adsp;/dsp;/opt"
export LD_LIBRARY_PATH="$QAIRT_ROOT/lib/aarch64-oe-linux-gcc11.2:/home/ubuntu/miniconda3/envs/tracking/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$integration_root"
export YOLO11S_SNPE_DAEMON_BIN="$integration_root/build/snpe-daemon-yolo11s-v3"
bash scripts/build_yolo11s_daemon_candidate_246.sh
printf 'INTEGRATION_LANE=%s\n' "$integration_root"
exec "$integration_python" -B scripts/yolo11s_edge_smoke.py \
  --dlc /home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/npu_run/edge_packet/yolo11s320_w8a8_qcs6490.dlc \
  --video /home/ubuntu/edge_multimodel/videos/clips_1min/traffic_1min.mp4 \
  --out "$integration_root/adapter_smoke" --frames 60 --warmup 3 --allow-diagnostic
