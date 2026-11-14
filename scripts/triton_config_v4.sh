#!/usr/bin/env bash
# Triton on Cloud249 for the v4 lanes (P1/P2, 10/11).
#   bash scripts/triton_config_v4.sh show                  # every model's config.pbtxt + how the server runs (read only)
#   bash scripts/triton_config_v4.sh queue                 # back up, bounded queues on the fast-lane models, restart
#   bash scripts/triton_config_v4.sh add-crowd MODEL.onnx  # UC3 DM-Count as model crowd_density (slow lane), restart
#   bash scripts/triton_config_v4.sh add NAME MODEL.onnx   # any slow-lane model listed in SLOW (P3: pose_rtmo_s,
#                                                          # action_stgcnpp), restart
#   bash scripts/triton_config_v4.sh ratelimit             # recreate the server with the rate limiter: fast lane
#                                                          # priority 1, slow lane 4, 4 executions at a time on the GPU
#   bash scripts/triton_config_v4.sh no-ratelimit          # back to the server without it (same configs otherwise)
#   bash scripts/triton_config_v4.sh restore DIR           # put a config backup back, restart
# After any Triton restart the Cloud workers must restart too (shared-memory regions were registered with the old
# server): scripts/run_cloud_v4.sh stop; RTDETR_SPLIT_HEAD_ID=... scripts/run_cloud_v4.sh start 4 (and run_slow_v4.sh).
set -uo pipefail
repo="${TRITON_REPO:-$HOME/triton_repo_20261105}"
container=v4-triton
image=nvcr.io/nvidia/tritonserver:25.08-py3
# 12/11, 3 Edges x 4 cams: D-FINE-X (1 instance, batch ~1.02, ~15 ms) got ~50 % of D frames while the GPU averaged 18 %:
# a second instance, and 6 slots (was 4: 5 fast-lane instances were already sharing them).
FAST="rtdetr_full:2 rtdetr_tail_q:2 rtdetr_tail_u8:2 dfinex_full:2 rtdetr_tail:1"   # tail_q: CUT2 wire (14/11)
# name:max batch:max queue delay us (P3 adds pose / action). 10/11 level 3: a DM-Count batch of 4 at 1280 holds the GPU
# ~140 ms and, sharing slots with the fast lane, cut B/C/D to 46 % of frames at ~295 ms; batch 1 (~35 ms) and taking
# every slot (it runs only on an idle GPU, the fast lane waits at most one such run) is the next try.
# 11/11 P3: RTMO-s (640, ~4x lighter than DM-Count at 1280) batched up to 4, waiting at most 5 ms - to check by measurement.
# ST-GCN++ (2x100x17x3 skeletons, ~1.4 M parameters) is tiny: batch up to 8.
SLOW="crowd_density:1:0 pose_rtmo_s:4:5000 action_stgcnpp:8:5000 plate_ocr:64:2000"   # plate_ocr: UC1 text lines (13/11)
SLOTS=6                          # executions at a time on the GPU under the rate limiter (4 -> 6 on 12/11, see FAST)

ready() {   # wait until Triton answers and lists every model READY (TensorRT engines build at load: up to ~15 min)
  for _ in $(seq 1 450); do
    curl -sf localhost:18000/v2/health/ready >/dev/null && break
    # a model that failed to load stays UNAVAILABLE: stop waiting and show why (12/11: waited 15 min for nothing)
    curl -s -X POST localhost:18000/v2/repository/index | grep -q UNAVAILABLE && break
    sleep 2
  done
  curl -sf -X POST localhost:18000/v2/repository/index | tr '{' '\n' | grep -o '"name":"[^"]*","version":"[^"]*","state":"[^"]*"' \
    || { echo "Triton not ready" >&2; return 1; }
}

limiter() {   # priority [slots] -> the instance's rate_limiter block when the server runs with it, else nothing
  [[ "${RL:-0}" == 1 ]] && echo " rate_limiter { resources [ { name: \"gpu_exec\" global: true count: ${2:-1} } ] priority: $1 }"
}

fast_config() {   # name count: fast lane, batch 4 / 8 ms (5/11), bounded queue (P1)
  cat <<EOF
# v4 fast lane (5/11 batch up to 4 frames from any camera, wait at most 8 ms; 10/11 P1 bounded queue: at most 64
# requests waiting, each at most 1 s = the frame's deadline, then rejected instead of served late).
name: "$1"
backend: "onnxruntime"
max_batch_size: 4
dynamic_batching {
  preferred_batch_size: [ 4 ]
  max_queue_delay_microseconds: 8000
  default_queue_policy { max_queue_size: 64 default_timeout_microseconds: 1000000 timeout_action: REJECT allow_timeout_override: true }
}
instance_group [ { count: $2 kind: KIND_GPU gpus: [ 0 ]$(limiter 1) } ]
$(trt_block "$1")
EOF
}

# 12/11: under 3 Edges x 4 cams each image took 3-5x its alone time (D-FINE-X ~47 ms, RT-DETR ~21 ms) on ONNX Runtime
# FP32. TRT_MODELS run through ONNX Runtime's TensorRT provider in FP16: one engine for batch 1-4 (profile shapes),
# built at load by a batch-4 warmup (not on the first live frame), cached on the host in $TRT_CACHE (mounted at
# /trt_cache; 12/11 ORT would not create /tmp/trt_cache/<model> itself). Gate: scripts/triton_check.py "Triton GPU
# vs ORT CPU" >= 0.99 per model, else `trt-off`.
TRT_CACHE="${TRT_CACHE:-$HOME/triton_trt_cache}"
# 12/11 FP16 on 249: RT-DETR 674 img/s (ORT FP32 b4 ~313), D-FINE-X 365 (~125), but box agreement vs ORT CPU 0.889 /
# 0.947 (gate 0.99, b4 also differed from b1): FAIL. TRT_PRECISION=FP32 (TF32 kernels, fused graph) is the next try.
TRT_PRECISION="${TRT_PRECISION:-FP32}"
# Unset = the models whose config has TensorRT now (13/11: add / queue rewrote every config with TRT_MODELS empty, so
# adding a model silently put RT-DETR and D-FINE-X back on ONNX Runtime). trt-off sets it to "" explicitly.
TRT_MODELS="${TRT_MODELS-$(grep -l '"tensorrt"' "$repo"/*/config.pbtxt 2>/dev/null | xargs -r -n1 dirname | xargs -r -n1 basename | tr '\n' ' ')}"
trt_block() {   # name -> optimization + warmup for a model listed in TRT_MODELS
  [[ " $TRT_MODELS " == *" $1 "* ]] || return 0
  local lo opt hi warm
  if [[ $1 == crowd_density ]]; then   # DM-Count: one 1280x1280 image (batch 1, 13/11)
    lo="image:1x3x1280x1280"; opt="$lo"; hi="$lo"
    warm='model_warmup [ { name: "batch1" batch_size: 1
  inputs { key: "image" value: { data_type: TYPE_FP32 dims: [ 3, 1280, 1280 ] zero_data: true } } } ]'
  else                                 # RT-DETR / D-FINE-X: images + orig_target_sizes, batch 1-4
    lo="images:1x3x640x640,orig_target_sizes:1x2"; opt="images:4x3x640x640,orig_target_sizes:4x2"; hi="$opt"
    warm='model_warmup [ { name: "batch4" batch_size: 4
  inputs { key: "images" value: { data_type: TYPE_FP32 dims: [ 3, 640, 640 ] zero_data: true } }
  inputs { key: "orig_target_sizes" value: { data_type: TYPE_INT64 dims: [ 2 ] zero_data: true } } } ]'
  fi
  cat <<EOT
optimization { execution_accelerators { gpu_execution_accelerator : [ { name : "tensorrt"
  parameters { key: "precision_mode" value: "$TRT_PRECISION" }
  parameters { key: "max_workspace_size_bytes" value: "4294967296" }
  parameters { key: "trt_engine_cache_enable" value: "1" }
  parameters { key: "trt_engine_cache_path" value: "/trt_cache/$1_$TRT_PRECISION" }
  parameters { key: "trt_profile_min_shapes" value: "$lo" }
  parameters { key: "trt_profile_opt_shapes" value: "$opt" }
  parameters { key: "trt_profile_max_shapes" value: "$hi" } } ] } }
$warm
EOT
}

slow_config() {   # name max_batch delay_us: slow lane, one instance, long waits allowed (P2)
  cat <<EOF
# v4 slow lane (10/11 P2): batches from every camera, waits up to $(( $3 / 1000 )) ms to fill them; at most 256 waiting,
# each at most 10 s (the Edge's deadline); one instance, low priority under the rate limiter.
name: "$1"
backend: "onnxruntime"
max_batch_size: $2
dynamic_batching {
  max_queue_delay_microseconds: $3
  default_queue_policy { max_queue_size: 256 default_timeout_microseconds: 10000000 timeout_action: REJECT allow_timeout_override: true }
}
instance_group [ { count: 1 kind: KIND_GPU gpus: [ 0 ]$(limiter 4 $SLOTS) } ]
$(trt_block "$1")
EOF
}

backup() {
  local b="$HOME/triton_config_backup_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$b"
  for d in "$repo"/*/; do mkdir -p "$b/$(basename "$d")"; cp "$d/config.pbtxt" "$b/$(basename "$d")/"; done
  echo "backup: $b"
}

write_configs() {
  for m in $FAST; do
    [[ -d "$repo/${m%%:*}" ]] && fast_config "${m%%:*}" "${m##*:}" > "$repo/${m%%:*}/config.pbtxt"
  done
  for m in $SLOW; do
    IFS=: read -r name batch delay <<< "$m"
    [[ -d "$repo/$name" ]] && slow_config "$name" "$batch" "$delay" > "$repo/$name/config.pbtxt"
  done
}

recreate() {   # $@: extra tritonserver flags. Same image, ports, mount, GPU and IPC as the container of 5/11.
  docker stop "$container" >/dev/null && docker rm "$container" >/dev/null
  for m in $TRT_MODELS; do mkdir -p "$TRT_CACHE/${m}_$TRT_PRECISION"; done
  mkdir -p "$TRT_CACHE"
  docker run -d --name "$container" --restart unless-stopped --gpus '"device=0"' --ipc=host --shm-size 64m \
    -p 18000:8000 -p 18001:8001 -p 18002:8002 -v "$repo":/models:ro -v "$TRT_CACHE":/trt_cache "$image" \
    tritonserver --model-repository=/models "$@" >/dev/null
  ready
}

rate_limited() { docker inspect "$container" --format '{{json .Config.Cmd}}' | grep -q -- "--rate-limit=execution_count"; }

case "${1:?show|queue|add-crowd ONNX|add NAME ONNX|trt [MODELS]|trt-off|ratelimit|no-ratelimit|restore DIR}" in
  show)
    for d in "$repo"/*/; do echo "=== $d"; cat "$d/config.pbtxt"; done
    docker inspect "$container" --format '{{json .Config.Cmd}}'
    ;;
  queue)
    backup
    RL=$(rate_limited && echo 1 || echo 0) write_configs
    docker restart "$container" >/dev/null && ready || { echo "!! Triton did not come back: see the backup above" >&2; exit 1; }
    ;;
  add-crowd)
    onnx="${2:?DM-Count ONNX (scripts/crowd_eval.py onnx ... --size 1280)}"
    backup
    mkdir -p "$repo/crowd_density/1" && cp "$onnx" "$repo/crowd_density/1/model.onnx"
    sha256sum "$repo/crowd_density/1/model.onnx" | cut -c1-16
    RL=$(rate_limited && echo 1 || echo 0) write_configs
    docker restart "$container" >/dev/null && ready
    ;;
  add)
    name="${2:?model name listed in SLOW or FAST}"; onnx="${3:?MODEL.onnx}"
    [[ " $SLOW $FAST " == *" $name:"* ]] || { echo "$name is not in SLOW or FAST" >&2; exit 2; }
    backup
    mkdir -p "$repo/$name/1" && cp "$onnx" "$repo/$name/1/model.onnx"
    sha256sum "$repo/$name/1/model.onnx" | cut -c1-16
    RL=$(rate_limited && echo 1 || echo 0) write_configs
    docker restart "$container" >/dev/null && ready
    ;;
  trt)   # TensorRT ($TRT_PRECISION, default FP32) for the fast-lane models given (default D-FINE-X + RT-DETR)
    export TRT_MODELS="${2:-dfinex_full rtdetr_full}"
    backup
    RL=1 write_configs
    echo "recreating $container: TensorRT $TRT_PRECISION for $TRT_MODELS (engines build now, several minutes)"
    recreate --rate-limit=execution_count --rate-limit-resource=gpu_exec:$SLOTS
    ;;
  trt-off)   # back to ONNX Runtime FP32 for every model
    backup
    TRT_MODELS="" RL=1 write_configs
    recreate --rate-limit=execution_count --rate-limit-resource=gpu_exec:$SLOTS
    ;;
  ratelimit)
    backup
    RL=1 write_configs
    echo "recreating $container with --rate-limit=execution_count --rate-limit-resource=gpu_exec:$SLOTS"
    recreate --rate-limit=execution_count --rate-limit-resource=gpu_exec:$SLOTS
    ;;
  no-ratelimit)
    backup
    RL=0 write_configs
    recreate
    ;;
  restore)
    src="${2:?backup dir}"
    for d in "$src"/*/; do cp "$d/config.pbtxt" "$repo/$(basename "$d")/config.pbtxt"; done
    docker restart "$container" >/dev/null && ready
    ;;
  *) echo "show|queue|add-crowd ONNX|add NAME ONNX|trt [MODELS]|trt-off|ratelimit|no-ratelimit|restore DIR" >&2; exit 2 ;;
esac
