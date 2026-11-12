#!/usr/bin/env bash
# v4 Cloud inference workers on 249 (5/11): N copies of web/cloud_app.py on Kafka, models through Triton.
# Python: env qnn (has uvicorn/starlette/av/lap like v50; + tritonclient, zstandard); CLOUD_PY overrides.
# One Python process tops out near 350 frames/s (GIL), so N processes share the work: they join the same Kafka
# consumer groups, and each camera's key always lands in the same partition -> same worker (frame order and
# ByteTrack per camera stay in one process). Tickets must carry JPEG or a cut tensor (no RTSP cache in v4).
#   bash scripts/run_cloud_v4.sh start [N=4]     # creates the topics (N partitions) if missing, starts N workers
#   bash scripts/run_cloud_v4.sh stop
set -euo pipefail
cmd="${1:?start|stop}"
n="${2:-4}"
here="$(cd "$(dirname "$0")/.." && pwd)"
runs="$HOME/v4_runs"
if [[ "$cmd" == stop ]]; then
  pkill -f "v4-cloud-worker" || { echo "no worker running"; exit 0; }
  for _ in $(seq 1 30); do pgrep -f "v4-cloud-worker" >/dev/null || { echo "stopped"; exit 0; }; sleep 1; done
  echo "workers still running after 30 s" >&2
  exit 1
fi
[[ "$cmd" == start ]] || { echo "start|stop" >&2; exit 2; }
pgrep -f "v4-cloud-worker" >/dev/null && { echo "workers already running; stop first" >&2; exit 2; }
# plain topics + the controlv1 namespace the Edge Agent talks on (contracts/control_v1.py KAFKA_NAMESPACE)
for t in edge.media-frame-ref.v1 edge.detection-result.v1 edge.cloud-inference-ticket.v1 cloud.detection-result.v1          controlv1.edge.media-frame-ref.v1 controlv1.edge.detection-result.v1 controlv1.edge.cloud-inference-ticket.v1          controlv1.cloud.detection-result.v1 edge.tracked-result.v1 edge.count.v1 edge.event.v1 \
         edge.plate-crop.v1 cloud.plate-read.v1 cloud.slow-feedback.v1 edge.crowd-frame.v1 cloud.crowd.v1; do
  docker exec v4-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:29092 --create --if-not-exists \
    --topic "$t" --partitions "$n" --config retention.ms=60000 --config max.message.bytes=2097152 2>&1 | grep -v WARNING || true
done
# 8/11: each Edge reads only its own copy of the results (contracts/kafka_transport.node_results_topic). Created here so
# they get the short retention and exist before the first Edge subscribes; another node: V4_NODES="edge-246 ... edge-x".
for node in ${V4_NODES:-edge-246 edge-252 edge-247}; do
  for t in cloud.detection-result.v1 controlv1.cloud.detection-result.v1; do
    docker exec v4-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:29092 --create --if-not-exists \
      --topic "$t.$node" --partitions 1 --config retention.ms=60000 --config max.message.bytes=2097152 2>&1 | grep -v WARNING || true
  done
done
out="$runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out"
# ~/v4_pydeps: setuptools<81 for pkg_resources (tritonclient 2.41 needs it), kept out of env qnn on purpose
# 5/11: 4 workers burned 4-7 cores each with the GPU at 13 % (numeric libraries spawning a thread per core in every
# process); one compute thread per library per worker, the parallelism comes from the workers and Triton.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$here:$HOME/v4_pydeps" CLOUD_TRANSPORT=kafka KAFKA_BOOTSTRAP_SERVERS=localhost:29092 CLOUD_RUNTIME=triton CLOUD_CONTROL_V1=1 \
  TRITON_URL=localhost:18001 TRITON_REPO="$HOME/triton_repo_20261105" CLOUD_MAX_INFLIGHT=16 \
  CLOUD_MODELS="${CLOUD_MODELS:-rtdetrv2_small,dfine_x,rtdetr_split}" \
  RTDETRV2_SMALL_ONNX="$HOME/cloud_artifacts/rtdetrv2_s_dyn_20261105/rtdetrv2_s_640.onnx" RTDETRV2_ALLOW_IMPORTED_CANDIDATE=1 \
  DFINEX_ONNX="$HOME/cloud_artifacts/dfine_x_candidate_20261021/dfine_x_obj2coco.onnx" DFINEX_ALLOW_IMPORTED_CANDIDATE=1 \
  RTDETRV2_CONF_THR=0.5 DFINEX_CONF_THR=0.65 \
  RTDETR_SPLIT_HEAD_ID="${RTDETR_SPLIT_HEAD_ID:?set the Edge head id, e.g. rtl2-w8a8pc}"
for i in $(seq 1 "$n"); do
  # the argv tag lets stop/pgrep find only these workers
  CLOUD_APP_PORT=$((18100 + i)) nohup bash -c 'exec -a v4-cloud-worker-'"$i"' "$0" -B -m web.cloud_app' \
    "${CLOUD_PY:-$HOME/miniconda3/envs/qnn/bin/python}" > "$out/worker_$i.log" 2>&1 &
done
sleep 20
echo "logs: $out"
grep -l -E "Traceback|startup failed" "$out"/worker_*.log && echo "!! a worker failed, see the log" || echo "$n workers up"
pgrep -af "v4-cloud-worker" | cut -c1-80
