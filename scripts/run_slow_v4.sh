#!/usr/bin/env bash
# Cloud slow-lane worker on Cloud249 (P2/P3): UC3 crowd density, UC4 fight / crash checks (cloud_core/slow_worker.py).
#   bash scripts/run_slow_v4.sh start [PROCESSES=1]   # same consumer group: more processes share the load
#   bash scripts/run_slow_v4.sh stop
# Needs Triton (scripts/triton_config_v4.sh add-crowd) and the qnn env (tritonclient, aiokafka, cv2) like the workers.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
py="${CLOUD_PY:-$HOME/miniconda3/envs/qnn/bin/python}"
export PYTHONPATH="$here:$HOME/v4_pydeps" KAFKA_BOOTSTRAP_SERVERS=localhost:29092 TRITON_URL=localhost:18001 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
case "${1:?start|stop}" in
  start)
    pgrep -f "v4-slow-worker" >/dev/null && { echo "already running; stop first" >&2; exit 2; }
    for t in edge.crowd-frame.v1 cloud.crowd.v1 edge.event.v1 cloud.slow-feedback.v1 edge.action-clip.v1 edge.crash-frame.v1 cloud.action.v1; do
      docker exec v4-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:29092 --create --if-not-exists \
        --topic "$t" --partitions 4 --config retention.ms=60000 --config max.message.bytes=2097152 2>&1 | grep -v WARNING || true
    done
    out="$HOME/v4_runs/slow_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$out"
    for i in $(seq 1 "${2:-1}"); do
      nohup bash -c 'exec -a v4-slow-worker-'"$i"' "$0" -B -m cloud_core.slow_worker' "$py" > "$out/worker_$i.log" 2>&1 &
    done
    sleep 8
    tail -3 "$out"/worker_*.log | cut -c1-200
    pgrep -af "v4-slow-worker" | cut -c1-80
    echo "logs: $out"
    ;;
  stop)
    pkill -f "v4-slow-worker" && echo "stopped" || echo "not running"
    ;;
  *) echo "start|stop" >&2; exit 2 ;;
esac
