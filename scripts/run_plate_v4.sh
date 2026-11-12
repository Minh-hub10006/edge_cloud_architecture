#!/usr/bin/env bash
# UC1 plate OCR worker on Cloud249 (8/11): Kafka edge.plate-crop.v1 -> OCR -> cloud.plate-read.v1 (-> table plates).
#   bash scripts/run_plate_v4.sh check            # can the OCR Python import what it needs? (nothing is started)
#   bash scripts/run_plate_v4.sh start [WORKERS=16]   # Triton plate_ocr, batched (13/11); WORKERS = threads
#   PLATE_OCR_BACKEND=easyocr bash scripts/run_plate_v4.sh start 4   # old path: 4 EasyOCR processes, no batching
#   bash scripts/run_plate_v4.sh stop
# PLATE_OCR_PYTHON: a Python with easyocr + torch (v50: ~/miniconda3/bin/python); aiokafka may come from ~/v4_pydeps.
set -uo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
py="${PLATE_OCR_PYTHON:-$HOME/miniconda3/bin/python}"
home="${PLATE_OCR_HOME:-$HOME/plate_ocr_home}"
export PYTHONPATH="$here:$HOME/v4_pydeps" KAFKA_BOOTSTRAP_SERVERS=localhost:29092 OMP_NUM_THREADS=1
case "${1:?check|start|stop}" in
  check)
    "$py" - <<'EOF'
import importlib
for name in ("aiokafka", "easyocr", "torch", "cv2", "yaml", "PIL", "tritonclient.grpc"):
    try:
        m = importlib.import_module(name)
        print(f"ok {name} {getattr(m, '__version__', '')}")
    except Exception as exc:
        print(f"MISSING {name}: {exc}")
try:
    import torch
    print("cuda", torch.cuda.is_available())
except Exception:
    pass
EOF
    ls -d "$home/model" "$home/user_network" 2>&1
    ;;
  start)
    pgrep -f "v4-plate-ocr" >/dev/null && { echo "already running; stop first" >&2; exit 2; }
    for t in edge.plate-crop.v1 cloud.plate-read.v1; do
      docker exec v4-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:29092 --create --if-not-exists \
        --topic "$t" --partitions 4 --config retention.ms=60000 --config max.message.bytes=2097152 2>&1 | grep -v WARNING || true
    done
    out="$HOME/v4_runs/plate_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$out"
    backend="${PLATE_OCR_BACKEND:-triton}"
    workers="${2:-$([[ $backend == triton ]] && echo 16 || echo 4)}"
    nohup bash -c 'exec -a v4-plate-ocr "$0" -B -m cloud_core.plate_worker --model-home "$1" --workers "$2" --backend "$3"' \
      "$py" "$home" "$workers" "$backend" > "$out/worker.log" 2>&1 &
    sleep 25   # the OCR processes load their models first
    tail -3 "$out/worker.log" | cut -c1-200
    pgrep -af "v4-plate-ocr" | cut -c1-80
    echo "logs: $out"
    ;;
  stop)
    pkill -f "v4-plate-ocr" && echo "stopped" || echo "not running"
    ;;
  *) echo "check|start|stop" >&2; exit 2 ;;
esac
