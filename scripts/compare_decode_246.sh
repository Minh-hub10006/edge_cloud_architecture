#!/usr/bin/env bash
# CPU decode vs hardware decode on Edge 246 (7/11), same 2 streams (vn9, vn6), path B, 2 minutes each, then a table.
# Needs: v4 Cloud workers + Kafka on 249, ~/test_videos/{vn9,vn6}{,_main}.mp4, ~/edge_b_20261106/head_w8a8_pc.dlc,
# PyAV in ~/v4_pydeps.   bash scripts/compare_decode_246.sh
set -uo pipefail
cd "$(dirname "$0")/.."
P=$HOME/miniconda3/envs/tracking/bin/python
export PYTHONPATH=.:$HOME/v4_pydeps QAIRT_ROOT=$HOME/sigm/qairt/2.43.0.260128 QAIRT_ABI=aarch64-oe-linux-gcc11.2 \
  SNPE_PERSISTENT=1 SNPE_PERSISTENT_MODE=daemon \
  YOLO11S_SNPE_DAEMON_BIN=$HOME/edge_video_platform_benchmark_overlap_20261004/yolo11s_integration_v3.9dxyacpe/build/snpe-daemon-yolo11s-v3
out=$HOME/v4_runs_edge/decode_$(date +%H%M%S)
mkdir -p "$out"

echo "[0/3] decoder device opens (no ENOMEM/EBUSY)?"
python3 -c "
import os, sys
try:
    os.close(os.open('/dev/video32', os.O_RDWR | os.O_NONBLOCK)); print('   /dev/video32 opens')
except OSError as e:
    print('   /dev/video32:', e.strerror, '-> stop: the decoder needs a reboot of the Edge'); sys.exit(1)
" || exit 1
echo "[1/3] the hardware reader gives OpenCV's pixels (plain, rewind, start frame)"
$P -B - <<'EOF'
import os, cv2, numpy as np
from media.hw_capture import HwCapture
path = os.path.expanduser("~/test_videos/vn9.mp4")
hw, cpu = HwCapture(path, decoder="v4l2h264dec"), cv2.VideoCapture(path)
print(f"   stride {hw.stride} for {hw.width}x{hw.height}")
def diff(a, b):
    return int(np.abs(a.astype(int) - b.astype(int)).max())
hw_frames = [hw.read()[1] for _ in range(40)]
cpu_frames = [cpu.read()[1] for _ in range(40)]
plain = [diff(a, b) for a, b in zip(hw_frames, cpu_frames)]
first_bad = next((k for k, d in enumerate(plain) if d), None)
print(f"   plain read, 40 frames: max diff {max(plain)}, first different frame {first_bad}")
if first_bad is not None:  # same pictures one or two frames apart? then the order is off, not the pixels
    for shift in (-2, -1, 1, 2):
        pairs = [(hw_frames[k], cpu_frames[k + shift]) for k in range(5, 35)]
        print(f"   hardware frame k vs OpenCV frame k{shift:+d}: max diff {max(diff(a, b) for a, b in pairs)}")
hw.set(cv2.CAP_PROP_POS_FRAMES, 0); cpu.set(cv2.CAP_PROP_POS_FRAMES, 0)
print(f"   after rewind: max diff {max(diff(hw.read()[1], cpu.read()[1]) for _ in range(5))}")
hw.set(cv2.CAP_PROP_POS_FRAMES, 150); cpu.set(cv2.CAP_PROP_POS_FRAMES, 150)
seek = [diff(hw.read()[1], cpu.read()[1]) for _ in range(5)]
near = [diff(hw.read()[1], cv2.VideoCapture(path).read()[1]) for _ in range(0)]
print(f"   after start frame 150: max diff {max(seek)}")
worst = max(max(plain), max(seek))
hw.release()
print(f"   max pixel difference over 42 frames: {worst} -> {'PASS' if worst == 0 else 'DIFFERENT'}")
EOF

run() {  # $1 = label, $2 = decoder ('' = CPU)
  pkill -f "camera-id cam-[12] --stream-id v4-dec"; sleep 2
  ss -ltn | grep -q ":8554 " || (MTX_WEBRTCADDITIONALHOSTS=192.168.180.246 nohup ~/edge_artifacts/mediamtx/mediamtx media/mediamtx.yml > "$out/mediamtx.log" 2>&1 &)
  local i=0
  for v in vn9 vn6; do
    i=$((i+1))
    $P -B -u -m edge_agent.control_v1 --video ~/test_videos/$v.mp4 --main-video ~/test_videos/${v}_main.mp4 ${2:+--decoder $2} \
      --loop-source --frames 1800 --mode rtdetr_split_cloud --policy observe --cloud-inflight 2 --target-fps 15 --fps-levels 15 \
      --preview-fps 15 --transport kafka --kafka-bootstrap 192.168.180.249:29092 --rtsp-base rtsp://127.0.0.1:8554 \
      --cloud-frame-source jpeg --node-id edge-246 --camera-id cam-$i --stream-id v4-dec \
      --cloud-model-ids '{"rtdetr_split_cloud":"rtdetrv2_r18vd_sp2_120e_coco+layer2_split"}' \
      --capability '{"yolov8n_fps":101.6,"yolo11s_fps":86.3,"encode_fps":66.6}' \
      --rtdetr-head-dlc ~/edge_b_20261106/head_w8a8_pc.dlc --metrics-port 0 --run-timeout-s 240 \
      --state-jsonl "$out/$1_cam$i.jsonl" > "$out/$1_cam$i.log" 2>&1 &
  done
  wait
  sleep 20  # cool down between runs
}
echo "[2/3] 2 streams, CPU decode (OpenCV), 2 min"; run cpu ""
echo "[3/3] 2 streams, hardware decode (v4l2h264dec + swscale POINT), 2 min"; run hw v4l2h264dec

$P - "$out" <<'EOF'
import json, re, sys
from pathlib import Path
import numpy as np
out = Path(sys.argv[1])
def summary(log):
    text = log.read_text(errors="replace")
    i = text.rfind('{\n  "control_v1"')
    if i < 0:
        return None, text[-300:]
    s = json.loads(text[i:text.index("\n}", i) + 2])
    return s, None
print(f"{'':6s} {'stream':6s} {'inferred':>9s} {'drops':>6s} {'errors':>7s} {'worker->boxes p50/p95 ms':>25s} {'capture->boxes p50/p95 ms':>26s} {'CPU %':>6s} {'temp C max':>10s}")
for label in ("cpu", "hw"):
    for cam in (1, 2):
        s, err = summary(out / f"{label}_cam{cam}.log")
        if s is None:
            print(f"{label:6s} cam-{cam}  FAILED: {err!r}")
            continue
        rows = [json.loads(l) for l in open(out / f"{label}_cam{cam}.jsonl") if l.strip()]
        ev = [e for r in rows for e in r.get("frames", []) if e.get("status") == "accepted"]
        act = [e["active_path_latency_ms"] for e in ev if "active_path_latency_ms" in e]
        src = [e["source_completion_latency_ms"] for e in ev if "source_completion_latency_ms" in e]
        cpu = [r["raw"]["cpu_total_pct"] for r in rows if isinstance(r.get("raw", {}).get("cpu_total_pct"), (int, float))]
        tmp = [r["raw"]["temp_c"] for r in rows if isinstance(r.get("raw", {}).get("temp_c"), (int, float))]
        st = s["stats"]
        p = lambda v: f"{np.percentile(v, 50):6.0f} / {np.percentile(v, 95):5.0f}" if v else "     -"
        print(f"{label:6s} cam-{cam}  {st.get('frames_inferred', 0):9d} {s.get('analytics_queue_dropped', 0):6d} "
              f"{sum(s.get('error_reasons', {}).values()):7d} {p(act):>25s} {p(src):>26s} "
              f"{np.median(cpu) if cpu else float('nan'):6.1f} {max(tmp) if tmp else float('nan'):10.1f}")
print("logs:", out)
EOF
echo "NPU daemons left: $(pgrep -fc snpe-daemon)"
