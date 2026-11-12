#!/usr/bin/env bash
# Test-video live validation: start one control.sh stream per model given, 5 s apart, 120 s each, then print each
# stream's verdict. Run from the repo folder on an Edge:  bash ~/run_test.sh dfinex dfinex yolo11s
# The Edge is recognised by its account: Windows user1 = 247 (clock skew -> TTL), root = 252, otherwise 246.
SHA=54c6291d356299a889a642050070d5853caee94ff25f6185e014835750646471
if [ -d /c/Users/user1 ]; then
  NODE=247 O=0 VIDEO=C:/Users/user1/detrac_test_30m.mp4
  s=$(( $(date -d "$(curl -sI http://192.168.180.249:8006/ | tr -d '\r' | sed -n 's/^[Dd]ate: //p')" +%s) - $(date +%s) ))
  export CONTROL_TICKET_TTL_S=$(( s > 1 ? s + 2 : 3 )).5
  echo "247 cham ${s}s -> TTL=$CONTROL_TICKET_TTL_S"
elif [ "$(id -u)" = 0 ]; then
  NODE=252 O=1500 VIDEO=$HOME/detrac_test_30m.mp4
else
  NODE=246 O=1150 VIDEO=$HOME/detrac_test_30m.mp4
fi
# RUN_VIDEO=vn: the VN-Hard test clip instead (vn2/5/6/9/11 at 720x1280 15 fps, looped to 2020 s; plate OCR load).
if [ "${RUN_VIDEO:-detrac}" = vn ]; then
  VIDEO="$(dirname "$VIDEO")/vn_test_30m.mp4" SHA=62193fa8021d3d8af9611d17b8c89fc5936788bdee513a70f622b5d3a74a86d7
fi
export CONTROL_VIDEO="$VIDEO" CONTROL_VIDEO_SHA=$SHA CONTROL_DURATION_S=120
ROOT=$PWD
source scripts/models.sh
# Our SNPE daemons only, matched by our DLC paths (other projects' daemons, e.g. plate_yolov8n, are untouched).
# They outlive each stream so streams can share them; leftovers hold the DSP ("DSP unavailable", Oct 16).
stop_our_daemons() { command -v pkill >/dev/null && pkill -f -- "--dlc ($YOLOV8N_DLC|$YOLO11S_DLC|${PLATE_DLC:-none})"; return 0; }
stop_our_daemons
# shortcut: the installed YOLOv8n daemon always runs with init cache and rewrote its DLC when 3 started at once
# (Oct 16); a read-only DLC stops that. Rebuild the daemon with a --no-init-cache flag to drop this.
[ "$NODE" != 247 ] && [ -w "$YOLOV8N_DLC" ] && chmod a-w "$YOLOV8N_DLC"
# v50 (CONTROL_FRAME_SOURCE=jpeg, default): one MediaMTX on this Edge for every stream of the run, stopped at the end.
if [ "${CONTROL_FRAME_SOURCE:-jpeg}" = jpeg ]; then
  MEDIAMTX_BIN=$EDGE_MEDIAMTX_BIN MEDIA_PUBLIC_HOST=$EDGE_IP ensure_mediamtx M0
  echo "MediaMTX: rtsp://127.0.0.1:$RTSP_PORT, web http://$EDGE_IP:$WEBRTC_PORT"
fi
echo "Edge$NODE: $# luong ($*), bat dau video tu ${O}s"
rm -f ~/r_*.log
i=0 pids=()
for m in "$@"; do
  i=$((i+1))
  # model@S: start S seconds into the video (another Edge's camera served here; 246 DSP takes one YOLO stream)
  # The test video is 1815 s: wrap so the seek stays inside it (252's 3rd stream asked for 1950 s, Oct 22).
  start=$(( (O + i * 150) % 1800 )); [ "${m#*@}" != "$m" ] && start=${m#*@}
  CONTROL_START_S=$start bash scripts/control.sh edge "${m%%@*}" > ~/r_$i.log 2>&1 &
  pids+=($!)
  sleep 5
done
wait "${pids[@]}"  # not a bare wait: MediaMTX is a child too
stop_our_daemons
tail -qn 1 ~/r_*.log
# frames_by_mode={} means every frame was rejected even when the run says PASS (247 clock skew, Oct 14).
bad=0
for f in ~/r_*.log; do
  grep -q '\[C3\] PASS' "$f" && ! grep -q 'frames_by_mode={}' "$f" || { bad=1; echo "HONG: $f"; }
done
[ "$bad" = 0 ] && echo "==> OK ca $i luong" || echo "==> CO LUONG HONG: bao lai"
