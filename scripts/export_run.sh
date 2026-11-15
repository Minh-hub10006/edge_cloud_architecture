#!/usr/bin/env bash
# One Edge's results of the last MINUTES, for the content checks on the laptop (13/11). Read only. Run on 249:
#   bash scripts/export_run.sh edge-252 10        -> ~/run_export_edge-252.tgz
# Holds frames / detections / counts / events / plates / crowd / actions as CSV, plus the plate crops and event snapshots they
# name (at most 60 / 30). Laptop: python -B scripts/verify_db_export.py <unpacked dir> --videos DIR --map cam-1=vn9 ...
set -euo pipefail
node="${1:?NODE e.g. edge-252}"; minutes="${2:-10}"
out="$(mktemp -d)/run_export_$node"; mkdir -p "$out/media"
q() { docker exec v4-tsdb psql -U postgres -d v4 -c "\\copy ($1) TO STDOUT CSV HEADER" > "$out/$2.csv"; echo "$2: $(($(wc -l < "$out/$2.csv") - 1)) rows"; }
w="node_id='$node' AND"
q "SELECT camera_id, stream_epoch, frame_id, media_pts_ns, model_mode, track_id, class_name, x1, y1, x2, y2 FROM detections WHERE $w capture_time > now() - interval '$minutes minutes'" detections
q "SELECT camera_id, stream_epoch, frame_id, media_pts_ns, model_id, model_mode, box_count, latency_ms, extract(epoch from capture_time) * 1000 AS cap_ms FROM frames WHERE $w capture_time > now() - interval '$minutes minutes'" frames
q "SELECT *, extract(epoch from window_end) * 1000 AS end_ms FROM counts WHERE $w window_end > now() - interval '$minutes minutes'" counts
q "SELECT *, extract(epoch from started_at) * 1000 AS start_ms FROM events WHERE $w started_at > now() - interval '$minutes minutes'" events
q "SELECT * FROM plates WHERE $w capture_time > now() - interval '$minutes minutes'" plates
q "SELECT * FROM crowd WHERE $w capture_time > now() - interval '$minutes minutes'" crowd
q "SELECT * FROM actions WHERE $w capture_time > now() - interval '$minutes minutes'" actions
copy() {  # the image paths named in one CSV column, at most $3 files
  # the limit inside python: 13/11 "| head" closed the pipe early, python failed and set -e stopped the export
  python3 -c "import csv,sys; [print(p) for p in [r[sys.argv[2]] for r in csv.DictReader(open(sys.argv[1])) if r[sys.argv[2]]][:int(sys.argv[3])]]" "$1" "$2" "$3" \
    | while read -r f; do if [[ -f "$f" ]]; then cp "$f" "$out/media/"; fi; done
}
copy "$out/plates.csv" image_path 60
copy "$out/events.csv" snapshot_path 30
echo "images: $(ls "$out/media" | wc -l)"
tar -czf ~/run_export_"$node".tgz -C "$(dirname "$out")" "$(basename "$out")" && rm -rf "$(dirname "$out")"
ls -la ~/run_export_"$node".tgz
