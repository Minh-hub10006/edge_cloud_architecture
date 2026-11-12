#!/usr/bin/env bash
# One Edge's results of the last MINUTES, for the content checks on the laptop (13/11). Read only. Run on 249:
#   bash scripts/export_run.sh edge-252 10        -> ~/run_export_edge-252.tgz
# Holds detections / counts / events / plates / crowd / actions as CSV, plus the plate crops and event snapshots they
# name (at most 60 / 30). Laptop: python -B scripts/verify_db_export.py <unpacked dir> --videos DIR --map cam-1=vn9 ...
set -euo pipefail
node="${1:?NODE e.g. edge-252}"; minutes="${2:-10}"
out="$(mktemp -d)/run_export_$node"; mkdir -p "$out/media"
q() { docker exec v4-tsdb psql -U postgres -d v4 -c "\\copy ($1) TO STDOUT CSV HEADER" > "$out/$2.csv"; echo "$2: $(($(wc -l < "$out/$2.csv") - 1)) rows"; }
w="node_id='$node' AND"
q "SELECT camera_id, stream_epoch, frame_id, media_pts_ns, model_mode, track_id, class_name, x1, y1, x2, y2 FROM detections WHERE $w capture_time > now() - interval '$minutes minutes'" detections
q "SELECT * FROM counts WHERE $w window_end > now() - interval '$minutes minutes'" counts
q "SELECT * FROM events WHERE $w started_at > now() - interval '$minutes minutes'" events
q "SELECT * FROM plates WHERE $w capture_time > now() - interval '$minutes minutes'" plates
q "SELECT * FROM crowd WHERE $w capture_time > now() - interval '$minutes minutes'" crowd
q "SELECT * FROM actions WHERE $w capture_time > now() - interval '$minutes minutes'" actions
copy() {  # the image paths named in one CSV column, at most $3 files
  python3 -c "import csv,sys; [print(r[sys.argv[2]]) for r in csv.DictReader(open(sys.argv[1])) if r[sys.argv[2]]]" "$1" "$2" \
    | head -n "$3" | while read -r f; do [[ -f "$f" ]] && cp "$f" "$out/media/"; done
}
copy "$out/plates.csv" image_path 60
copy "$out/events.csv" snapshot_path 30
echo "images: $(ls "$out/media" | wc -l)"
tar -czf ~/run_export_"$node".tgz -C "$(dirname "$out")" "$(basename "$out")" && rm -rf "$(dirname "$out")"
ls -la ~/run_export_"$node".tgz
