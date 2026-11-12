#!/usr/bin/env bash
# Make every analytics test video frame-identical to its main stream (13/11).
#   bash scripts/fix_test_videos.sh HOST          # 246 | 252 | 247 (Git Bash)
# Boxes are computed on ~/test_videos/<v>.mp4 and drawn on <v>_main.mp4 by frame number (identity band). On 12/11 the
# analytics files were a 359-frame (24 s) cut while the main files have 2100 frames (140 s): after 24 s every box was
# drawn on another moment of the video. Each <v>.mp4 whose frame count differs from <v>_main.mp4 is rebuilt from it
# (band cropped: 1080x1920, H.264 main profile, no B-frames, 15 fps); the old file is kept in orig_<date>/.
set -uo pipefail
host="${1:?HOST 246|252|247}"
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1090
source "$here/scripts/hosts/edge-$host.sh"
dir="$EDGE_HOME/test_videos"
ff="${FFMPEG:-ffmpeg}"
command -v "$ff" >/dev/null || ff="$EDGE_HOME/ffmpeg/bin/ffmpeg.exe"   # 247
[[ -x "$ff" ]] || command -v "$ff" >/dev/null || { echo "no ffmpeg found (set FFMPEG=...)" >&2; exit 2; }
frames() { "$EDGE_PYTHON_DEFAULT" -c "import cv2,sys; c=cv2.VideoCapture(sys.argv[1]); print(int(c.get(7)))" "$1"; }
backup="$dir/orig_$(date +%Y%m%d)"
for main in "$dir"/*_main.mp4; do
  v="$(basename "$main" _main.mp4)"; a="$dir/$v.mp4"
  nm=$(frames "$main"); na=$([[ -f "$a" ]] && frames "$a" || echo 0)
  if [[ "$na" == "$nm" ]]; then echo "$v: $na = $nm frames, ok"; continue; fi
  echo "$v: analytics $na frames vs main $nm -> rebuilding from the main stream (a few minutes)"
  mkdir -p "$backup"; [[ -f "$a" ]] && cp -n "$a" "$backup/"
  "$ff" -hide_banner -loglevel error -y -i "$main" -vf "crop=1080:1920:0:0" -c:v libx264 -profile:v main \
    -preset veryfast -crf 18 -bf 0 -g 30 -pix_fmt yuv420p -r 15 -an "$dir/${v}_new.mp4" \
    && mv "$dir/${v}_new.mp4" "$a" || { echo "!! $v: ffmpeg failed, old file left in place" >&2; continue; }
  echo "$v: now $(frames "$a") frames (main $nm)"
done
