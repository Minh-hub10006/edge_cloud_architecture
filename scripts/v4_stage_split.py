"""Where does a frame's time go? Per stream of one v4 Edge run (8/11: 252's streams slow only when 246 runs too).

  python scripts/v4_stage_split.py RUN_DIR      (RUN_DIR from scripts/run_edge_v4.sh)

From the per-frame records the Edge Agent logs (cam<i>.jsonl, window["frames"]), medians / p95 in ms of:
  wait     capture -> the frame's worker started (queue on the Edge)
  encode   JPEG encode or NPU head
  rtt      request sent -> result back (Kafka there and back + Cloud)
  cloud    the Cloud's own handling time (cloud_handling_ms)
  wire     rtt - cloud (Kafka both ways + result delivery to this process)
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np


def frames(path):
    for line in open(path, errors="replace"):
        if line.startswith("{"):
            try:
                yield from json.loads(line).get("frames") or []
            except ValueError:
                continue


def q(values):
    return f"{np.percentile(values, 50):6.0f} /{np.percentile(values, 95):6.0f}" if values else "     - /     -"


def main():
    out = Path(sys.argv[1])
    spec = (out / "spec.txt").read_text().split()
    print(f"{'stream':7s} {'video:mode':11s} {'frames':>6s}  {'wait p50/p95':>14s} {'encode':>14s} {'rtt':>14s} "
          f"{'cloud':>14s} {'wire':>14s}  status")
    for i, item in enumerate(spec, 1):
        fs = list(frames(out / f"cam{i}.jsonl"))
        ok = [f for f in fs if f.get("status") == "accepted"]
        get = lambda key: [f[key] for f in ok if isinstance(f.get(key), (int, float))]
        wait = [(f["worker_start_mono_ns"] - f["capture_mono_ns"]) / 1e6 for f in ok
                if f.get("worker_start_mono_ns") and f.get("capture_mono_ns")]
        enc = get("jpeg_encode_ms") or [f.get("head_npu_ms", 0) + f.get("head_prepare_ms", 0) + f.get("head_pack_ms", 0)
                                        for f in ok if "head_npu_ms" in f]
        rtt, cloud = get("request_round_trip_ms"), get("cloud_handling_ms")
        wire = [f["request_round_trip_ms"] - f["cloud_handling_ms"] for f in ok
                if isinstance(f.get("request_round_trip_ms"), (int, float)) and isinstance(f.get("cloud_handling_ms"), (int, float))]
        status = dict(collections.Counter(f.get("status") for f in fs).most_common(3))
        print(f"cam-{i:<3d} {item:11s} {len(ok):6d}  {q(wait)} {q(enc)} {q(rtt)} {q(cloud)} {q(wire)}  {status}")


if __name__ == "__main__":
    main()
