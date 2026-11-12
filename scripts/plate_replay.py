"""K emulated plate cameras against the Cloud plate OCR service: extra Cloud load without extra Edge streams.

Each emulated camera replays the send times of one recorded real stream (an Edge plate.jsonl: one "read" per crop
sent, at frame / --source-fps seconds) from a random point, looping, and sends a real plate crop resized to that
read's recorded size (so 1-/2-line plates, hence OCR calls, match). Same wire and in-flight bound as an Edge.

  python -B scripts/plate_replay.py --target 192.168.180.249:50062 --cameras 20 --seconds 125 \
      --logs "C:/Users/ADMIN/sweep_plate/sweep_plate_ws570-249/runs/control/20261030_14*/plate.jsonl" \
      --crops C:/Users/ADMIN/plate_ocr_home/crops --out C:/Users/ADMIN/sweep_plate/replay_K20.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contracts.plate_ocr_wire import PlateOcrClient  # noqa: E402


def load_schedules(pattern, source_fps=15.0, min_reads=50):
    """[(t_s, w, h)] per recorded stream with at least min_reads reads (a VN-like stream)."""
    out = []
    for path in sorted(glob.glob(pattern)):
        reads = [x for x in map(json.loads, open(path, encoding="utf-8")) if x.get("event") == "read"]
        if len(reads) >= min_reads:
            f0 = min(r["frame"] for r in reads)
            out.append(sorted(((r["frame"] - f0) / source_fps, r["w"], r["h"]) for r in reads))
    return out


def camera_events(schedule, offset, seconds):
    """Times (from 0) and sizes of one camera's reads over `seconds`, starting `offset` s into its looped schedule."""
    span = schedule[-1][0] + 1.0
    out, lap = [], 0
    while True:
        for t, w, h in schedule:
            at = t + lap * span - offset
            if at >= seconds:
                return out
            if at >= 0:
                out.append((at, w, h))
        lap += 1


async def run(client, schedules, crops, *, cameras, seconds, seed=0, max_inflight=16, log=None):
    import cv2
    rng = random.Random(seed)
    epoch = time.time_ns() // 1000
    stats = {"sent": 0, "replies": 0, "errors": 0, "dropped": 0, "rtt_ms": []}

    async def send(cam, n, jpeg, inflight):
        sent = time.time_ns()
        try:
            reply = await client.read({"stream": [f"emu-{cam:02d}", "plate", "replay", epoch], "track": n,
                                       "frame": n, "sent_utc_ns": sent}, jpeg)
        except Exception as exc:
            stats["errors"] += 1
            if log is not None:
                log.write(json.dumps({"cam": cam, "error": type(exc).__name__}) + "\n")
            return
        finally:
            inflight[0] -= 1
        rtt = (time.time_ns() - sent) / 1e6
        stats["replies"] += 1
        stats["rtt_ms"].append(rtt)
        if log is not None:
            log.write(json.dumps({"cam": cam, "n": n, "rtt_ms": rtt, **reply}) + "\n")

    async def camera(cam):
        sched = schedules[cam % len(schedules)]
        events = camera_events(sched, rng.uniform(0, sched[-1][0] + 1.0), seconds)
        inflight, tasks, start = [0], [], time.perf_counter()
        for n, (at, w, h) in enumerate(events):
            delay = start + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            if inflight[0] >= max_inflight:  # an Edge drops a crop rather than queue without bound
                stats["dropped"] += 1
                continue
            crop = cv2.resize(crops[rng.randrange(len(crops))], (max(4, w), max(4, h)))
            jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            inflight[0] += 1
            stats["sent"] += 1
            tasks.append(asyncio.create_task(send(cam, n, jpeg, inflight)))
        await asyncio.gather(*tasks)

    await asyncio.gather(*(camera(c) for c in range(cameras)))
    return stats


def main():
    import cv2
    import numpy as np
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--cameras", type=int, required=True)
    p.add_argument("--seconds", type=float, default=125)
    p.add_argument("--logs", required=True, help="glob of recorded Edge plate.jsonl files")
    p.add_argument("--crops", required=True, help="folder of real plate crop JPEGs")
    p.add_argument("--source-fps", type=float, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="new JSONL file: one line per reply")
    a = p.parse_args()
    schedules = load_schedules(a.logs, a.source_fps)
    crops = [c for c in (cv2.imread(f) for f in sorted(glob.glob(os.path.join(a.crops, "*.jpg")))) if c is not None]
    if not schedules or not crops or a.cameras < 0:
        p.error("need recorded VN-like plate.jsonl schedules, crops and cameras >= 0")
    rate = sum(len(s) / (s[-1][0] + 1.0) for s in schedules) / len(schedules)
    print(f"{len(schedules)} schedules ({rate:.1f} reads/s each on average), {len(crops)} crops, "
          f"{a.cameras} cameras x {a.seconds:.0f} s -> {a.target}", flush=True)
    log = open(a.out, "x", encoding="utf-8") if a.out else None

    async def go():
        client = PlateOcrClient(a.target, timeout_s=10)
        try:
            return await run(client, schedules, crops, cameras=a.cameras, seconds=a.seconds, seed=a.seed, log=log)
        finally:
            await client.close()
    try:
        s = asyncio.run(go())
    finally:
        if log is not None:
            log.close()
    rtt = np.array(s.pop("rtt_ms") or [np.nan])
    print(json.dumps({**s, "reads_per_s": s["replies"] / a.seconds, "rtt_p50": float(np.nanpercentile(rtt, 50)),
                      "rtt_p95": float(np.nanpercentile(rtt, 95))}), flush=True)


if __name__ == "__main__":
    main()
