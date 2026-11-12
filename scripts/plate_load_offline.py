"""Plate OCR load of real videos, measured with the integrated plate path (Edge PlateBranch -> gRPC -> Cloud plate
OCR service), on one machine, every frame analysed (no real-time drops).

  KMP_DUPLICATE_LIB_OK=TRUE python -B scripts/plate_load_offline.py --detector best_320.pt --imgsz 320 \
      --model-home C:/Users/ADMIN/plate_ocr_home --cpu --out results/plate_load/vn_320 VIDEO...

The branch reads every plate track every --every frames for its whole life (cap 0, confirm 0) and logs each reply in
order; any stopping policy (confirm k equal valid reads, at most c reads) is a prefix of that sequence, so one run
gives the load of every policy (`summarize`). Not captured here: crops sent while a confirming reply is in flight.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloud_core.plate_ocr_service import _init, make_reader, start_server  # noqa: E402
from contracts.plate_ocr_wire import PlateOcrClient  # noqa: E402
from edge_agent.plate_branch import PlateBranch, load_plate_detector  # noqa: E402
from edge_agent.plate_policy import PlateReadPolicy  # noqa: E402
from inference.plate_ocr import valid_plate  # noqa: E402

POLICIES = [("all_every3", 0, 0), ("cap1", 1, 0), ("confirm1_cap5", 5, 1), ("confirm2_cap3", 3, 2),
            ("confirm2_cap5", 5, 2), ("confirm2_cap10", 10, 2), ("cap5", 5, 0), ("cap10", 10, 0)]


def stop_index(texts, cap, confirm):
    """(reads, end) of a (cap, confirm) policy on a track whose every-k replies were `texts`; end as PlateTrack."""
    votes = Counter()
    for i, t in enumerate(texts, 1):
        if confirm and valid_plate(t):
            votes[t] += 1
            if votes[t] >= confirm:
                return i, "confirmed"
        if cap and i >= cap:
            return i, "cap"
    return len(texts), "lost"


def summarize(reads, tracks, seconds, every):
    by_track = defaultdict(list)
    for r in sorted(reads, key=lambda r: r["frame"]):
        by_track[r["track"]].append(r)
    calls_per_read = sum(r["calls"] for r in reads) / max(1, len(reads))
    out = {"seconds": seconds, "plate_tracks": len(tracks),
           "plate_tracks_read": len(by_track), "reads_all": len(reads),
           "two_line_share": sum(r.get("lines") == 2 for r in reads) / max(1, len(reads)),
           "valid_reply_share": sum(valid_plate(r["text"]) for r in reads) / max(1, len(reads)),
           "tracks_with_valid_text": sum(any(valid_plate(r["text"]) for r in rs) for rs in by_track.values()),
           "calls_per_read": calls_per_read,
           "crop_w_median": sorted(r["w"] for r in reads)[len(reads) // 2] if reads else None,
           "crop_h_median": sorted(r["h"] for r in reads)[len(reads) // 2] if reads else None,
           "ocr_ms_per_call": sum(r["ocr_ms"] for r in reads) / max(1, sum(r["calls"] for r in reads)),
           "every": every, "policies": {}}
    for name, cap, confirm in POLICIES:
        stops = [stop_index([r["text"] for r in rs], cap, confirm) for rs in by_track.values()]
        used = [rs[:n] for rs, (n, _) in zip(by_track.values(), stops)]
        n_reads = sum(len(u) for u in used)
        out["policies"][name] = {
            "reads_per_track": n_reads / max(1, len(used)),
            "reads_per_s": n_reads / seconds, "calls_per_s": sum(r["calls"] for u in used for r in u) / seconds,
            "ends": dict(Counter(end for _, end in stops))}
    return out


async def measure(video, detector, client, every, fps_hint):
    import cv2
    lines = []

    class Log:
        def write(self, s): lines.append(json.loads(s))
        def flush(self): pass
    branch = PlateBranch(detector, PlateReadPolicy(every=every, cap=0, confirm=0), client,
                         stream_key=("offline", os.path.basename(video), "plate", 0), log=Log(),
                         queue_capacity=8, max_inflight=1_000_000)
    done = asyncio.Event()
    task = asyncio.create_task(branch.run(done))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    i = 0
    while True:
        ok, px = await asyncio.to_thread(cap.read)
        if not ok:
            break
        while branch.queue.pending >= 4 or len(branch.inflight) >= 32:  # back-pressure: analyse every frame
            await asyncio.sleep(.002)
        branch.queue.put_latest(SimpleNamespace(identity=SimpleNamespace(source_frame_id=i), pixels=px))
        i += 1
    done.set()
    stats = await task
    assert branch.queue.dropped == 0 and stats.get("crops_dropped", 0) == 0, "offline run must not drop"
    return lines, stats, i / fps


async def main_async(a):
    detector = load_plate_detector(a.detector, imgsz=a.imgsz, conf=a.conf)
    os.makedirs(a.out, exist_ok=True)
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(make_reader, a.model_home, a.passes,
                                                                       not a.cpu)) as pool:
        server, port = await start_server("127.0.0.1:0", pool)
        client = PlateOcrClient(f"127.0.0.1:{port}", timeout_s=120)
        try:
            for video in a.videos:
                name = os.path.splitext(os.path.basename(video))[0]
                lines, stats, seconds = await measure(video, detector, client, a.every, 15)
                with open(os.path.join(a.out, f"{name}.jsonl"), "w", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(x) + "\n" for x in lines)
                reads = [x for x in lines if x["event"] == "read"]
                tracks = [x for x in lines if x["event"] == "track"]
                errors = [x for x in lines if x["event"] == "error"]
                s = {"video": name, **summarize(reads, tracks, seconds, a.every), "errors": len(errors),
                     "frames": stats["frames"], "detect_ms_mean": stats["detect_ms_mean"],
                     "detector": a.detector, "imgsz": a.imgsz, "conf": a.conf, "passes": a.passes}
                with open(os.path.join(a.out, "summary.jsonl"), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(s) + "\n")
                p = s["policies"]
                print(f"{name}: {seconds:.0f}s tracks {s['plate_tracks']} read {s['plate_tracks_read']} "
                      f"reads/s all {p['all_every3']['reads_per_s']:.2f} c2cap5 {p['confirm2_cap5']['reads_per_s']:.2f} "
                      f"calls/read {s['calls_per_read']:.1f} 2-line {s['two_line_share']:.0%} "
                      f"valid {s['valid_reply_share']:.0%} errors {len(errors)}", flush=True)
        finally:
            await client.close()
            await server.stop(0)


def dump_crops(out, videos, crops_dir, per_video=40):
    """Real plate crops (as sent to the Cloud) from a finished run's logs, evenly spaced, for the 4090 co-run."""
    import cv2
    from edge_agent.plate_branch import crop_plate
    os.makedirs(crops_dir, exist_ok=True)
    for video in videos:
        name = os.path.splitext(os.path.basename(video))[0]
        reads = [x for x in map(json.loads, open(os.path.join(out, f"{name}.jsonl"), encoding="utf-8"))
                 if x["event"] == "read"]
        cap = cv2.VideoCapture(video)
        for k, r in enumerate(reads[::max(1, len(reads) // per_video)][:per_video]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
            ok, px = cap.read()
            if ok:
                cv2.imwrite(os.path.join(crops_dir, f"{name}_{k:03d}.jpg"), crop_plate(px, r["box"]),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--detector", default="")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--conf", type=float, default=.3)
    p.add_argument("--model-home", default="")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--dump-crops", default="", help="only write real crops from the logs in --out to this folder")
    a = p.parse_args()
    if a.dump_crops:
        return dump_crops(a.out, a.videos, a.dump_crops)
    if not a.detector or not a.model_home:
        p.error("--detector and --model-home are required to measure")
    asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
