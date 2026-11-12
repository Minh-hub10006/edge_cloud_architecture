"""Watch the check page's API from any machine while streams run (6/11): is the database keeping up, per stream?

  python -B scripts/check_probe.py [--url http://192.168.180.249:18200] [--seconds 60]

Polls like the page (5 times per second per stream, only new frames). Reports per stream: frames/s received, API time
p50/p95, gaps in frame numbers, and how far the newest stored frame is behind the capture clock of the database.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

import numpy as np


def get(url):
    t = time.perf_counter()
    with urllib.request.urlopen(url, timeout=5) as r:
        data = r.read()
    return json.loads(data), (time.perf_counter() - t) * 1000, len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.180.249:18200")
    ap.add_argument("--seconds", type=float, default=60)
    a = ap.parse_args()
    state, end = {}, time.time() + a.seconds
    while time.time() < end:
        streams, _, _ = get(a.url + "/api/streams")
        for s in streams:
            key = (s["node_id"], s["camera_id"], s["stream_id"], s["stream_epoch"])
            st = state.setdefault(key, {"after": -1, "times": [], "bytes": [], "seen": [], "start": time.time(), "age": []})
            st["age"].append(s["age_s"])
            q = urllib.parse.urlencode({"node": key[0], "camera": key[1], "stream": key[2], "epoch": key[3], "after_ms": st["after"]})
            r, ms, size = get(f"{a.url}/api/recent?{q}")
            st["times"].append(ms)
            st["bytes"].append(size)
            for f in r["frames"]:
                st["seen"].append(f["pts_ms"])
                st["after"] = max(st["after"], f["pts_ms"])
        time.sleep(.2)
    if not state:
        print("no stream seen")
    for key, st in state.items():
        seen = sorted(set(st["seen"]))
        dur = max(time.time() - st["start"], 1e-9)
        steps = np.diff(seen) if len(seen) > 1 else np.array([0])
        print(f"{key[1]}: {len(seen) / dur:5.1f} frames with boxes/s | API p50 {np.percentile(st['times'], 50):6.1f} p95 "
              f"{np.percentile(st['times'], 95):6.1f} max {max(st['times']):7.1f} ms | {np.mean(st['bytes']) / 1024:6.1f} KB per reply | "
              f"largest frame gap {steps.max() / 66.67:4.1f} frames | newest stored p50 {np.median(st['age']):.2f} s, max {max(st['age']):.2f} s old")


if __name__ == "__main__":
    main()
