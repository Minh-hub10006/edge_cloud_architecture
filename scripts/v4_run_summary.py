"""Table of one v4 Edge run (7/11): per stream inferred/drops/errors, latencies, CPU and temperature.

  python scripts/v4_run_summary.py RUN_DIR      (RUN_DIR from scripts/run_edge_v4.sh: spec.txt, cam<i>.log, cam<i>.jsonl)
"""
import json
import sys
from pathlib import Path

import numpy as np


def last_summary(log: Path):
    text = log.read_text(errors="replace") if log.exists() else ""
    i = text.rfind('{\n  "control_v1"')
    if i < 0:
        tail = [line for line in text.splitlines() if line.strip()][-2:]
        return None, " | ".join(tail)[-240:]
    return json.loads(text[i:text.index("\n}", i) + 2]), None


def p(values):
    return f"{np.percentile(values, 50):5.0f} /{np.percentile(values, 95):5.0f}" if values else "    -"


def main():
    out = Path(sys.argv[1])
    spec = (out / "spec.txt").read_text().split()
    print(f"run {out.name}: {len(spec)} streams")
    print(f"{'stream':7s} {'video:mode':12s} {'inferred':>8s} {'of':>6s} {'drops':>6s} {'errors':>6s} "
          f"{'worker->boxes p50/p95':>22s} {'capture->boxes p50/p95':>23s} {'CPU %':>6s} {'temp C':>7s}")
    for i, item in enumerate(spec, 1):
        s, err = last_summary(out / f"cam{i}.log")
        if s is None:
            print(f"cam-{i:<3d} {item:12s} FAILED: {err}")
            continue
        rows = [json.loads(l) for l in open(out / f"cam{i}.jsonl") if l.strip()] if (out / f"cam{i}.jsonl").exists() else []
        ev = [e for r in rows for e in r.get("frames", []) if e.get("status") == "accepted"]
        act = [e["active_path_latency_ms"] for e in ev if "active_path_latency_ms" in e]
        src = [e["source_completion_latency_ms"] for e in ev if "source_completion_latency_ms" in e]
        raw = [r.get("raw", {}) for r in rows]
        cpu = [x["cpu_total_pct"] for x in raw if isinstance(x.get("cpu_total_pct"), (int, float))]
        tmp = [x["temp_c"] for x in raw if isinstance(x.get("temp_c"), (int, float))]
        st = s["stats"]
        print(f"cam-{i:<3d} {item:12s} {st.get('frames_inferred', 0):8d} {st.get('analytics_selected', 0):6d} "
              f"{s.get('analytics_queue_dropped', 0):6d} {sum(s.get('error_reasons', {}).values()):6d} {p(act):>22s} {p(src):>23s} "
              f"{np.median(cpu) if cpu else float('nan'):6.1f} {max(tmp) if tmp else float('nan'):7.1f}")
        if s.get("error_reasons"):
            print(f"        errors: {s['error_reasons']}")


if __name__ == "__main__":
    main()
