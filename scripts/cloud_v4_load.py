"""Load the v4 Cloud workers through Kafka like C cameras would (5/11, step 3 gate). Run on 249 or an Edge.

  PYTHONPATH=. python scripts/cloud_v4_load.py --frames DIR --head head.onnx --head-id ID --mode split|rtdetr|dfinex
      [--cameras 1 4 8 16] [--seconds 10] [--fps 15] [--bits 2]

Each fake camera sends one ticket, waits for the reply (with ByteTrack done by the worker), then sends the next:
rtdetr/dfinex = JPEG q85 640 (edge_agent encode_cloud_jpeg), split = the cut tensor (float head + cut_tensor.pack,
--bits 2 = CUT2 as live since 14/11, 8 = CUT1). --fps 15 (15/11) paces each camera like a real one: a frame whose
slot passed while waiting is skipped (as the Edge does); --fps 0 = as fast as replies come (closed loop).
Reports frames/s, per-camera frames/s and share of slots served, reply latency p50/p95, rejected replies.
"""
import argparse
import asyncio
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

TRITON = {"rtdetr": "rtdetr_full", "dfinex": "dfinex_full", "split": "rtdetr_tail_u8"}   # split + --bits 2: rtdetr_tail_q
METRICS = ("nv_inference_request_success", "nv_inference_exec_count", "nv_inference_request_duration_us",
           "nv_inference_queue_duration_us", "nv_inference_compute_infer_duration_us")


def triton_counters(model, url="http://localhost:18002/metrics"):
    import re
    import urllib.request
    try:
        text = urllib.request.urlopen(url, timeout=5).read().decode()
    except OSError:
        return None
    return [float(re.search(rf'^{m}{{model="{model}",version="1"}} (\S+)', text, re.M).group(1)) for m in METRICS]


def cpu_busy():
    """Whole-host CPU busy share since the previous call (from /proc/stat)."""
    v = [float(x) for x in open("/proc/stat").readline().split()[1:]]
    idle, total = v[3] + v[4], sum(v)
    prev = getattr(cpu_busy, "prev", (idle, total))
    cpu_busy.prev = (idle, total)
    return 1 - (idle - prev[0]) / max(total - prev[1], 1)


MODEL_IDS = {"rtdetr": "rtdetrv2_s_candidate_6b5b2ca4fb32", "dfinex": "dfine_x_candidate_9461641566be",
             "split": "rtdetrv2_r18vd_sp2_120e_coco+layer2_split"}


async def run(a, payloads, cams):
    from contracts import CloudInferenceTicket, FrameIdentity, ModelMode
    from contracts.errors import CloudInferenceRejected
    from contracts.kafka_transport import KafkaCloudClient
    mode = {"rtdetr": ModelMode.RTDETR_CLOUD, "dfinex": ModelMode.DFINEX_CLOUD, "split": ModelMode.RTDETR_SPLIT_CLOUD}[a.mode]
    client = KafkaCloudClient(a.bootstrap, node_id=f"load-{uuid4().hex[:6]}", max_pending=256)
    await client.start()
    await asyncio.sleep(2)
    lat, rejected, detail, inside = [], {}, {}, []
    end = time.perf_counter() + a.seconds
    epoch = time.time_ns() // 1_000_000

    async def camera(k):
        i, start = 0, time.perf_counter()
        while time.perf_counter() < end:
            if a.fps:
                slot = start + i / a.fps
                if (wait := slot - time.perf_counter()) > 0:
                    await asyncio.sleep(wait)
                else:
                    i = max(i, int((time.perf_counter() - start) * a.fps))   # skip the slots already missed
            now = time.time_ns()
            frame = FrameIdentity(node_id="load", camera_id=f"cam{k}", stream_id="s", stream_epoch=epoch,
                                  source_frame_id=i, capture_utc_ns=now, source_pts=i * 66_666_667)
            body = payloads[(k + i) % len(payloads)]
            ticket = CloudInferenceTicket(frame=frame, model_mode=mode, model_id=MODEL_IDS[a.mode], decision_id="load/v4",
                                          preview_pts=i * 66_666_667, deadline_utc_ns=now + 2_000_000_000,
                                          request_id=uuid4().hex, **({"tensor": body} if a.mode == "split" else {"jpeg": body}))
            t = time.perf_counter()
            try:
                result = await client.submit_cloud_ticket(ticket)
                lat.append(time.perf_counter() - t)
                inside.append([float(result.attributes.get(k, "nan")) for k in  # worker's own stage times
                               ("cloud_e2e_ms", "cloud_admission_ms", "cloud_adapter_wall_ms", "cloud_adapter_ms")])
            except CloudInferenceRejected as exc:
                rejected[exc.status] = rejected.get(exc.status, 0) + 1
                detail.setdefault(exc.status, exc.detail[:300])
            except asyncio.TimeoutError:
                rejected["timeout"] = rejected.get("timeout", 0) + 1
            i += 1

    t0 = time.perf_counter()
    model = "rtdetr_tail_q" if a.mode == "split" and a.bits == 2 else TRITON[a.mode]
    m0 = triton_counters(model)
    cpu_busy()
    await asyncio.gather(*(camera(k) for k in range(cams)))
    cpu = cpu_busy()
    m1 = triton_counters(model)
    dur = time.perf_counter() - t0
    await client.close()
    r = np.array(lat) * 1000
    if len(r):
        e2e, adm, wall, own = np.nanpercentile(np.array(inside), 50, axis=0)
        p = (f"p50 {np.percentile(r, 50):6.1f}  p95 {np.percentile(r, 95):6.1f} ms | inside worker p50 {e2e:5.1f} ms = "
             f"slot wait {adm:5.1f} + adapter call {wall:5.1f} (adapter itself {own:5.1f})")
    else:
        p = "no replies"
    served = f" | {len(r) / dur / cams:5.1f} per camera = {len(r) / (dur * cams * a.fps):.0%} of slots" if a.fps else ""
    print(f"   {a.mode:6s} {cams:2d} cameras: {len(r) / dur:6.1f} frames/s{served}  reply {p}  rejected {rejected or 0}",
          flush=True)
    if m0 and m1:
        n, ex, req, queue, comp = (b - x for b, x in zip(m1, m0))
        print(f"      Triton {model}: per request {req / max(n, 1) / 1000:5.1f} ms = queue {queue / max(n, 1) / 1000:5.1f}"
              f" + GPU compute share {comp / max(n, 1) / 1000:5.1f} ms | mean batch {n / max(ex, 1):.2f} | host CPU busy {cpu:.0%}",
              flush=True)
    for status, text in detail.items():
        print(f"      first {status}: {text}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    for k in ("--frames", "--head", "--head-id"):
        ap.add_argument(k, required=True)
    ap.add_argument("--mode", choices=list(MODEL_IDS), required=True)
    ap.add_argument("--bootstrap", default="localhost:29092")
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--bits", type=int, choices=(2, 8), default=2)
    a = ap.parse_args()
    frames = [cv2.imread(str(p)) for p in sorted(Path(a.frames).glob("*.jpg"))]
    small = [cv2.resize(f, (640, 640), interpolation=cv2.INTER_AREA) for f in frames]
    if a.mode == "split":
        import onnxruntime as ort
        from contracts.cut_tensor import pack
        head = ort.InferenceSession(a.head, providers=["CPUExecutionProvider"])
        payloads = [pack(head.run(None, {"images": cv2.dnn.blobFromImage(s, 1 / 255.0, swapRB=True)})[0],
                         width=f.shape[1], height=f.shape[0], head_id=a.head_id, bits=a.bits)
                    for s, f in zip(small, frames)]
    else:
        payloads = [cv2.imencode(".jpg", s, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() for s in small]
    print(f"{a.mode}: {len(payloads)} payloads, {np.mean([len(p) for p in payloads]) / 1024:.0f} KB mean", flush=True)
    for c in a.cameras:
        asyncio.run(run(a, payloads, c))


if __name__ == "__main__":
    main()
