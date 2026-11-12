"""Step 4 gate on an NPU Edge (246/252): the RT-DETR head on the persistent SNPE daemon, then path B over the LAN.

  python scripts/edge_head_check.py --dlc head_w8a8_pc.dlc --frames DIR [--head-onnx head.onnx]
      [--bootstrap 192.168.180.249:29092 --cameras 1 4 --seconds 10]

1. Parity: the daemon's cut tensor vs the float head (onnxruntime CPU) on the same frames: correlation (G2 >= .99).
2. Edge cost per frame over 100 runs: prepare / NPU / pack (uint8 + zstd), and KB on the wire.
3. With --bootstrap: C cameras send path-B tickets to the v4 Cloud workers (scripts/run_cloud_v4.sh, started with
   RTDETR_SPLIT_HEAD_ID=rtl2-w8a8pc) like real cameras: 15 fps each without waiting for replies, at most 3 frames in
   flight per camera (else the frame is dropped); frames/s, latency frame -> boxes p50/p95, drops.
"""
import argparse
import asyncio
import sys
import time
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contracts.cut_tensor import unpack  # noqa: E402
from inference.rtdetr_head_snpe import HEAD_ID, RtdetrHeadSnpe, prepare  # noqa: E402


async def lan(a, head, frames, cams):
    from contracts import CloudInferenceTicket, FrameIdentity, ModelMode
    from contracts.errors import CloudInferenceRejected
    from contracts.kafka_transport import KafkaCloudClient
    client = KafkaCloudClient(a.bootstrap, node_id=f"edge-check-{uuid4().hex[:6]}", max_pending=64)
    await client.start()
    await asyncio.sleep(2)
    lat, rejected, parts = [], {}, []
    end, epoch = time.perf_counter() + a.seconds, time.time_ns() // 1_000_000

    dropped = [0]

    async def one(k, i):
        start = time.perf_counter()
        # only the NPU call takes turns (RtdetrHeadSnpe._npu); prepare and pack run in parallel threads
        data, _ = await asyncio.to_thread(head.encode, frames[(k + i) % len(frames)])
        encoded = time.perf_counter()
        now = time.time_ns()
        ticket = CloudInferenceTicket(
            frame=FrameIdentity(node_id="edge-check", camera_id=f"cam{k}", stream_id="s", stream_epoch=epoch,
                                source_frame_id=i, capture_utc_ns=now, source_pts=i * 66_666_667),
            model_mode=ModelMode.RTDETR_SPLIT_CLOUD, model_id="rtdetrv2_r18vd_sp2_120e_coco+layer2_split",
            decision_id="load/v4-edge", preview_pts=i * 66_666_667, deadline_utc_ns=now + 2_000_000_000,
            request_id=uuid4().hex, tensor=data)
        try:
            result = await client.submit_cloud_ticket(ticket)
            lat.append(time.perf_counter() - start)
            parts.append([(encoded - start) * 1000, (time.perf_counter() - encoded) * 1000,
                          float(result.attributes.get("cloud_e2e_ms", "nan"))])
        except CloudInferenceRejected as exc:
            rejected[exc.status] = rejected.get(exc.status, 0) + 1
        except asyncio.TimeoutError:
            rejected["timeout"] = rejected.get("timeout", 0) + 1

    async def camera(k):
        """Like a real camera: one frame every 1/15 s whatever the replies do; at most 3 in flight, else drop it."""
        i, running, t = 0, set(), time.perf_counter()
        while time.perf_counter() < end:
            running = {x for x in running if not x.done()}
            if len(running) < 3:
                running.add(asyncio.create_task(one(k, i)))
            else:
                dropped[0] += 1
            i += 1
            t += 1 / 15
            await asyncio.sleep(max(0.0, t - time.perf_counter()))
        if running:
            await asyncio.gather(*running)

    t0 = time.perf_counter()
    await asyncio.gather(*(camera(k) for k in range(cams)))
    dur = time.perf_counter() - t0
    await client.close()
    r = np.array(lat) * 1000
    tail = "no replies"
    if len(r):
        enc, reply, inside = np.nanpercentile(np.array(parts), 50, axis=0)
        tail = (f"p50 {np.percentile(r, 50):6.1f}  p95 {np.percentile(r, 95):6.1f} ms = Edge encode {enc:5.1f} + "
                f"Kafka/LAN/Cloud reply {reply:5.1f} (inside worker {inside:5.1f})")
    print(f"   {cams} cameras over LAN: {len(r) / dur:5.1f} frames/s (max {15 * cams}), frame -> boxes {tail}, "
          f"dropped at the Edge {dropped[0]}, rejected {rejected or 0}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlc", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--head-onnx")
    ap.add_argument("--bootstrap")
    ap.add_argument("--cameras", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seconds", type=float, default=10)
    a = ap.parse_args()
    frames = [cv2.imread(str(p)) for p in sorted(Path(a.frames).glob("*.jpg"))]
    head = RtdetrHeadSnpe(a.dlc)
    payloads = [head.encode(f)[0] for f in frames]
    if a.head_onnx:
        import onnxruntime as ort
        ref = ort.InferenceSession(a.head_onnx, providers=["CPUExecutionProvider"])
        corr = []
        for f, p in zip(frames, payloads):
            want = ref.run(None, {"images": np.ascontiguousarray(prepare(f).transpose(0, 3, 1, 2))})[0]
            got, _, _ = unpack(p, expected_head_id=HEAD_ID)
            corr.append(float(np.corrcoef(want.ravel(), got.ravel())[0, 1]))
        print(f"== parity NPU vs float head: corr min {min(corr):.4f} mean {np.mean(corr):.4f} "
              f"-> {'PASS' if min(corr) >= .99 else 'FAIL'} (G2 >= .99; includes the uint8 wire step)", flush=True)
    st = np.array([list(head.encode(frames[i % len(frames)])[1].values()) for i in range(100)])
    p, n, k = st[10:].mean(axis=0)
    print(f"== Edge cost per frame: prepare {p:.1f} + NPU {n:.1f} + pack {k:.1f} = {p + n + k:.1f} ms "
          f"(~{1000 / (p + n + k):.0f} frames/s on one NPU) | wire {np.mean([len(x) for x in payloads]) / 1024:.0f} KB", flush=True)
    for threads in (2, 4):
        from concurrent.futures import ThreadPoolExecutor
        t0, n = time.perf_counter(), 100
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(lambda i: head.encode(frames[i % len(frames)]), range(n)))
        print(f"== Edge throughput with {threads} threads: {n / (time.perf_counter() - t0):.0f} frames/s", flush=True)
    if a.bootstrap:
        for c in a.cameras:
            asyncio.run(lan(a, head, frames, c))
    head.session.close()


if __name__ == "__main__":
    main()
