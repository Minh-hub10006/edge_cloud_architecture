"""Edge plate path, independent of where vehicle detection runs: detect plates on every published frame, pick crops
with PlateReadPolicy, send them to the Cloud plate OCR service and log every read and every finished plate track."""
from __future__ import annotations

import asyncio
import json
from collections import Counter
from time import perf_counter_ns, time_ns

from .fanout import BoundedLatestQueue
from .plate_policy import PlateReadPolicy


def plate_decoder(imgsz: int, conf: float):
    """The NPU plate adapter's letterbox + decoder, without an NPU session.
    shortcut: built with __new__ because the adapter resolves a DLC in __init__; split it out if reused more."""
    from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector
    dec = YOLOSnpeDetector.__new__(YOLOSnpeDetector)
    dec.variant, dec.imgsz, dec.score_thresh, dec.max_det = "plate", imgsz, conf, 20
    dec.input_layout, dec._pad, dec._tensor, dec.session = "nchw", None, None, None
    return dec


def load_plate_detector(path: str, *, imgsz: int = 320, conf: float = .3):
    """-> detect(frame_bgr) -> [(x1, y1, x2, y2, conf)] in frame pixels.
    .dlc or .onnx: the platform's NPU (or ORT CUDA) plate adapter. .pt: the same letterbox and decoder around the
    float PyTorch model (laptop / offline runs), so only INT8 quantization differs from the Edge."""
    if path.endswith(".pt"):
        import torch
        from ultralytics import YOLO
        net = YOLO(path).model.float().eval()
        dec = plate_decoder(imgsz, conf)

        def run(tensor):
            with torch.no_grad():
                pred = net(torch.from_numpy(tensor))[0].numpy()  # [1, 4 + classes, N]
            return {"boxes": pred[:, :4], "scores": pred[:, 4:]}
    else:
        from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector
        dec = YOLOSnpeDetector(dlc_path=path, variant="plate", imgsz=imgsz, score_thresh=conf)
        if dec.session is None:  # the adapter only prints a missing DLC; a plate path without a model is an error
            raise RuntimeError(f"plate detector {path} did not load")

        def run(tensor):
            return dec.session.infer_raw(tensor)[1]

    def detect(frame):
        tensor, r = dec._preprocess(frame)
        outs = run(tensor)
        return [(b["left"], b["top"], b["right"], b["bottom"], b["confidence"])
                for b in (dec.decode_outputs(outs, r) if outs else [])]
    return detect


def crop_plate(frame, box, pad: int = 5):
    """The team's cat_bien: the plate box plus `pad` pixels, clipped to the frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box[:4])
    return frame[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)]


class PlateBranch:
    def __init__(self, detector, policy: PlateReadPolicy, client, *, stream_key, max_inflight: int = 16,
                 jpeg_quality: int = 95, queue_capacity: int = 2, log=None) -> None:
        if max_inflight < 1 or not 1 <= jpeg_quality <= 100:
            raise ValueError("max_inflight >= 1, jpeg_quality 1..100")
        self.detector, self.policy, self.client, self.log = detector, policy, client, log
        self.stream = list(stream_key)
        self.max_inflight, self.jpeg_quality = max_inflight, jpeg_quality
        self.queue = BoundedLatestQueue(queue_capacity)
        self.stats = Counter()
        self.reasons = Counter()
        self.inflight: set[asyncio.Task] = set()  # OCR requests awaiting a reply

    def _write(self, record: dict) -> None:
        if self.log is not None:
            self.log.write(json.dumps({"stream": self.stream, **record}, separators=(",", ":")) + "\n")

    async def _send(self, track, frame_id: int, jpeg: bytes, box) -> None:
        sent = time_ns()
        try:
            reply = await self.client.read({"stream": self.stream, "track": track.track_id, "frame": frame_id,
                                            "sent_utc_ns": sent}, jpeg)
        except Exception as exc:
            self.stats["errors"] += 1
            self.reasons[type(exc).__name__] += 1
            self._write({"event": "error", "frame": frame_id, "track": track.track_id, "error": type(exc).__name__})
            return
        self.policy.record(track, reply.get("text", ""))
        self.stats["replies"] += 1
        self._write({"event": "read", "frame": frame_id, "track": track.track_id, "bytes": len(jpeg),
                     "box": [round(v, 1) for v in box[:4]],
                     "rtt_ms": (time_ns() - sent) / 1e6, **reply})

    async def run(self, done: asyncio.Event) -> dict:
        """Consume self.queue until `done` is set and the queue is empty, then wait for replies in flight."""
        import cv2
        inflight = self.inflight
        detect_ns = 0
        while not done.is_set() or self.queue.pending:
            try:
                frame = await asyncio.wait_for(self.queue.get(), .05)
            except asyncio.TimeoutError:
                continue
            frame_id = frame.identity.source_frame_id
            began = perf_counter_ns()
            try:
                boxes = await asyncio.to_thread(self.detector, frame.pixels)
            except Exception as exc:
                self.stats["detect_errors"] += 1
                self.reasons[type(exc).__name__] += 1
                continue
            detect_ns += perf_counter_ns() - began
            self.stats["frames"] += 1
            self.stats["plates_detected"] += len(boxes)
            for track in self.policy.update(frame_id, boxes):
                crop = crop_plate(frame.pixels, track.box)
                if crop.size == 0:  # box outside the frame (seen on Edge252's NPU): skip it, never stop the stream
                    self.policy.cancel(track)
                    self.stats["crops_empty"] += 1
                    if self.stats["crops_empty"] <= 5:
                        self._write({"event": "empty_crop", "frame": frame_id, "box": [float(v) for v in track.box]})
                    continue
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if len(inflight) >= self.max_inflight or not ok:
                    self.policy.cancel(track)
                    self.stats["crops_dropped"] += 1
                    continue
                self.stats["crops_sent"] += 1
                task = asyncio.create_task(self._send(track, frame_id, buf.tobytes(), track.box))
                inflight.add(task)
                task.add_done_callback(inflight.discard)
        await asyncio.gather(*inflight, return_exceptions=True)
        tracks = self.policy.close()
        for summary in tracks:
            self._write(summary)
        if self.log is not None:
            self.log.flush()
        return {**self.stats, "tracks": len(tracks), "queue_dropped": self.queue.dropped,
                "detect_ms_mean": detect_ns / 1e6 / max(1, self.stats["frames"]), "error_reasons": dict(self.reasons),
                "ends": dict(Counter(t["end"] for t in tracks))}
