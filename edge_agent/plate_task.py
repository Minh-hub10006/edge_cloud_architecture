"""UC1 on the Edge, slow lane (8/11): plate crops of tracked vehicles, sent to the Cloud OCR through Kafka.

Only frames that carry their own full-resolution copy (CapturedFrame.hires, 3 per s) are used, so the vehicle box and
the pixels belong to the same instant. Per vehicle track at most `plate_cap` reads (default 2): the first when the
vehicle is big enough, the next one at least `plate_gap_s` later and only if it got clearly bigger (closer to the
camera). The plate detector (the v50 NPU/ONNX plate model, edge_agent.plate_branch.load_plate_detector) runs on the
vehicle crop, at most `plate_detects_per_s` times per second per camera. Nothing waits for the OCR: the Cloud reads
each crop and stores (camera, track, text) in the database (cloud_core/plate_worker.py -> table plates).
"""
from __future__ import annotations

import asyncio
import base64
from collections import OrderedDict

from .slow_lane import next_event
from .slow_tasks import VEHICLES, stream_fields

DEFAULTS = {"plate_cap": 2, "plate_gap_s": .6, "plate_grow": 1.15, "plate_min_vehicle_px": 64,
            "plate_detects_per_s": 4.0, "plate_tries": 4, "plate_jpeg_quality": 95}


class PlateTask:
    name = "plates"

    def __init__(self, sink=None, config=None, detector=None) -> None:
        self.cfg = {**DEFAULTS, **(config or {})}
        self.sink, self.detector = sink, detector
        if self.detector is None and self.cfg.get("plate_detector"):
            from .plate_branch import load_plate_detector
            self.detector = load_plate_detector(self.cfg["plate_detector"], imgsz=int(self.cfg.get("plate_imgsz", 320)),
                                                conf=float(self.cfg.get("plate_conf", .3)))
        if self.detector is None:
            raise ValueError("the plates task needs a plate detector (--plate-detector)")

    async def run(self, queue, ring, stop) -> dict:
        import cv2
        cfg = self.cfg
        tracks: OrderedDict[int, dict] = OrderedDict()   # track id -> {reads, tries, last_ns, height}
        throttle = cfg.get("throttle") or (lambda: 1.0)   # P1: the Cloud asks to slow down when its OCR lags
        rate = cfg["plate_detects_per_s"]
        tokens, last_t = rate, None   # token bucket: up to `rate` detector runs per second, bursts of 1 s
        stats = dict.fromkeys(("frames", "detects", "no_plate", "sent", "skipped_budget", "empty_crop", "errors"), 0)
        while (event := await next_event(queue, stop)) is not None:
            frame = event.frame
            hires = getattr(frame, "hires", None)
            if hires is None:
                continue
            stats["frames"] += 1
            h, w = hires.shape[:2]
            t = frame.capture_mono_ns
            if last_t is not None:
                tokens = min(rate, tokens + (t - last_t) / 1e9 * rate / throttle())
            last_t = t
            due = []
            for box in event.result.boxes:
                if box.class_id not in VEHICLES or box.track_id is None:
                    continue
                px = ((box.left * w), (box.top * h), (box.right * w), (box.bottom * h))
                height = px[3] - px[1]
                s = tracks.get(box.track_id) or {"reads": 0, "tries": 0, "last_ns": None, "height": 0.0}
                tracks[box.track_id] = s
                tracks.move_to_end(box.track_id)
                if (s["reads"] >= cfg["plate_cap"] or s["tries"] >= cfg["plate_tries"]
                        or min(px[2] - px[0], height) < cfg["plate_min_vehicle_px"]):
                    continue
                if s["last_ns"] is not None and (t - s["last_ns"] < cfg["plate_gap_s"] * 1e9
                                                 or height < s["height"] * cfg["plate_grow"]):
                    continue
                due.append((height, box, px, s))
            while len(tracks) > 2000:
                tracks.popitem(last=False)
            for height, box, px, s in sorted(due, key=lambda d: -d[0]):   # biggest (closest) vehicles first
                if tokens < 1:
                    stats["skipped_budget"] += 1
                    continue
                tokens -= 1
                s["tries"] += 1
                s["last_ns"], s["height"] = t, height
                x1, y1, x2, y2 = (int(max(0, v)) for v in px)
                vehicle = hires[y1:min(h, y2), x1:min(w, x2)]
                try:
                    plates = await asyncio.to_thread(self.detector, vehicle)
                except Exception as exc:   # one bad detector call skips this vehicle, never the task
                    stats["errors"] += 1
                    if stats["errors"] <= 3:
                        print(f"[plates] detector {type(exc).__name__}: {exc}", flush=True)
                    continue
                stats["detects"] += 1
                if not plates:
                    stats["no_plate"] += 1
                    continue
                p = max(plates, key=lambda b: b[4])
                from .plate_branch import crop_plate
                crop = crop_plate(vehicle, p)
                if crop.size == 0:   # 8/11 on 246: plate boxes outside the vehicle crop (also seen by v50 on 252)
                    stats["empty_crop"] += 1
                    continue
                ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(cfg["plate_jpeg_quality"])])
                if not ok:
                    continue
                s["reads"] += 1
                stats["sent"] += 1
                if self.sink is not None:
                    await self.sink("plate", {
                        **stream_fields(frame), "track_id": box.track_id, "frame_id": frame.identity.source_frame_id,
                        "capture_utc_ns": frame.identity.capture_utc_ns, "vehicle_class": box.class_id,
                        "read_no": s["reads"], "det_conf": round(float(p[4]), 3),
                        "plate_box": [round(x1 + p[0]), round(y1 + p[1]), round(x1 + p[2]), round(y1 + p[3])],
                        "jpeg_b64": base64.b64encode(jpeg.tobytes()).decode()})
        return {**stats, "vehicles_tracked": len(tracks)}
