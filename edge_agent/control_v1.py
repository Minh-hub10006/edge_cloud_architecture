"""Opt-in simple monitor -> rule policy -> real route executor.

Run independently of phase7/benchmark. All frames retain the action that
selected them. Media remains fixed-profile RTSP; there is no DRL training.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from collections import Counter, deque
import json
import math
from pathlib import Path
import subprocess
import sys
from time import monotonic_ns, perf_counter_ns, time_ns
from uuid import uuid4

from contracts import CloudInferenceTicket, DetectionResult, ModelMode, Placement
from contracts.confidence_calibration import PlattConfidenceCalibrator, calibrate_detection_result
from contracts.control_v1 import finite
from contracts.errors import CloudInferenceRejected
from contracts.reference_sink import MediaReferenceSink
from edge_agent import BoundedLatestQueue
from edge_agent.phase7 import _box_change, _tracked_boxes
from edge_agent.resources_v2 import box_features, scene_features
from edge_agent.slow_lane import build_slow_lane
from edge_agent.state_metrics import BlurDifficultyNormalizer, laplacian_variance
from edge_agent.state_v1 import POLICIES, SUPPORTED_MODES, RuleController, TimeWindowMonitor, PrometheusV1
from inference.frame_refs import MediaReferenceMap
from media.pipeline import MediaCadence


# 8/11, VN 200 frames: laplacian_variance at blur_width 320 on stretched 640x640 hardware frames is 0.69x the
# full-frame value (correlation 0.998). shortcut: one scale from 2 portrait videos; recalibrate the blur normalizer
# on hardware frames before trusting blur_difficulty for DRL.
HW_BLUR_SCALE = 1 / 0.69


def unstretch(frame):
    """Edge detectors (YOLO, trained on letterboxed frames) get the camera's aspect back: on stretched 640x640 frames
    YOLO11s F1 fell 0.4536 -> 0.3957, squeezed back to 360x640 it is 0.4512 (VN, 8/11). Boxes stay normalized."""
    if frame.source_size is None:
        return frame.pixels
    import cv2
    w, h = frame.source_size
    side = max(frame.pixels.shape[:2])
    return cv2.resize(frame.pixels, (round(side * w / max(w, h)), round(side * h / max(w, h))),
                      interpolation=cv2.INTER_AREA)


def encode_cloud_jpeg(pixels, size, quality):
    """BGR frame -> JPEG of the Cloud models' square input (640x640 stretch): boxes are normalized, so this is
    exactly what the model sees, without sending pixels its resize would drop."""
    import cv2
    ok, data = cv2.imencode(".jpg", cv2.resize(pixels, (size, size), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return data.tobytes()


PROBE_VEHICLES = frozenset({2, 3, 5, 7})  # COCO car, motorcycle, bus, truck


def _iou(a, b):
    w = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    h = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top))
    inter = w * h
    union = (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - inter
    return inter / union if union > 0 else 0.0


def probe_missed(edge_boxes, cloud_boxes, *, edge_conf=.30, cloud_conf=.65, iou=.5):
    """Vehicles the reference Cloud model finds and the Edge model misses on the same frame (greedy IoU match).
    The Edge cannot see its own misses; this is the hardness signal of docs/PLAN_PROBE_VI.md (AUC .76 offline)."""
    edge = [b for b in edge_boxes if b.class_id in PROBE_VEHICLES and b.confidence >= edge_conf]
    used, missed = set(), 0
    for ref in sorted((b for b in cloud_boxes if b.class_id in PROBE_VEHICLES and b.confidence >= cloud_conf),
                      key=lambda b: -b.confidence):
        best, best_k = 0.0, None
        for k, cand in enumerate(edge):
            if k not in used:
                v = _iou(ref, cand)
                if v > best:
                    best, best_k = v, k
        if best >= iou:
            used.add(best_k)
        else:
            missed += 1
    return missed


EDGE_TRACK_ID_BASE = 1_000_000  # Edge track IDs live above the Cloud's, so a mode switch never merges two vehicles


async def run_pipeline(source, publisher, client, detectors, controller, monitor, *,
                       cloud_model_id="dfine_full", cloud_model_ids=None, queue_capacity=8, priming_frames=30,
                       padding_frames=30, ticket_ttl_s=1.5, max_frame_age_s=.5,
                       drain_s=10, blur_normalizer=None, blur_width=320,
                       calibrators=None, emit=None, exporter=None, resource_sampler=None,
                       resource_interval_s=1.0, cloud_inflight=1, cloud_jpeg_quality=None, cloud_jpeg_size=640,
                       probe_period_s=0.0, probe_frames=2, probe_timeout_s=1.0,
                       probe_cloud_mode=ModelMode.DFINEX_CLOUD, probe_edge_mode=ModelMode.YOLO11S_EDGE,
                       plate=None, split_head=None, edge_tracker=None, slow_lane=None):
    """Injected dependencies permit real-transport tests without GPU/video hardware.

    cloud_inflight: analytics workers. With one, a Cloud model runs at most 1000/round-trip FPS (~67 ms on the
    LAN = right at 15 FPS); two let the next ticket leave while one waits. Edge models stay serial (one lock).
    cloud_jpeg_quality: None = Cloud reads frames from the RTSP stream (MediaFrameRefs sent); 1..100 = each Cloud
    ticket carries a JPEG of its frame and no MediaFrameRef leaves the Edge (video stays on the Edge's MediaMTX).
    probe_period_s > 0: probe_frames frames per period also get the other side's result (probe_edge_mode on the
    Edge, probe_cloud_mode as a low-priority JPEG ticket) in background tasks; frames never wait for a probe.
    plate: an edge_agent.plate_branch.PlateBranch; it gets every published frame whatever the analytics action.
    edge_tracker: v4 mode A, a cloud_core.tracking.CloudByteTrack run here: Edge results get their track IDs on the Edge
    (offset by EDGE_TRACK_ID_BASE so they never collide with Cloud IDs after a switch) and go to the database through
    client.publish_tracked_result; nothing waits for the Cloud.
    slow_lane: edge_agent.slow_lane.SlowLane (8/11): every captured frame goes to its ring and every result to its BUS,
    both non-blocking; its tasks run beside the fast lane until the media lane is done and they drained.
    split_head: inference.rtdetr_head_snpe.RtdetrHeadSnpe for path B (rtdetr_split_cloud): the ticket carries its cut
    tensor instead of pixels. With cloud_inflight > 1 frames encode in parallel (only the NPU call takes turns, inside
    the adapter) and leave in frame order, because the Cloud rejects a frame older than one it has already seen.
    """
    if cloud_inflight < 1:
        raise ValueError("cloud_inflight must be >= 1")
    ttl_s = finite(ticket_ttl_s, "ticket_ttl_s", positive=True)
    finite(max_frame_age_s, "max_frame_age_s", positive=True)
    finite(drain_s, "drain_s", positive=True)
    if priming_frames < 0 or padding_frames < 0 or blur_width <= 0:
        raise ValueError("invalid priming/padding/blur configuration")
    if cloud_jpeg_quality is not None and not (1 <= cloud_jpeg_quality <= 100 and 32 <= cloud_jpeg_size <= 1280):
        raise ValueError("cloud JPEG quality must be 1..100 and size 32..1280")
    if probe_period_s and (probe_period_s < 0 or probe_frames < 1 or probe_timeout_s <= 0):
        raise ValueError("probe period/frames/timeout must be positive")
    emit = emit or (lambda value: print(json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True))
    # Cloud model IDs are artifact identities the Cloud checks per ticket; Edge IDs come from adapters.
    cloud_ids = {ModelMode.DFINE_CLOUD: cloud_model_id, **(cloud_model_ids or {})}
    for mode in controller.modes:
        if mode.placement is Placement.EDGE and mode not in detectors:
            raise ValueError(f"no loaded Edge detector for allowed mode {mode.value}")
        if mode.placement is Placement.CLOUD and not cloud_ids.get(mode):
            raise ValueError(f"no Cloud model_id for allowed mode {mode.value}")
    if ModelMode.RTDETR_SPLIT_CLOUD in controller.modes and split_head is None:
        raise ValueError("rtdetr_split_cloud needs split_head (--rtdetr-head-dlc)")
    if probe_period_s and (probe_edge_mode not in detectors or not cloud_ids.get(probe_cloud_mode)):
        raise ValueError(f"probes need a loaded {probe_edge_mode.value} and a {probe_cloud_mode.value} model_id")
    calibrators = calibrators or {}
    probe_tasks, probe_next = set(), [0]
    head_sent = [None]  # Cloud tickets (B tensor, C/D JPEG): the event of the latest frame that started encoding
    state_upload = []
    media_queue, analytics_queue = BoundedLatestQueue(queue_capacity), BoundedLatestQueue(queue_capacity)
    refs = MediaReferenceMap()
    sink = MediaReferenceSink(client)
    sink.start()
    source_done, media_done, control_stop = asyncio.Event(), asyncio.Event(), asyncio.Event()
    stats = Counter()
    reasons = Counter()
    mode_frames = Counter()
    windows = deque(maxlen=32)
    pending_switches = 0
    previous_tracks = None
    tasks = []
    plate_task = []
    slow_task = []

    def advance(now=None):
        nonlocal pending_switches
        now = monotonic_ns() if now is None else now
        if now < monitor.end:
            return
        before = controller.action
        snapshot = monitor.finish(before, now=now, switches=pending_switches)
        pending_switches = 0
        action, reason = controller.decide(snapshot, now=now)
        snapshot["next_action"] = {"model_mode": action.mode.value, "target_fps": action.fps,
                                   "decision_id": action.decision_id, "reason": reason}
        snapshot["context"]["cooldown_remaining_ms"] = max(0, action.since_ns + controller.cooldown_ns - now) / 1e6
        if action.decision_id != before.decision_id:
            pending_switches = 1
            stats["action_changes"] += 1
        windows.append(snapshot)
        emit(snapshot)
        if controller.policy == "cloud" and (not state_upload or state_upload[-1].done()):
            # The Cloud scheduler needs every stream's state; background, at most one in flight, losses are fine.
            state_upload[:] = [asyncio.ensure_future(upload_state(snapshot))]
        if exporter is not None:
            exporter.publish(snapshot)

    async def upload_state(snapshot):
        raw = {k: v for k, v in snapshot["raw"].items()
               if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0}
        try:
            ok = await asyncio.wait_for(client.submit_window_state({
                "control_v1": 1, "stream_key": list(monitor.key), "window_id": snapshot["window_id"],
                "sent_utc_ns": time_ns(), "context": snapshot["context"], "capability": snapshot["capability"],
                "raw": raw}), timeout=1.0)
            stats["state_uploaded" if ok else "state_upload_rejected"] += 1
        except Exception:
            stats["state_upload_failed"] += 1

    async def reference(reference):
        await refs.publish(reference)  # local: preview_pts for results and the web overlay
        if cloud_jpeg_quality is None:
            await sink.submit(reference)
    publisher.on_frame_submitted = reference

    async def capture():
        next_blur = 0
        previous_gray = None
        iterator = source.__aiter__()
        try:
            async for frame in iterator:
                if frame.identity.stream_key != monitor.key:
                    raise ValueError("source stream identity mismatch")
                advance()
                media_queue.put_latest(frame)
                if slow_lane is not None:
                    slow_lane.on_frame(frame)
                stats["captured"] += 1
                if frame.capture_mono_ns >= next_blur:
                    next_blur = frame.capture_mono_ns + 200_000_000
                    def measure():
                        import cv2
                        pixels = frame.pixels
                        height, width = pixels.shape[:2]
                        scene = scene_features(pixels, previous_gray)
                        if width != blur_width:
                            pixels = cv2.resize(pixels, (blur_width, max(1, round(height * blur_width / width))))
                        variance = laplacian_variance(pixels)
                        return variance * (HW_BLUR_SCALE if frame.source_size else 1), scene
                    try:
                        # Motion compares samples 200 ms apart, the same cadence everywhere.
                        variance, (scene, previous_gray) = await asyncio.to_thread(measure)
                        advance()
                        monitor.add("laplacian_variance", variance)
                        for name, value in scene.items():
                            monitor.add(name, value)
                        if blur_normalizer is not None:
                            monitor.add("blur_difficulty_mean", blur_normalizer.difficulty(variance))
                    except Exception as exc:
                        stats["monitor_errors"] += 1
                        print(f"[Control v1 blur] {type(exc).__name__}: {exc}", flush=True)
        finally:
            source_done.set()
            if hasattr(iterator, "aclose"):
                await iterator.aclose()

    async def media():
        cadence = MediaCadence(publisher.profile.fps)
        analytics_cadence = None
        cadence_decision = None
        last_frame = None
        try:
            while not source_done.is_set() or media_queue.pending:
                try:
                    frame = await asyncio.wait_for(media_queue.get(), .05)
                except asyncio.TimeoutError:
                    continue
                if not cadence.accept(frame.cadence_time_ns):
                    stats["media_fps_skipped"] += 1
                    continue
                began = perf_counter_ns()
                await publisher.publish(frame)
                # Time to hand the frame to x264: near 1000/fps means the encoder is the bottleneck.
                monitor.add("encoder_write_ms", (perf_counter_ns() - began) / 1e6)
                last_frame = frame
                if plate is not None:
                    plate.queue.put_latest(frame)
                advance()
                action = controller.action
                if publisher.frames_submitted <= priming_frames:
                    continue
                if action.decision_id != cadence_decision:
                    analytics_cadence = MediaCadence(action.fps)
                    cadence_decision = action.decision_id
                # Select on the media grid, not source time: a 25 fps source thinned to 15 fps media is
                # unevenly spaced, and a cadence restarted at each switch then locked onto 10 fps 1/3 of the time.
                if analytics_cadence.accept(round(publisher.frames_submitted * 1e9 / publisher.profile.fps)):
                    # Bind route at selection, never relabel an old in-flight job.
                    analytics_queue.put_latest((frame, action))
                    stats["analytics_selected"] += 1
            # Keep RTSP alive while the final analytics jobs drain. Padding
            # carries no source inference reference, but encoded_seq advances.
            if last_frame is not None:
                for _ in range(padding_frames):
                    await publisher.publish_padding(last_frame)
                    await asyncio.sleep(1 / publisher.profile.fps)
            media_done.set()
        finally:
            media_done.set()

    local_lock = asyncio.Lock()  # FIFO: Edge results keep frame order across workers

    async def probe(frame, action, result):
        """Both sides' boxes for one frame -> probe_missed. Off the frame path: drops instead of queueing."""
        stats["probe_sent"] += 1
        try:
            edge_boxes = result.boxes if action.mode is probe_edge_mode else None
            cloud_boxes = result.boxes if action.mode is probe_cloud_mode else None
            if edge_boxes is None:
                if local_lock.locked():  # the Edge accelerator is busy with real frames
                    raise CloudInferenceRejected("probe_edge_busy", "Edge model busy")
                async with local_lock:
                    began = perf_counter_ns()
                    edge_boxes, _ms = await asyncio.to_thread(lambda: detectors[probe_edge_mode].infer(unstretch(frame)))
                    monitor.add("probe_edge_ms", (perf_counter_ns() - began) / 1e6)
            if cloud_boxes is None:
                began = perf_counter_ns()
                jpeg = await asyncio.to_thread(encode_cloud_jpeg, frame.pixels, cloud_jpeg_size,
                                               cloud_jpeg_quality or 85)
                monitor.add("probe_jpeg_ms", (perf_counter_ns() - began) / 1e6)
                ticket = CloudInferenceTicket(frame=frame.identity, model_mode=probe_cloud_mode,
                    model_id=cloud_ids[probe_cloud_mode], decision_id=f"probe/{action.decision_id}", preview_pts=0,
                    deadline_utc_ns=time_ns() + round(probe_timeout_s * 1e9), request_id=uuid4().hex, jpeg=jpeg)
                began = perf_counter_ns()
                reply = await asyncio.wait_for(client.submit_cloud_ticket(ticket), probe_timeout_s)
                monitor.add("probe_rtt_ms", (perf_counter_ns() - began) / 1e6)
                cloud_boxes = reply.boxes
            monitor.add("probe_missed", probe_missed(edge_boxes, cloud_boxes))
            monitor.probe_last_ns = monotonic_ns()
            stats["probe_ok"] += 1
        except CloudInferenceRejected as exc:
            stats["probe_dropped"] += 1
            reasons[f"probe_{exc.status}"] += 1
        except (asyncio.TimeoutError, TimeoutError):
            stats["probe_dropped"] += 1
            reasons["probe_timeout"] += 1
        except Exception as exc:
            stats["probe_errors"] += 1
            reasons[f"probe_{type(exc).__name__}"] += 1

    async def infer():
        nonlocal previous_tracks
        while not media_done.is_set() or analytics_queue.pending:
            try:
                frame, action = await asyncio.wait_for(analytics_queue.get_latest(), .05)
            except asyncio.TimeoutError:
                continue
            started = monotonic_ns()
            local = action.mode.placement is Placement.EDGE
            model_id = detectors[action.mode].model_id if local else cloud_ids[action.mode]
            event = {"stream_key": list(monitor.key), "frame_id": frame.identity.source_frame_id,
                     "source_pts": frame.identity.source_pts, "model_id": model_id,
                     "model_mode": action.mode.value, "placement": action.mode.placement.value,
                     "decision_id": action.decision_id,
                     "target_fps": action.fps, "worker_start_mono_ns": started,
                     "capture_mono_ns": frame.capture_mono_ns,
                     "pre_worker_wait_ms": max(0, started - frame.capture_mono_ns) / 1e6,
                     "status": "pending", "timely": False}
            try:
                if started - frame.capture_mono_ns > max_frame_age_s * 1e9:
                    raise CloudInferenceRejected("stale_source", "frame too old before worker admission")
                ref = await refs.wait_for(frame.identity, timeout_s=.25)
                if ref is None:
                    raise CloudInferenceRejected("missing_media_reference")
                request_id = uuid4().hex
                event.update(request_id=request_id, preview_pts=ref.media_pts_ns, encoded_seq=ref.encoded_seq)
                deadline = frame.identity.capture_utc_ns + round(ttl_s * 1e9)
                if local:
                    async with local_lock:
                        boxes, inference_ms = await asyncio.to_thread(lambda: detectors[action.mode].infer(unstretch(frame)))
                    event["edge_inference_ms"] = finite(inference_ms, "edge inference")
                    if monotonic_ns() - frame.capture_mono_ns >= ttl_s * 1e9:
                        raise CloudInferenceRejected("deadline_exceeded", "local inference exceeded deadline")
                    request = DetectionResult(frame=frame.identity, model_mode=action.mode,
                        model_id=model_id, decision_id=action.decision_id, boxes=boxes,
                        inference_ms=inference_ms, produced_utc_ns=time_ns(), request_id=request_id,
                        preview_pts=ref.media_pts_ns, attributes={"media_identity": "pixel-v1",
                            "media_fps": publisher.profile.fps,
                            "media_encoded_seq": ref.encoded_seq, "deadline_utc_ns": deadline})
                    transport_started = monotonic_ns()
                    if edge_tracker is not None:
                        tracked = await asyncio.to_thread(edge_tracker.update, request)
                        result = replace(tracked, boxes=tuple(
                            box if box.track_id is None else replace(box, track_id=box.track_id + EDGE_TRACK_ID_BASE)
                            for box in tracked.boxes), attributes={**tracked.attributes, "tracker": "edge"})
                        await client.publish_tracked_result(result)
                    else:
                        result = await client.submit_detection(request)
                else:
                    jpeg = tensor = b""
                    # Frames encode in parallel (cloud_inflight > 1) but leave in frame order: the Cloud rejects a frame
                    # older than one it has already seen (7/11: 3 out_of_order per minute on C/D without this).
                    previous, mine = head_sent[0], asyncio.Event()
                    head_sent[0] = mine
                    try:
                        if action.mode is ModelMode.RTDETR_SPLIT_CLOUD:
                            tensor, head_stages = await asyncio.to_thread(split_head.encode, frame.pixels)
                            event.update(tensor_bytes=len(tensor), head_prepare_ms=head_stages["prepare_ms"],
                                         head_npu_ms=head_stages["npu_ms"], head_pack_ms=head_stages["pack_ms"])
                        elif cloud_jpeg_quality is not None:
                            began = perf_counter_ns()
                            jpeg = await asyncio.to_thread(encode_cloud_jpeg, frame.pixels, cloud_jpeg_size,
                                                           cloud_jpeg_quality)
                            event.update(jpeg_bytes=len(jpeg), jpeg_encode_ms=(perf_counter_ns() - began) / 1e6)
                            monitor.add("jpeg_kbytes", len(jpeg) / 1000)
                            monitor.add("jpeg_encode_ms", event["jpeg_encode_ms"])
                        if previous is not None:
                            await previous.wait()
                    finally:
                        mine.set()
                    request = CloudInferenceTicket(frame=frame.identity, model_mode=action.mode,
                        model_id=model_id, decision_id=action.decision_id, preview_pts=ref.media_pts_ns,
                        deadline_utc_ns=deadline, request_id=request_id, jpeg=jpeg, tensor=tensor)
                    transport_started = monotonic_ns()
                    result = await client.submit_cloud_ticket(request)
                completed = monotonic_ns()
                round_trip = (completed - transport_started) / 1e6
                if local and edge_tracker is not None:
                    handling = 0.0  # tracked here, no Cloud round trip
                else:
                    from contracts.control_v1 import validate_result
                    result = validate_result(result, request)
                    completed = monotonic_ns()
                    round_trip = (completed - transport_started) / 1e6
                    handling = finite(result.attributes["cloud_handling_ms"], "cloud handling")
                event.update(status="accepted", timely=completed - frame.capture_mono_ns <= ttl_s * 1e9,
                    source_completion_latency_ms=(completed - frame.capture_mono_ns) / 1e6,
                    active_path_latency_ms=(completed - started) / 1e6,
                    transport_overhead_ms=max(0, round_trip - handling), request_round_trip_ms=round_trip,
                    cloud_handling_ms=handling,
                    cloud_processing_ms=result.attributes.get("cloud_e2e_ms"))
                calibrator = calibrators.get(model_id)
                if calibrator is not None:
                    result = calibrate_detection_result(result, calibrator)
                event["calibration_profile_id"] = result.confidence_calibration_id
                event["raw_confidences"] = [box.confidence for box in result.boxes]
                event["calibrated_confidences"] = [box.calibrated_confidence for box in result.boxes
                                                   if box.calibrated_confidence is not None]
                for name, value in box_features(result.boxes).items():
                    monitor.add(name, value)
                tracks = {} if not result.boxes else _tracked_boxes(result)
                if previous_tracks is not None:
                    old_model, old_pts, old_tracks = previous_tracks
                    gap = (result.preview_pts - old_pts) / 1e6
                    if model_id == old_model and 0 < gap <= 500:
                        event["box_change_mean"] = _box_change(old_tracks, tracks)
                        event["box_gap_ms"] = gap
                previous_tracks = (model_id, result.preview_pts, tracks)
                if slow_lane is not None:
                    slow_lane.on_result(frame, result)
                stats["frames_inferred"] += 1
                if probe_period_s and monotonic_ns() >= probe_next[0]:
                    probe_next[0] = monotonic_ns() + round(probe_period_s * 1e9 / probe_frames)
                    task = asyncio.create_task(probe(frame, action, result))
                    probe_tasks.add(task)
                    task.add_done_callback(probe_tasks.discard)
                mode_frames[action.mode.value] += 1
                stats["detections_total"] += len(result.boxes)
                stats["late_completed"] += not event["timely"]
            except CloudInferenceRejected as exc:
                event.update(status=exc.status, error_detail=str(exc)[:500])
                stats["cloud_rejections"] += 1
                reasons[exc.status] += 1
                previous_tracks = None
            except Exception as exc:
                event.update(status=type(exc).__name__, error_detail=str(exc)[:500])
                stats["processing_errors"] += 1
                reasons[type(exc).__name__] += 1
                previous_tracks = None
            finally:
                event["completed_mono_ns"] = monotonic_ns()
                advance()
                monitor.observation(event)

    async def ticker():
        while not control_stop.is_set():
            advance()
            monitor.queue(analytics_queue.pending, analytics_queue.capacity)
            await asyncio.sleep(.05)

    async def resources():
        while resource_sampler is not None and not control_stop.is_set():
            try:
                for name, value in (await asyncio.to_thread(resource_sampler.sample)).items():
                    monitor.add(name, value)
            except Exception as exc:
                stats["resource_errors"] += 1
                print(f"[Control v1 resources] {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(resource_interval_s)

    async def feedback():
        while not control_stop.is_set():
            try:
                value = await asyncio.wait_for(client.get_feedback(
                    stream_key=list(monitor.key) if controller.policy == "cloud" else None), timeout=3)
                if value is not None:
                    monitor.feedback.accept(value)
            except Exception as exc:
                stats["feedback_errors"] += 1
                print(f"[Control v1 feedback] {type(exc).__name__}: {exc}", flush=True)
            await asyncio.sleep(.2)

    async def jobs():
        capture_task = asyncio.create_task(capture())
        media_task = asyncio.create_task(media())
        infer_task = asyncio.gather(*(infer() for _ in range(cloud_inflight)))
        async def drain_guard():
            await media_done.wait()
            try:
                await asyncio.wait_for(asyncio.shield(infer_task), timeout=drain_s)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("control v1 inference drain timeout; run incomplete") from exc
        guard = asyncio.create_task(drain_guard())
        tasks.extend((capture_task, media_task, infer_task, guard))
        if plate is not None:
            plate_task.append(asyncio.create_task(plate.run(media_done)))
            tasks.append(plate_task[0])
        if slow_lane is not None:
            slow_task.append(asyncio.create_task(slow_lane.run(media_done)))
            tasks.append(slow_task[0])
        # FIRST_EXCEPTION prevents a failed publisher from leaving source/worker alive.
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            if task.exception() is not None:
                raise task.exception()
        await asyncio.gather(*pending)

    try:
        async with publisher:
            ticker_task = asyncio.create_task(ticker())
            feedback_task = asyncio.create_task(feedback())
            resource_task = asyncio.create_task(resources())
            job_task = asyncio.create_task(jobs())
            # Tick failures (e.g. log disk full) are explicit, never silently ignored.
            done, _ = await asyncio.wait((ticker_task, job_task), return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is not None:
                    raise task.exception()
            await job_task
            await asyncio.gather(*probe_tasks, return_exceptions=True)  # bounded by probe_timeout_s
            control_stop.set()
            ticker_task.cancel()
            feedback_task.cancel()
            resource_task.cancel()
            await asyncio.gather(ticker_task, feedback_task, resource_task, return_exceptions=True)
            now = monotonic_ns()
            if now > monitor.start and (monitor.events or monitor.samples):
                final = monitor.finish(controller.action, now=now, switches=pending_switches)
                final["partial_window"] = True
                windows.append(final)
                emit(final)
                if exporter is not None:
                    exporter.publish(final)
        await sink.close()
        return {"control_v1": 1, "stream_key": list(monitor.key), "stats": dict(stats),
                "error_reasons": dict(reasons), "media_encoded": publisher.frames_submitted,
                "media_padding_encoded": publisher.padding_frames_submitted,
                "media_queue_dropped": media_queue.dropped,
                "analytics_queue_dropped": analytics_queue.dropped,
                "media_ref_queue_dropped": sink.dropped, "media_ref_send_failed": sink.failed,
                "windows_retained": len(windows), "capacity_drops": monitor.overflow,
                "duplicate_observations": monitor.duplicates, "policy": controller.policy,
                "frames_by_mode": dict(mode_frames),
                "plate": plate_task[0].result() if plate_task else None,
                "slow_lane": slow_task[0].result() if slow_task else None,
                "kafka_background_dropped": getattr(getattr(client, "raw", client), "background_dropped", None),
                "training_ready": False, "quality_constraint_active": False}
    finally:
        control_stop.set()
        for task in tasks + [locals().get(name) for name in ("ticker_task", "feedback_task", "resource_task", "job_task")]:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
        for name in ("ticker_task", "feedback_task", "resource_task", "job_task"):
            task = locals().get(name)
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
        for task in list(probe_tasks) + state_upload:
            task.cancel()
        await asyncio.gather(*state_upload, return_exceptions=True)
        await asyncio.gather(*probe_tasks, return_exceptions=True)
        await sink.close()
        if hasattr(source, "aclose"):
            await source.aclose()


def parser():
    p = argparse.ArgumentParser(description="Opt-in control v1: real telemetry and rule/cycle smoke executor")
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, default=1800)
    p.add_argument("--loop-source", action="store_true",
                   help="replay the clip until --frames (long dataset runs); identity keeps increasing")
    p.add_argument("--decoder", default="", help="hardware decode of --video: v4l2h264dec (246) or qtic2vdec (252); "
                   "empty = OpenCV on the CPU (pixel-identical, about twice the CPU)")
    p.add_argument("--hw-size", type=int, default=0, help="with --decoder: resize on the GPU (qtivtransform) to this "
                   "square size, e.g. 640; ~9x less CPU, RT-DETR F1 -1.1 pt on VN (8/11). 0 = full frames")
    p.add_argument("--slow-tasks", default="", help="slow lane tasks, comma separated (edge_agent/slow_lane.TASKS); "
                   "empty = no slow lane. 'check' proves the BUS/ring/full-resolution plumbing")
    p.add_argument("--camera-profile", default="", help="JSON file of per-camera slow-lane profiles (tasks, "
                   "thresholds, zones; edge_agent/camera_profile.py)")
    p.add_argument("--outbox-dir", default="~/v4_outbox", help="where slow-lane events wait until Kafka confirmed them")
    p.add_argument("--slow-config", default="", help="slow lane thresholds for this camera as JSON, e.g. "
                   "'{\"congestion_vehicles\": 30}' (edge_agent/slow_tasks.DEFAULTS)")
    p.add_argument("--hires-every", type=int, default=5, help="slow lane: full-resolution frame on every Nth frame "
                   "(15 fps / 5 = 3 per s: 6-9 chances per passing vehicle for its plate, 8/11); with --hw-size a "
                   "second GPU branch of the decoder")
    p.add_argument("--main-video", default="", help="prepared H.264 file without B-frames forwarded as the main stream "
                   "without re-encoding (the camera's main stream); analytics keep reading --video")
    p.add_argument("--start-frame", type=int, default=0, help="staged runs: begin this many frames into --video")
    p.add_argument("--mode", choices=[m.value for m in SUPPORTED_MODES], default="dfine_cloud")
    p.add_argument("--allowed-modes", default="", help="comma separated; default initial mode only")
    p.add_argument("--policy", choices=POLICIES, default="observe")
    p.add_argument("--seed", type=int, default=None, help="explore policy RNG seed (recorded)")
    p.add_argument("--actor", default="", help="drl policy: edge-drl-actor-v2 JSON")
    p.add_argument("--capability", default="{}",
                   help='JSON device profile {"yolov8n_fps":..,"yolo11s_fps":..,"encode_fps":..} (measured caps)')
    p.add_argument("--target-fps", type=float, default=15)
    p.add_argument("--fps-levels", default="5,10,15,30")
    p.add_argument("--cooldown-s", type=float, default=3)
    p.add_argument("--window-ms", type=float, default=1000)
    p.add_argument("--latency-cap-ms", type=float, default=1000)
    p.add_argument("--transport-cap-ms", type=float, default=100)
    p.add_argument("--transport", choices=["grpc", "kafka"], default="grpc")
    p.add_argument("--grpc-target", default="192.168.180.249:50061")
    p.add_argument("--kafka-bootstrap", default="192.168.180.95:9092")
    p.add_argument("--node-id", default="edge-246")
    p.add_argument("--camera-id", default="traffic-01")
    p.add_argument("--stream-id", default="control-v1")
    p.add_argument("--stream-epoch", type=int, default=None)
    p.add_argument("--rtsp-base", default="rtsp://192.168.180.249:8554")
    p.add_argument("--http-port", type=int, default=8003, help="Cloud web port, only for the printed live URL")
    p.add_argument("--preview-fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--bitrate-kbps", type=int, default=2500)
    p.add_argument("--queue-capacity", type=int, default=8)
    p.add_argument("--cloud-inflight", type=int, default=1, help="Cloud tickets in flight (analytics workers)")
    p.add_argument("--cloud-frame-source", choices=("rtsp", "jpeg"), default="rtsp",
                   help="jpeg: Cloud tickets carry the frame; rtsp: Cloud decodes the published stream (v49)")
    p.add_argument("--cloud-jpeg-quality", type=int, default=85)
    p.add_argument("--cloud-jpeg-size", type=int, default=640, help="square side = Cloud model input")
    p.add_argument("--probe-period-s", type=float, default=0.0,
                   help="probe window: --probe-frames frames get YOLO11s + D-FINE-X boxes (0 = off)")
    p.add_argument("--probe-frames", type=int, default=2)
    p.add_argument("--priming-frames", type=int, default=30)
    p.add_argument("--padding-frames", type=int, default=30)
    p.add_argument("--ticket-ttl", type=float, default=1.5)
    p.add_argument("--max-frame-age", type=float, default=.5)
    p.add_argument("--rpc-timeout", type=float, default=6)
    p.add_argument("--drain-s", type=float, default=10)
    p.add_argument("--yolo-dlc", default="")
    p.add_argument("--yolo-imgsz", type=int, default=320)
    p.add_argument("--yolo-layout", choices=["nchw", "nhwc"], default="nchw")
    p.add_argument("--confidence", type=float, default=.25)
    p.add_argument("--yolo11s-dlc", default="", help="YOLO11s artifact (DLC, or ONNX on GPU Edges)")
    p.add_argument("--yolo11s-confidence", type=float, default=.1)
    p.add_argument("--yolo11s-nms-iou", type=float, default=.5)
    p.add_argument("--cloud-model-id", default="dfine_full")
    p.add_argument("--cloud-model-ids", default="{}",
                   help='JSON {"rtdetr_cloud": "<artifact model_id>", ...} for Cloud modes')
    p.add_argument("--edge-tracking", action="store_true",
                   help="v4: Edge models track on the Edge (ByteTrack) and publish to Kafka; needs --transport kafka")
    p.add_argument("--rtdetr-head-dlc", default="", help="path B: RT-DETR head, DLC (head_w8a8_pc) on an NPU Edge or head.onnx on a GPU Edge")
    p.add_argument("--blur-width", type=int, default=320)
    p.add_argument("--blur-blurry-variance", type=float, default=None)
    p.add_argument("--blur-sharp-variance", type=float, default=None)
    p.add_argument("--calibration-profiles", default="", help="JSON profiles fitted offline; no guessed defaults")
    p.add_argument("--state-jsonl", default="", help="new exclusive output file, never overwrite a prior run")
    p.add_argument("--metrics-port", type=int, default=9108)
    p.add_argument("--plate-detector", default="", help="plate detector (.dlc/.onnx NPU adapter, .pt ultralytics); "
                   "set = send plate crops of every published frame to the Cloud plate OCR")
    p.add_argument("--plate-imgsz", type=int, default=320)
    p.add_argument("--plate-conf", type=float, default=.3)
    p.add_argument("--plate-target", default="", help="Cloud plate OCR host:port (default: grpc-target host:50062)")
    p.add_argument("--plate-every", type=int, default=3, help="re-read a plate every N frames")
    p.add_argument("--plate-cap", type=int, default=5, help="max reads per plate (0 = no limit)")
    p.add_argument("--plate-confirm", type=int, default=2, help="stop after N equal valid reads (0 = never)")
    p.add_argument("--plate-inflight", type=int, default=16)
    p.add_argument("--plate-log", default="", help="new JSONL file: every plate read and finished plate track")
    p.add_argument("--run-timeout-s", type=float, default=600)
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p


async def run(args):
    from inference import phase7_edge_catalog
    from media.ffmpeg_publisher import FFmpegH264Publisher
    from media.opencv_source import OpenCVFrameSource
    from media.profile import MediaProfile, rtsp_publish_url
    from edge_agent.resources_v2 import ResourceSampler
    mode = ModelMode(args.mode)
    modes = tuple(dict.fromkeys(ModelMode(x.strip()) for x in args.allowed_modes.split(",") if x.strip())) or (mode,)
    if args.policy == "observe":
        modes = (mode,)
    levels = tuple(float(x) for x in args.fps_levels.split(",") if float(x) <= args.preview_fps)
    if not 1 <= args.preview_fps <= 240 or not 0 <= args.metrics_port <= 65535:
        raise ValueError("invalid preview FPS or metrics port")
    if args.frames <= 0 or not 1 <= args.queue_capacity <= 32:
        raise ValueError("positive frame count and queue capacity 1..32 required")
    # Microsecond epoch is unique for sessions while still exactly representable
    # by browser JSON numbers; UTC nanoseconds would exceed JS MAX_SAFE_INTEGER.
    key = (args.node_id, args.camera_id, args.stream_id, args.stream_epoch if args.stream_epoch is not None else time_ns() // 1000)
    capability = json.loads(args.capability)
    if not isinstance(capability, dict):
        raise ValueError("--capability must be a JSON object of positive numbers")
    capability = {str(k): finite(v, k, positive=True) for k, v in capability.items()}
    cloud_ids = json.loads(args.cloud_model_ids)
    if not isinstance(cloud_ids, dict):
        raise ValueError("--cloud-model-ids must be a JSON object")
    cloud_ids = {ModelMode(name): str(value) for name, value in cloud_ids.items()}
    actor = None
    if args.policy == "drl":
        if not args.actor:
            raise ValueError("--policy drl requires --actor")
        from edge_agent.drl_policy import MlpActor
        actor = MlpActor.load(args.actor)
    controller = RuleController(mode, args.target_fps, allowed_modes=modes, fps_levels=levels,
        stream_key=key, policy=args.policy, cooldown_s=args.cooldown_s, seed=args.seed, actor=actor)
    blur = None
    if args.blur_blurry_variance is not None or args.blur_sharp_variance is not None:
        if args.blur_blurry_variance is None or args.blur_sharp_variance is None:
            raise ValueError("set both blur thresholds, calibrated at --blur-width")
        blur = BlurDifficultyNormalizer(finite(args.blur_blurry_variance, "blurry variance"),
                                       finite(args.blur_sharp_variance, "sharp variance"))
    calibrators = {}
    if args.calibration_profiles:
        with open(args.calibration_profiles, encoding="utf-8") as file:
            profiles = json.load(file)
        for profile in profiles:
            calibrator = PlattConfidenceCalibrator(**profile)
            if calibrator.model_id in calibrators:
                raise ValueError("duplicate model calibration profile")
            calibrators[calibrator.model_id] = calibrator
    if args.transport == "grpc":
        from contracts.control_client_v1 import GrpcControlClient
        client = GrpcControlClient(args.grpc_target, timeout_s=args.rpc_timeout)
    else:
        from contracts.control_client_v1 import KafkaControlClient
        client = KafkaControlClient(args.kafka_bootstrap, node_id=args.node_id, timeout_s=args.rpc_timeout)
    exporter = log = plate_client = plate_log = plate = split_head = edge_tracker = None
    try:
        await client.start()
        cam_profile = None
        if args.camera_profile:   # P1: tasks, thresholds and zones of this camera; --slow-tasks still wins if given
            from edge_agent.camera_profile import load_profile
            cam_profile = load_profile(args.camera_profile, key[0], key[1])
            args.slow_tasks = args.slow_tasks or ",".join(cam_profile["tasks"])
        slow_names = {n.strip() for n in args.slow_tasks.split(",")}
        if "plates" in slow_names and not args.plate_detector:
            raise ValueError("--slow-tasks plates needs --plate-detector")
        if args.plate_detector and "plates" not in slow_names:   # v50 plate branch (every frame, gRPC OCR)
            from contracts.plate_ocr_wire import PlateOcrClient
            from edge_agent.plate_branch import PlateBranch, load_plate_detector
            from edge_agent.plate_policy import PlateReadPolicy
            plate_client = PlateOcrClient(args.plate_target or f"{args.grpc_target.rsplit(':', 1)[0]}:50062",
                                          timeout_s=args.rpc_timeout)
            plate_log = open(Path(args.plate_log), "x", encoding="utf-8") if args.plate_log else None
            print(f"[Control v1] Loading plate detector {args.plate_detector}", flush=True)
            plate = PlateBranch(
                await asyncio.to_thread(load_plate_detector, args.plate_detector, imgsz=args.plate_imgsz,
                                        conf=args.plate_conf),
                PlateReadPolicy(every=args.plate_every, cap=args.plate_cap, confirm=args.plate_confirm),
                plate_client, stream_key=key, max_inflight=args.plate_inflight, log=plate_log)
        catalog = phase7_edge_catalog(yolo_dlc=args.yolo_dlc or None, yolo_imgsz=args.yolo_imgsz,
                                     yolo_layout=args.yolo_layout, confidence=args.confidence,
                                     yolo11s_dlc=args.yolo11s_dlc or None, yolo11s_allow_diagnostic=True,
                                     yolo11s_confidence=args.yolo11s_confidence,
                                     yolo11s_nms_iou=args.yolo11s_nms_iou)
        detectors = {}
        if args.edge_tracking:
            if args.transport != "kafka":
                raise ValueError("--edge-tracking publishes to Kafka: use --transport kafka")
            from cloud_core.tracking import CloudByteTrack
            edge_tracker = CloudByteTrack(frame_rate=args.preview_fps)
            await asyncio.to_thread(edge_tracker.prepare)
        if args.rtdetr_head_dlc:
            from inference.rtdetr_head_snpe import open_head
            split_head = await asyncio.to_thread(open_head, args.rtdetr_head_dlc)
        # Every allowed Edge model stays loaded so a switch never waits for model load.
        for edge_mode in (m for m in modes if m.placement is Placement.EDGE):
            print(f"[Control v1] Loading {edge_mode.value} (serial inference, one model per frame)", flush=True)
            detectors[edge_mode] = await asyncio.to_thread(catalog.load, edge_mode)
        monitor = TimeWindowMonitor(key, window_ms=args.window_ms, fps_cap=args.preview_fps,
                                    latency_cap_ms=args.latency_cap_ms, transport_cap_ms=args.transport_cap_ms,
                                    capability=capability)
        # Require a functioning reverse telemetry channel before starting media.
        until = monotonic_ns() + round(args.rpc_timeout * 1e9)
        while monotonic_ns() < until:
            value = await client.get_feedback()
            if value is not None:
                monitor.feedback.accept(value)
                break
            await asyncio.sleep(.1)
        else:
            raise RuntimeError("no control-v1 feedback; check CLOUD_CONTROL_V1 and transport")
        if args.metrics_port:
            exporter = PrometheusV1(key)
            exporter.start(args.metrics_port)
        if args.state_jsonl:
            log = open(Path(args.state_jsonl), "x", encoding="utf-8")
        def emit(value):
            text = json.dumps(value, separators=(",", ":"), allow_nan=False)
            print(f"[Control v1 state] {text}", flush=True)
            if log is not None:
                log.write(text + "\n")
                log.flush()
        source = OpenCVFrameSource(args.video, node_id=key[0], camera_id=key[1], stream_id=key[2],
                                   stream_epoch=key[3], max_frames=args.frames, loop=args.loop_source,
                                   start_frame=args.start_frame, decoder=args.decoder or None, hw_size=args.hw_size,
                                   hires_every=args.hires_every if slow_names & {"plates", "crowd", "check", "fight", "crash"} else 0)
        profile = MediaProfile(width=args.width, height=args.height, fps=args.preview_fps,
                               bitrate_kbps=args.bitrate_kbps, gop_frames=args.preview_fps)
        if args.main_video:  # like a camera's main stream: forwarded as it is, the Edge encodes nothing
            from media.copy_publisher import CopyPublisher
            loop_frames = 0
            if args.loop_source:   # the main file must loop with the analytics source (runs longer than the file)
                import cv2
                cap = cv2.VideoCapture(args.main_video)
                loop_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            publisher = CopyPublisher(profile=profile, stream_key=key, output=rtsp_publish_url(args.rtsp_base, key),
                                      main_video=args.main_video, start_s=args.start_frame / profile.fps,
                                      loop_frames=loop_frames)
        else:
            publisher = FFmpegH264Publisher(profile=profile, stream_key=key,
                output=rtsp_publish_url(args.rtsp_base, key), frame_identity=True)
        host = args.grpc_target.rsplit(":", 1)[0]
        print(f"Live URL: http://{host}:{args.http_port}/v1/streams/{key[0]}/{key[1]}/{key[2]}/{key[3]}/live?fps={profile.fps}", flush=True)
        result = await run_pipeline(source, publisher, client, detectors, controller, monitor,
            cloud_model_id=args.cloud_model_id, cloud_model_ids=cloud_ids, queue_capacity=args.queue_capacity,
            priming_frames=args.priming_frames, padding_frames=args.padding_frames,
            ticket_ttl_s=args.ticket_ttl, max_frame_age_s=args.max_frame_age, drain_s=args.drain_s,
            blur_normalizer=blur, blur_width=args.blur_width, calibrators=calibrators,
            cloud_inflight=args.cloud_inflight,
            cloud_jpeg_quality=args.cloud_jpeg_quality if args.cloud_frame_source == "jpeg" else None,
            cloud_jpeg_size=args.cloud_jpeg_size,
            probe_period_s=args.probe_period_s, probe_frames=args.probe_frames,
            emit=emit, exporter=exporter, resource_sampler=ResourceSampler(), plate=plate, split_head=split_head, edge_tracker=edge_tracker,
            slow_lane=build_slow_lane(args.slow_tasks, sink=getattr(client, "publish_slow", None),
                                      outbox_path=Path(args.outbox_dir).expanduser() / f"{key[0]}_{key[1]}.jsonl",
                                      config={"plate_detector": args.plate_detector, "plate_imgsz": args.plate_imgsz,
                                              "plate_conf": args.plate_conf,
                                              "throttle": lambda: getattr(client, "slow_throttle", 1.0),
                                              **(cam_profile["config"] if cam_profile else {}),
                                              "zones": cam_profile["zones"] if cam_profile else {},
                                              **(json.loads(args.slow_config) if args.slow_config else {})}))
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    finally:
        if split_head is not None:
            split_head.close()
        # every NPU daemon this agent started (YOLO11s, YOLOv8n, plate, head): a leaked one keeps a DSP session
        try:
            from legacy.adapters.npu.snpe_daemon_manager import _DAEMONS
            for handle in list(_DAEMONS.values()):
                handle.close()  # a daemon reused from another process is left alone (close() checks)
        except ImportError:
            pass
        await client.close()
        if plate_client is not None:
            await plate_client.close()
        if plate_log is not None:
            plate_log.close()
        if exporter is not None:
            exporter.close()
        if log is not None:
            log.close()


def main():
    args = parser().parse_args()
    finite(args.run_timeout_s, "run_timeout_s", positive=True)
    if not args._worker:
        # Owned-worker watchdog bounds a hung native adapter without touching
        # shared SNPE daemons or unrelated Python/Cloud processes.
        try:
            completed = subprocess.run([sys.executable, "-u", "-m", "edge_agent.control_v1",
                                        "--_worker", *sys.argv[1:]], timeout=args.run_timeout_s)
        except subprocess.TimeoutExpired:
            raise SystemExit("control v1 TIMEOUT_INCOMPLETE: owned worker stopped, not a successful run")
        raise SystemExit(completed.returncode)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
