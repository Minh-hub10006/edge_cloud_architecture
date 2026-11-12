"""Bounded real-inference consumer for the Edge analytics branch."""

from __future__ import annotations

import asyncio
from time import monotonic_ns, perf_counter_ns, time_ns
from typing import Any
from collections.abc import Callable
from uuid import uuid4

from contracts import CloudInferenceTicket, DetectionResult, ModelMode, Placement
from contracts.errors import CloudInferenceRejected
from edge_agent import BoundedLatestQueue, CapturedFrame

from .adapters import FrameDetector
from .frame_refs import MediaReferenceMap


async def run_edge_inference_worker(
    queue: BoundedLatestQueue[CapturedFrame], *, mode: ModelMode,
    client: Any, references: MediaReferenceMap,
    detector: FrameDetector | None = None, decision_id: str = "phase7-fixed",
    cloud_model_id: str = "dfine_full",
    reference_wait_s: float = 0.25, ticket_ttl_s: float = 5.0,
    stop: asyncio.Event,
    latest_only: bool = True, max_frame_age_s: float = 0.5,
    preview_fps: int = 15,
    on_observation: Callable[[CapturedFrame, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if mode.placement is Placement.EDGE and detector is None:
        raise ValueError(f"{mode.value} requires a real local detector")
    if mode.placement is Placement.CLOUD and detector is not None:
        raise ValueError(f"{mode.value} runs on Cloud; no Edge detector is allowed")
    stats = {
        "frames_inferred": 0, "frames_without_media_ref": 0,
        "detections_total": 0, "model_errors": 0,
        "transport_errors": 0, "cloud_rejections": 0,
        "cloud_rejection_reasons": {}, "transport_error_reasons": {},
        "stale_frames_dropped": 0, "monitor_errors": 0,
    }
    while not stop.is_set() or queue.pending:
        try:
            frame = await asyncio.wait_for(queue.get_latest() if latest_only else queue.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        observed: dict[str, Any] | None = ({
            "media_matched": None, "transport_rtt_ms": None,
            "edge_backlog_norm": min(1.0, max(0.0, (queue.last_batch_size - 1) / queue.capacity)),
            "cloud_inflight_norm": None, "edge_inference_ms": None,
            "cloud_e2e_ms": None, "completed": False, "result": None,
        } if on_observation is not None else None)
        try:
            if time_ns() - frame.identity.capture_utc_ns > max_frame_age_s * 1e9:
                stats["stale_frames_dropped"] += 1
                continue
            reference = await references.wait_for(frame.identity, timeout_s=reference_wait_s)
            if reference is None:
                stats["frames_without_media_ref"] += 1
                if mode.placement is Placement.CLOUD:
                    continue
            request_id = uuid4().hex
            if mode.placement is Placement.EDGE:
                assert detector is not None
                try:
                    boxes, inference_ms = await asyncio.to_thread(detector.infer, frame.pixels)
                    if observed is not None:
                        observed["edge_inference_ms"] = float(inference_ms)
                    result = DetectionResult(
                        frame=frame.identity, model_mode=mode,
                        model_id=detector.model_id, decision_id=decision_id,
                        boxes=boxes, inference_ms=inference_ms, produced_utc_ns=time_ns(),
                        request_id=request_id,
                        preview_pts=None if reference is None else reference.media_pts_ns,
                        attributes={"placement": "edge", "media_identity": "pixel-v1",
                                    "media_fps": preview_fps,
                                    **getattr(detector, "result_attributes", {}),
                                    "media_encoded_seq": None if reference is None else reference.encoded_seq},
                    )
                except Exception:
                    stats["model_errors"] += 1
                    continue
                try:
                    if time_ns() > frame.identity.capture_utc_ns + round(ticket_ttl_s * 1e9):
                        stats["stale_frames_dropped"] += 1
                        continue
                    transport_started = perf_counter_ns()
                    await client.submit_detection(result)
                    if observed is not None:
                        observed["transport_rtt_ms"] = (perf_counter_ns() - transport_started) / 1e6
                except Exception:
                    stats["transport_errors"] += 1
                    continue
                stats["detections_total"] += len(result.boxes)
            else:
                assert reference is not None
                ticket = CloudInferenceTicket(
                    frame=frame.identity, model_mode=mode,
                    model_id=cloud_model_id, decision_id=decision_id,
                    preview_pts=reference.media_pts_ns,
                    deadline_utc_ns=frame.identity.capture_utc_ns + round(ticket_ttl_s * 1_000_000_000),
                    request_id=request_id,
                )
                try:
                    transport_started = perf_counter_ns()
                    result = await client.submit_cloud_ticket(ticket)
                    if observed is not None:
                        observed["media_matched"] = True
                    round_trip_ms = (perf_counter_ns() - transport_started) / 1e6
                    result_attributes = getattr(result, "attributes", {})
                    cloud_service_ms = result_attributes.get("cloud_e2e_ms")
                    if cloud_service_ms is not None:
                        if observed is not None:
                            observed["cloud_e2e_ms"] = float(cloud_service_ms)
                        # Approximate transport+broker overhead, excluding the
                        # service time reported by Cloud. This is not one-way RTT.
                        if observed is not None:
                            observed["transport_rtt_ms"] = max(0.0, round_trip_ms - float(cloud_service_ms))
                    if observed is not None:
                        observed["cloud_inflight_norm"] = result_attributes.get("cloud_inflight_norm")
                except CloudInferenceRejected as exc:
                    if observed is not None and exc.status in {
                        "media_missed", "media_disconnected", "pts_mismatch", "evicted", "stale",
                    }:
                        observed["media_matched"] = False
                    stats["cloud_rejections"] += 1
                    reasons = stats["cloud_rejection_reasons"]
                    reasons[exc.status] = reasons.get(exc.status, 0) + 1
                    print(f"[Edge inference] frame={frame.identity.source_frame_id} "
                          f"request={request_id} {exc}", flush=True)
                    continue
                except Exception as exc:
                    stats["transport_errors"] += 1
                    reason = type(exc).__name__
                    reasons = stats["transport_error_reasons"]
                    reasons[reason] = reasons.get(reason, 0) + 1
                    print(f"[Edge transport] frame={frame.identity.source_frame_id} "
                          f"request={request_id} {reason}: {exc}", flush=True)
                    continue
                stats["detections_total"] += len(result.boxes)
            if observed is not None:
                observed["completed"] = True
                observed["result"] = result
            stats["frames_inferred"] += 1
        finally:
            if on_observation is not None and observed is not None:
                try:
                    on_observation(frame, observed)
                except Exception as exc:
                    stats["monitor_errors"] += 1
                    print(f"[Edge monitor] frame={frame.identity.source_frame_id} "
                          f"{type(exc).__name__}: {exc}", flush=True)
    return stats
