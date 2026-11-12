"""Cloud gRPC service: sidecar refs, Edge detections and cached-frame inference."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from time import monotonic_ns, time_ns
from typing import Any, Mapping

from contracts import CloudInferenceTicket, DetectionResult, ModelMode
from contracts.grpc_codec import (
    reference_from_proto, result_from_proto, result_to_proto, ticket_from_proto,
)
from contracts.proto import edge_video_platform_pb2 as pb
from contracts.proto import edge_video_platform_pb2_grpc as pb_grpc

from .frame_cache import CloudFrameCache, FrameLookupStatus
from .result_hub import ResultHub, SubmitStatus
from .rtsp_frame_tap import RTSPFrameTapManager


def decode_ticket_jpeg(data):
    """Ticket JPEG -> BGR uint8 pixels, or None when it does not decode (the size cap is checked on receipt)."""
    import cv2
    import numpy as np
    pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return pixels if pixels is not None and pixels.size else None


class CloudDataPlaneServicer(pb_grpc.EdgeCloudDataPlaneServicer):
    def __init__(
        self, *, hub: ResultHub, frame_cache: CloudFrameCache,
        frame_taps: RTSPFrameTapManager, dfine_adapter: Any = None,
        cloud_adapters: Mapping[ModelMode, Any] | None = None,
        max_cloud_inflight: int = 2,
        benchmark_enabled: bool = False,
        preview_fps: int = 15,
    ) -> None:
        if max_cloud_inflight <= 0:
            raise ValueError("max_cloud_inflight must be positive")
        self.hub = hub
        self.frame_cache = frame_cache
        self.frame_taps = frame_taps
        self.cloud_adapters = dict(cloud_adapters or {})
        if dfine_adapter is not None:
            if ModelMode.DFINE_CLOUD in self.cloud_adapters:
                raise ValueError("D-FINE adapter configured twice")
            self.cloud_adapters[ModelMode.DFINE_CLOUD] = dfine_adapter
        if any(mode.placement.value != "cloud" for mode in self.cloud_adapters):
            raise ValueError("cloud_adapters can only contain Cloud modes")
        self._slots = asyncio.Semaphore(max_cloud_inflight)
        self._live_pending = {}  # stream key -> [(source frame id, done event)] of live tickets in flight
        self.max_cloud_inflight = max_cloud_inflight
        self.preview_fps = preview_fps  # media FPS of JPEG tickets (same PREVIEW_FPS on every Edge)
        self.last_inference = OrderedDict()
        self.benchmark_enabled = benchmark_enabled
        self.benchmark_identities = {}
        self._benchmark_streams = set()
        self._benchmark_pending = 0
        self._benchmark_ordering = None

    def _reserve_benchmark_order(self, request):
        if self._benchmark_ordering is None:
            from benchmark.ordering import BenchmarkOrdering
            self._benchmark_ordering = BenchmarkOrdering()
        return self._benchmark_ordering.reserve(request)

    async def PublishMediaFrameRef(self, request, context):
        try:
            status, message = await self.handle_media_reference(reference_from_proto(request))
            return pb.RpcAck(status=status, message=message)
        except (TypeError, ValueError, RuntimeError) as exc:
            return pb.RpcAck(status="invalid", message=str(exc))

    async def SubmitDetectionResult(self, request, context):
        try:
            status, message, _ = await self.handle_detection_result(result_from_proto(request))
            return pb.RpcAck(status=status, message=message)
        except (TypeError, ValueError, KeyError) as exc:
            return pb.RpcAck(status="invalid", message=str(exc))

    async def handle_media_reference(self, reference):
        await self.frame_taps.publish_reference(reference)
        return "accepted", ""

    async def handle_detection_result(self, result):
        measured = result.decision_id.startswith("benchmark/")
        if measured and not self.benchmark_enabled:
            return "benchmark_disabled", "CLOUD_BENCHMARK=1 required", None
        started = monotonic_ns() if measured else None
        if result.model_mode.placement.value != "edge":
            return "invalid", "only Edge model results use this RPC", None
        if measured:
            if self._benchmark_pending >= 64:
                return "benchmark_overloaded", "max 64 pending benchmark requests", None
            try:
                order = self._reserve_benchmark_order(result)
            except ValueError as exc:
                return "benchmark_order_invalid", str(exc), None
            self._benchmark_pending += 1
            async def track():
                try:
                    if order is not None:
                        await order.wait(result.attributes.get("benchmark_deadline_utc_ns", time_ns() + 10_000_000_000))
                    return await self.hub.submit(result, publish_event=False)
                finally:
                    if order is not None:
                        order.finish()
                    self._benchmark_pending -= 1
            work = asyncio.create_task(track())
            # Keep the tracker lane owned until its native update finishes,
            # even if the RPC caller disconnects or times out.
            work.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
            try:
                outcome = await asyncio.shield(work)
            except (TimeoutError, asyncio.TimeoutError):
                return "deadline_exceeded", "deadline waiting for ordered tracking", None
        else:
            outcome = await self.hub.submit(result, publish_event=True)
        if measured and outcome.result is not None and outcome.status is SubmitStatus.ACCEPTED:
            from benchmark.cloud import finish_result
            tracked = finish_result(outcome.result, started_ns=started,
                                    stages={"cloud_hub_submit_ms": (monotonic_ns() - started) / 1e6,
                                            "cloud_bytetrack_update_ms": outcome.tracker_update_ms},
                                    tracking_completed_ns=outcome.tracking_completed_mono_ns)
            return outcome.status.value, "", tracked
        return outcome.status.value, "", outcome.result

    async def SubmitCloudInference(self, request, context):
        try:
            ticket = ticket_from_proto(request)
            status, result, message = await self.handle_cloud_ticket(ticket)
            reply = pb.DetectionReply(status=status, message=message)
            if result is not None:
                reply.result.CopyFrom(result_to_proto(result))
            return reply
        except (TypeError, ValueError, KeyError) as exc:
            return pb.DetectionReply(status="invalid", message=str(exc))

    async def handle_cloud_ticket(self, ticket):
        if ticket.decision_id.startswith("probe/"):
            return await self._execute_probe(ticket)
        observer = getattr(self, "control_ticket_observer", None)
        if observer is not None:
            adapter = self.cloud_adapters.get(ticket.model_mode)
            if adapter is not None and adapter.model_id == ticket.model_id:
                observer.arrival(ticket)
        measured = ticket.decision_id.startswith("benchmark/") and self.benchmark_enabled
        if not measured:
            # Edges may keep several tickets in flight; ByteTrack (ResultHub) needs each stream's frames in
            # order, so a ticket that finishes early waits (until its deadline) for the earlier ones.
            key, mine = ticket.frame.stream_key, (ticket.frame.source_frame_id, asyncio.Event())
            pending = self._live_pending.setdefault(key, [])
            earlier = [done for frame_id, done in pending if frame_id < mine[0]]
            pending.append(mine)
            try:
                return await self._execute_cloud_ticket(ticket, live_order=earlier)
            finally:
                mine[1].set()
                pending.remove(mine)
                if not pending:
                    del self._live_pending[key]
        # Pipelined benchmark requests have dense ordinals and an ordered
        # tracking gate. Older clients retain one request per stream.
        key = ticket.frame.stream_key
        from benchmark.ordering import ordinal
        legacy = ordinal(ticket) is None
        if (legacy and key in self._benchmark_streams) or self._benchmark_pending >= 64:
            return "benchmark_overloaded", None, "benchmark admission bound reached"
        try:
            order = self._reserve_benchmark_order(ticket)
        except ValueError as exc:
            return "benchmark_order_invalid", None, str(exc)
        if legacy:
            self._benchmark_streams.add(key)
        self._benchmark_pending += 1
        started = monotonic_ns()
        async def execute():
            try:
                return await self._execute_cloud_ticket(ticket, started_ns=started,
                                                        benchmark_order=order)
            finally:
                if order is not None:
                    order.finish()
                if legacy:
                    self._benchmark_streams.discard(key)
                self._benchmark_pending -= 1
        work = asyncio.create_task(execute())
        work.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        return await asyncio.shield(work)

    async def _execute_probe(self, ticket):
        """docs/PLAN_PROBE_VI.md: run the reference model on a JPEG probe frame and return its boxes only (no
        ResultHub, tracking or frame-order gate). Real tickets keep priority: a probe runs only on a free GPU slot."""
        started = monotonic_ns()
        adapter = self.cloud_adapters.get(ticket.model_mode)
        if adapter is None or adapter.model_id != ticket.model_id:
            return "unsupported_model", None, "probe model is not loaded"
        if not ticket.jpeg:
            return "invalid", None, "probe tickets carry their frame as JPEG"
        if self._slots.locked():
            return "probe_busy", None, "no free GPU slot"
        pixels = await asyncio.to_thread(decode_ticket_jpeg, ticket.jpeg)
        if pixels is None:
            return "invalid_jpeg", None, "probe jpeg does not decode to a BGR image"
        remaining = (ticket.deadline_utc_ns - time_ns()) / 1e9
        if remaining <= 0:
            return "deadline_exceeded", None, "probe deadline before inference"
        try:
            await asyncio.wait_for(self._slots.acquire(), remaining)
        except (TimeoutError, asyncio.TimeoutError):
            return "probe_busy", None, "GPU slot not free before the probe deadline"
        try:
            boxes, inference_ms = await asyncio.to_thread(adapter.infer, pixels)
        finally:
            self._slots.release()
        if time_ns() >= ticket.deadline_utc_ns:
            return "deadline_exceeded", None, "probe deadline passed during inference"
        return "accepted", DetectionResult(
            frame=ticket.frame, model_mode=ticket.model_mode, model_id=adapter.model_id,
            decision_id=ticket.decision_id, boxes=boxes, inference_ms=inference_ms, produced_utc_ns=time_ns(),
            request_id=ticket.request_id, preview_pts=ticket.preview_pts,
            attributes={"probe": True, "runtime": getattr(adapter, "runtime", "unknown"),
                        "cloud_e2e_ms": (monotonic_ns() - started) / 1e6}), ""

    async def _execute_cloud_ticket(self, ticket, *, started_ns=None, admission_ms=None,
                                    benchmark_order=None, live_order=()):
        service_started_ns = started_ns or monotonic_ns()
        measured = ticket.decision_id.startswith("benchmark/")
        if measured and not self.benchmark_enabled:
            return "benchmark_disabled", None, "CLOUD_BENCHMARK=1 required"
        adapter = self.cloud_adapters.get(ticket.model_mode)
        if adapter is None:
            return "unsupported_model", None, ticket.model_mode.value
        if measured and ticket.model_id != adapter.model_id:
            return "model_identity_mismatch", None, "benchmark requested model differs from loaded model"
        join_started = monotonic_ns() if measured else None
        if ticket.tensor:
            # Path B: the adapter takes the packed cut tensor itself (inference/triton_detr.py RtdetrSplitAdapter).
            pixels = ticket.tensor
            media_fps = self.preview_fps
            encoded_seq = round(ticket.preview_pts * media_fps / 1e9)
        elif ticket.jpeg:
            pixels = await asyncio.to_thread(decode_ticket_jpeg, ticket.jpeg)
            if pixels is None:
                return "invalid_jpeg", None, "ticket jpeg does not decode to a BGR image"
            # The Edge's media PTS is encoded_seq / fps exactly; the web overlay joins boxes to video on it.
            media_fps = self.preview_fps
            encoded_seq = round(ticket.preview_pts * media_fps / 1e9)
        else:
            lookup = await self.frame_cache.wait_for(ticket)
            if lookup.status is not FrameLookupStatus.AVAILABLE or lookup.frame is None:
                detail = (f"frame={ticket.frame.source_frame_id} pts={ticket.preview_pts} "
                          f"cache={self.frame_cache.diagnostics(ticket.frame.stream_key)}")
                print(f"[Cloud ticket] request={ticket.request_id} "
                      f"status={lookup.status.value} {detail}", flush=True)
                return lookup.status.value, None, detail
            decoded = lookup.frame.decoded
            pixels, media_fps, encoded_seq = decoded.pixels, decoded.media_fps, decoded.encoded_seq
        join_ms = (monotonic_ns() - join_started) / 1e6 if measured else None
        try:
            # Waiting for RTSP or earlier tracking does not occupy a GPU slot (live and benchmark
            # alike): only the adapter call owns the shared model semaphore, queued until the deadline.
            acquired = False
            admission_started = monotonic_ns()
            remaining = (ticket.deadline_utc_ns - time_ns()) / 1e9
            if remaining <= 0:
                return "deadline_exceeded", None, "deadline before Cloud inference"
            await asyncio.wait_for(self._slots.acquire(), remaining)
            acquired = True
            admission_ms = (monotonic_ns() - admission_started) / 1e6
            infer_started = monotonic_ns()
            adapter_stages = {}
            try:
                if measured and callable(getattr(adapter, "infer_detailed", None)):
                    boxes, inference_ms, adapter_stages = await asyncio.to_thread(
                        adapter.infer_detailed, pixels)
                else:
                    boxes, inference_ms = await asyncio.to_thread(
                        adapter.infer, pixels)
            finally:
                if acquired:
                    self._slots.release()
            infer_ms = (monotonic_ns() - infer_started) / 1e6
            if time_ns() >= ticket.deadline_utc_ns:
                return "deadline_exceeded", None, "deadline passed during inference"
            result = DetectionResult(
                frame=ticket.frame, model_mode=ticket.model_mode,
                model_id=getattr(adapter, "model_id", ticket.model_id),
                decision_id=ticket.decision_id,
                boxes=boxes, inference_ms=inference_ms, produced_utc_ns=time_ns(),
                request_id=ticket.request_id, preview_pts=ticket.preview_pts,
                attributes={
                    "runtime": getattr(adapter, "runtime", "unknown"),
                    "requested_model_id": ticket.model_id,
                    "media_identity": "pixel-v1",
                    "media_encoded_seq": encoded_seq,
                    "media_fps": media_fps,
                    "frame_source": "tensor" if ticket.tensor else "jpeg" if ticket.jpeg else "rtsp",
                    # Instrumentation fields are measurements, not policy commands.
                    # This service duration excludes the Edge↔Cloud response path.
                    "cloud_e2e_ms": (monotonic_ns() - service_started_ns) / 1_000_000,
                    # v4 stage split (5/11): wait for an inference slot, adapter call as seen by the service
                    # (thread pool + GIL included), and the adapter's own time
                    "cloud_admission_ms": admission_ms,
                    "cloud_adapter_wall_ms": infer_ms,
                    "cloud_adapter_ms": inference_ms,
                    "cloud_inflight_norm": min(
                        1.0,
                        (self.max_cloud_inflight - self._slots._value)
                        / self.max_cloud_inflight,
                    ),
                },
            )
            reorder_started = monotonic_ns()
            if benchmark_order is not None:
                await benchmark_order.wait(ticket.deadline_utc_ns)
            for earlier in live_order:
                remaining = (ticket.deadline_utc_ns - time_ns()) / 1e9
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(earlier.wait(), remaining)
                except (TimeoutError, asyncio.TimeoutError):
                    break  # the earlier ticket is late; this one still goes (the Hub drops the late one)
            reorder_ms = (monotonic_ns() - reorder_started) / 1e6
            hub_started = monotonic_ns() if measured else None
            outcome = await self.hub.submit(result, deadline_utc_ns=ticket.deadline_utc_ns,
                                            publish_event=not measured)
            if outcome.status is not SubmitStatus.ACCEPTED or outcome.result is None:
                return outcome.status.value, None, "result rejected by ResultHub ordering"
            if measured:
                from benchmark.cloud import finish_result
                from dataclasses import replace
                tracked = finish_result(outcome.result, started_ns=service_started_ns,
                    stages={"cloud_media_join_ms": join_ms, "cloud_adapter_call_ms": infer_ms,
                            "cloud_adapter_reported_ms": inference_ms,
                            "cloud_admission_ms": admission_ms,
                            "cloud_reorder_ms": reorder_ms,
                            "cloud_hub_submit_ms": (monotonic_ns() - hub_started) / 1e6,
                            "cloud_bytetrack_update_ms": outcome.tracker_update_ms,
                            **adapter_stages},
                    tracking_completed_ns=outcome.tracking_completed_mono_ns,
                    identity=self.benchmark_identities.get(ticket.model_mode))
                outcome = replace(outcome, result=tracked)
            if not measured and not ticket.tensor:  # the web snapshot draws on pixels; a tensor ticket has none
                key = ticket.frame.stream_key
                self.last_inference[key] = (pixels, outcome.result)
                self.last_inference.move_to_end(key)
                while len(self.last_inference) > 8:
                    self.last_inference.popitem(last=False)
            return "accepted", outcome.result, ""
        except (TimeoutError, asyncio.TimeoutError):  # distinct classes before Python 3.11 (Cloud runs 3.10)
            return "deadline_exceeded", None, "deadline waiting for inference slot or ordered tracking"
        except Exception as exc:
            return "inference_error", None, str(exc)[:500]
