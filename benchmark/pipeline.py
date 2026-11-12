"""Bounded, opt-in pipeline runner using production capture/media/adapters.

Capacity has blocking backpressure and FIFO inference. Realtime has paced
offered frames and drop-oldest analytics. The completion point is the Cloud
ByteTrack on Cloud. Browser publication is disabled for measured requests.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from collections import deque
import json
from pathlib import Path
from time import monotonic_ns, perf_counter_ns, time_ns  # perf_counter: Windows monotonic ticks ~16 ms
from typing import Any
from uuid import uuid4

from contracts import CloudInferenceTicket, DetectionResult, ModelMode
from contracts.errors import CloudInferenceRejected
from .config import RunConfig, finite_nonnegative
from .model import source_revision
from .resources import ResourceSampler
from .summarize import summarize, write_report
from .trace import JsonlWriter, frame_event


@dataclass(frozen=True)
class PipelineOptions:
    fps: int = 30
    queue_depth: int = 16
    priming_frames: int = 30
    padding_frames: int = 15
    rpc_timeout_s: float = 10
    frame_ttl_s: float = 10
    phase_timeout_s: float = 600
    resource_sample_s: float = 1
    max_inflight: int = 1
    split_preprocess: bool = True

    def __post_init__(self):
        for name, low, high in (("fps", 1, 240), ("queue_depth", 1, 32),
                                ("priming_frames", 0, 240), ("padding_frames", 0, 240),
                                ("max_inflight", 1, 32)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"invalid {name}")
        if self.max_inflight > self.queue_depth:
            raise ValueError("max_inflight must not exceed queue_depth")
        for name in ("rpc_timeout_s", "frame_ttl_s", "phase_timeout_s"):
            if finite_nonnegative(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        value = finite_nonnegative(self.resource_sample_s, "resource_sample_s")
        if value != 0 and value < .1:
            raise ValueError("resource sampling must be disabled or >=0.1s")


async def measure_pipeline(config: RunConfig, source, publisher, client,
                           directory: Path, *, options: PipelineOptions,
                           mode: ModelMode, detector=None, identity: dict[str, Any] | None = None,
                           media_config: dict[str, Any] | None = None, ready=None) -> dict[str, Any]:
    if config.profile not in {"pipeline_capacity", "realtime"} or mode.placement.value != config.placement:
        raise ValueError("pipeline profile and placement required")
    if (detector is not None) != (config.placement == "edge"):
        raise ValueError("Edge needs detector; Cloud must not run an Edge detector")
    if publisher is None and config.placement == "cloud":
        raise ValueError("Cloud inference requires RTSP input media")
    if detector is not None and detector.model_id != config.model_key:
        raise ValueError("loaded Edge model_id differs from requested model_key")
    directory.mkdir(parents=True, exist_ok=False)
    identity = dict(identity or {})
    manifest = {"schema_version": 1, "config": config.as_dict(), "code": source_revision(),
                "identity": identity, "media_config": media_config or {},
                "options": options.__dict__, "tracking": True, "monitor": False,
                "completion": "Cloud ByteTrack completed; subscriber/browser publish disabled",
                "media_mode": "rtsp" if publisher is not None else "none",
                "capacity_policy": "unpaced file, bounded FIFO/backpressure, pipelined requests, ordered Cloud tracking",
                "realtime_policy": "paced offered fps, bounded drop-oldest analytics, warmup FIFO",
                "stages": {"ffmpeg_write_ms": "resize/marker + stdin/drain, NOT codec encode time",
                           "reference_ack_ms": "Cloud ACK (grpc) or broker ACK (kafka), NOT frame availability",
                           "request_rtt_ms": "Edge monotonic submit to Cloud completed reply",
                           "e2e_ms": "Edge capture to tracked reply; includes queue/media and reply transport",
                           "cloud_hub_submit_ms": "hub lock + ordering + tracking; browser publish disabled",
                           "cloud_bytetrack_update_ms": "Cloud tracker update including thread dispatch",
                           "edge_adapter_call_ms": "SNPE/decode/normalize when split; full adapter otherwise",
                           "edge_adapter_reported_ms": "adapter work including preparation, excludes stage queues",
                           "bytetrack_completion_fps": "(N-1)/(last-first tracked completion), Cloud monotonic only"}}
    with (directory / "manifest.json").open("x", encoding="utf-8") as target:
        json.dump(manifest, target, indent=2, ensure_ascii=False, allow_nan=False)
    writer = JsonlWriter(directory / "trace.jsonl")
    setup = JsonlWriter(directory / "setup.jsonl")
    sampler = None
    iterator = source.__aiter__()
    tasks = []
    fatal = None
    started = finished = None
    observed_cloud_identity = None
    last_frame = None
    last_ref = None
    emitted = {"warmup": 0, "measure": 0}
    phase_failures = {"warmup": 0, "measure": 0}
    offer_started = offer_finished = None
    offered = 0
    tracking_first = tracking_last = None
    tracking_count = 0
    tracking_ordered = True
    decision_id = f"benchmark/{config.campaign_id}/{config.run_id}"

    async def bounded(awaitable):
        return await asyncio.wait_for(awaitable, timeout=options.rpc_timeout_s)

    async def native_call(function, argument):
        # Cancellation cannot stop a native session in a worker thread. Drain
        # it before reusing/closing session scratch; the external supervisor
        # remains responsible for terminating a genuinely hung native process.
        task = asyncio.create_task(asyncio.to_thread(function, argument))
        try:
            return await bounded(asyncio.shield(task))
        except BaseException:
            await asyncio.shield(asyncio.gather(task, return_exceptions=True))
            raise

    async def next_frame():
        nonlocal last_frame
        try:
            last_frame = await iterator.__anext__()
            return last_frame
        except StopAsyncIteration as exc:
            raise RuntimeError("source EOF before requested warmup + measured frames") from exc

    async def publish(frame):
        nonlocal last_ref
        if publisher is None:
            return None, {}
        began = perf_counter_ns()
        ref = await bounded(publisher.publish(frame))
        last_ref = ref
        write_ms = (perf_counter_ns() - began) / 1e6
        began = perf_counter_ns()
        await bounded(client.publish_media_reference(ref))
        return ref, {"ffmpeg_write_ms": write_ms,
                     "reference_ack_ms": (perf_counter_ns() - began) / 1e6}

    def record(phase, item, status, *, error=None, request_id=None, result=None, finished_ns=None):
        nonlocal tracking_first, tracking_last, tracking_count, tracking_ordered
        frame, ref, stages, enqueued_ns, prepared = item
        end = finished_ns or monotonic_ns()
        stages["e2e_ms"] = (end - frame.capture_mono_ns) / 1e6
        event = frame_event(config, frame_id=frame.identity.source_frame_id, phase=phase,
                            status=status, started_ns=frame.capture_mono_ns, finished_ns=end,
                            durations_ms=stages, detections=len(result.boxes) if result else 0,
                            error=error)
        event.update(camera_id=frame.identity.camera_id, stream_id=frame.identity.stream_id,
                     stream_epoch=frame.identity.stream_epoch, source_pts=frame.identity.source_pts,
                     preview_pts=ref.media_pts_ns if ref is not None else None,
                     encoded_seq=ref.encoded_seq if ref is not None else None,
                     request_id=request_id, completion=result.attributes.get("benchmark_completion") if result else None)
        if phase == "measure" and result is not None:
            completed_ns = result.attributes.get("benchmark_tracking_completed_mono_ns")
            if type(completed_ns) is int and completed_ns > 0:
                tracking_ordered &= tracking_last is None or completed_ns > tracking_last
                tracking_first = tracking_first or completed_ns
                tracking_last = completed_ns
                tracking_count += 1
                event["cloud_tracking_completed_mono_ns"] = completed_ns
        writer.write(event)
        emitted[phase] += 1
        phase_failures[phase] += status != "ok"

    async def phase(name: str, count: int):
        nonlocal tasks, observed_cloud_identity
        if not count:
            return
        capture_queue = asyncio.Queue(maxsize=options.queue_depth)
        queue = asyncio.Queue(maxsize=options.queue_depth)
        ref_queue = asyncio.Queue(maxsize=options.queue_depth)
        permits = asyncio.Semaphore(options.queue_depth)
        done = asyncio.Event()
        paced = config.profile == "realtime"
        lossless = config.profile == "pipeline_capacity" or name == "warmup"
        due = monotonic_ns()
        split = (detector is not None and options.split_preprocess
                 and callable(getattr(detector, "prepare_frame", None))
                 and callable(getattr(detector, "infer_prepared", None)))

        async def enqueue(target, item):
            if not lossless and target.full():
                old = target.get_nowait()
                record(name, old, "dropped", error="analytics_backlog")
            await target.put(item)

        async def producer():
            # Always hand the end sentinel downstream, also on source/prepare
            # failure, so media/reference/consumer workers finish instead of
            # waiting on an empty queue until the phase timeout.
            try:
                await produce()
            finally:
                target = capture_queue if publisher is not None else queue
                try:
                    target.put_nowait(None)
                except asyncio.QueueFull:
                    # A cancelled phase cancels the workers too; never block cleanup.
                    if not asyncio.current_task().cancelling():
                        await target.put(None)

        async def produce():
            nonlocal offer_started, offer_finished, offered
            for index in range(count):
                if lossless:
                    await permits.acquire()
                if paced:
                    delay = (due + round(index * 1e9 / options.fps) - monotonic_ns()) / 1e9
                    if delay > 0:
                        await asyncio.sleep(delay)
                read_started = perf_counter_ns()
                frame = await next_frame()
                if name == "measure":
                    offer_started = offer_started or frame.capture_mono_ns
                    offer_finished = frame.capture_mono_ns
                    offered += 1
                read_ms = (perf_counter_ns() - read_started) / 1e6
                stages = {"source_next_ms": read_ms}
                prepared = await native_call(detector.prepare_frame, frame.pixels) if split else None
                if prepared is not None:
                    stages.update(prepared.stages_ms)
                item = (frame, None, stages, monotonic_ns(), prepared)
                await enqueue(capture_queue if publisher is not None else queue, item)

        async def media_worker():
            nonlocal last_ref
            while (item := await capture_queue.get()) is not None:
                frame, _, stages, enqueued_ns, prepared = item
                began = perf_counter_ns()
                ref = await bounded(publisher.publish(frame))
                last_ref = ref
                stages["ffmpeg_write_ms"] = (perf_counter_ns() - began) / 1e6
                stages.update(getattr(publisher, "last_publish_stages", {}))
                await ref_queue.put((frame, ref, stages, enqueued_ns, prepared))
            await ref_queue.put(None)
            # Keep RTSP alive while the decoder exposes the last real frames.
            for _ in range(options.padding_frames if publisher is not None else 0):
                if done.is_set():
                    break
                await bounded(publisher.publish_padding(last_frame))
                await asyncio.sleep(1 / options.fps)

        async def reference_worker():
            while (item := await ref_queue.get()) is not None:
                began = perf_counter_ns()
                await bounded(client.publish_media_reference(item[1]))
                item[2]["reference_ack_ms"] = (perf_counter_ns() - began) / 1e6
                await enqueue(queue, item)
            await queue.put(None)

        async def consumer():
            nonlocal observed_cloud_identity
            pending = deque()
            sequence = 0
            requests = set()

            async def complete(item, request, begin):
                nonlocal observed_cloud_identity
                stages = item[2]
                result = None
                status, error = "ok", None
                try:
                    method = client.submit_detection if detector is not None else client.submit_cloud_ticket
                    result = await bounded(method(request))
                    from .client import MeasuredClient
                    MeasuredClient.check(request, result)
                    remote = result.attributes.get("benchmark_stages_ms", {})
                    if not isinstance(remote, dict) or any(not k.startswith("cloud_") for k in remote):
                        raise ValueError("invalid Cloud stage names")
                    stages.update({k: finite_nonnegative(v, k) if v is not None else None
                                   for k, v in remote.items()})
                    if config.placement == "cloud":
                        current = result.attributes.get("benchmark_model_identity", {})
                        if observed_cloud_identity is not None and current != observed_cloud_identity:
                            raise ValueError("Cloud artifact/runtime identity changed within run")
                        observed_cloud_identity = current
                except CloudInferenceRejected as exc:
                    status = "timeout" if exc.status == "deadline_exceeded" else "error"
                    error = f"cloud:{exc.status}: {exc}"
                except TimeoutError as exc:
                    status, error = "timeout", str(exc) or "request timeout"
                except Exception as exc:
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                stages["request_rtt_ms"] = (perf_counter_ns() - begin) / 1e6
                service = result.attributes.get("benchmark_cloud_service_ms") if status == "ok" else None
                if service is not None:
                    # Network + gRPC serialization: RTT minus the Cloud's own service time.
                    stages["cloud_service_ms"] = finite_nonnegative(service, "cloud_service_ms")
                    stages["transport_overhead_ms"] = max(0.0, stages["request_rtt_ms"] - stages["cloud_service_ms"])
                # Preserve the actual reply endpoint time even when records
                # are written later in dispatch order.
                return item, status, error, request.request_id, result, monotonic_ns()

            async def collect():
                item, status, error, request_id, result, completed_ns = await pending.popleft()
                record(name, item, status, error=error, request_id=request_id,
                       result=result if status == "ok" else None, finished_ns=completed_ns)
                if lossless:
                    permits.release()

            try:
              while (item := await queue.get()) is not None:
                frame, ref, stages, enqueued_ns, prepared = item
                stages["analytics_queue_ms"] = (monotonic_ns() - enqueued_ns) / 1e6
                request_id = f"bp1-{sequence}-{uuid4().hex}"
                try:
                    deadline = frame.identity.capture_utc_ns + round(options.frame_ttl_s * 1e9)
                    if time_ns() >= deadline:
                        if lossless:
                            raise TimeoutError("frame expired before request")
                        record(name, item, "dropped", error="stale_frame")
                        continue
                    else:
                        if detector is not None:
                            begin = perf_counter_ns()
                            try:
                                if prepared is not None:
                                    boxes, infer_ms, detail = await native_call(detector.infer_prepared, prepared)
                                    stages.update(detail)
                                else:
                                    boxes, infer_ms = await native_call(detector.infer, frame.pixels)
                            finally:
                                stages["edge_adapter_call_ms"] = (perf_counter_ns() - begin) / 1e6
                            stages["edge_adapter_reported_ms"] = finite_nonnegative(infer_ms, "inference_ms")
                            request = DetectionResult(frame=frame.identity, model_mode=mode,
                                model_id=detector.model_id, decision_id=decision_id + "/" + name, boxes=boxes,
                                inference_ms=infer_ms, produced_utc_ns=time_ns(), request_id=request_id,
                                preview_pts=ref.media_pts_ns if ref is not None else None,
                                attributes={"benchmark_deadline_utc_ns": deadline,
                                            **getattr(detector, "result_attributes", {})})
                        else:
                            request = CloudInferenceTicket(frame=frame.identity, model_mode=mode,
                                model_id=config.model_key, decision_id=decision_id + "/" + name,
                                preview_pts=ref.media_pts_ns, deadline_utc_ns=deadline, request_id=request_id)
                        begin = perf_counter_ns()
                        task = asyncio.create_task(complete(item, request, begin))
                        requests.add(task)
                        task.add_done_callback(requests.discard)
                        pending.append(task)
                        sequence += 1
                        if len(pending) >= options.max_inflight:
                            await collect()
                except Exception as exc:
                    record(name, item, "timeout" if isinstance(exc, TimeoutError) else "error",
                           error=f"{type(exc).__name__}: {exc}", request_id=request_id)
                    if lossless:
                        permits.release()
              while pending:
                await collect()
              done.set()
            finally:
                for task in requests:
                    task.cancel()
                await asyncio.gather(*requests, return_exceptions=True)

        tasks = [asyncio.create_task(producer()), asyncio.create_task(consumer())]
        if publisher is not None:
            tasks.extend((asyncio.create_task(media_worker()), asyncio.create_task(reference_worker())))
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=options.phase_timeout_s)
        if name == "warmup" and phase_failures[name]:
            raise RuntimeError("warmup failed; measurement not started (see trace)")

    try:
        if options.resource_sample_s:
            sampler = ResourceSampler(directory / "resources-edge.jsonl", config.node_id,
                                      options.resource_sample_s)
            sampler.start()
        if publisher is not None:
            await bounded(publisher.start())
        for index in range(options.priming_frames if publisher is not None else 0):
            frame = await next_frame()
            ref, stages = await publish(frame)
            setup.write({"schema_version": 1, "phase": "media_priming", "source_frame_id": frame.identity.source_frame_id,
                         "encoded_seq": ref.encoded_seq, "stages_ms": stages})
            await asyncio.sleep(1 / options.fps)
        await phase("warmup", config.warmup_frames)
        if ready is not None:
            # A slow peer at READY must not starve this stream's RTSP publisher.
            barrier = asyncio.create_task(ready(config))
            try:
                while not barrier.done():
                    if publisher is not None and last_frame is not None:
                        await bounded(publisher.publish_padding(last_frame))
                        if last_ref is not None:
                            await bounded(client.publish_media_reference(last_ref))
                    await asyncio.wait((barrier,), timeout=1 / options.fps)
                await barrier
            finally:
                if not barrier.done():
                    barrier.cancel()
                await asyncio.gather(barrier, return_exceptions=True)
        started = monotonic_ns()
        await phase("measure", config.measured_frames)
        finished = monotonic_ns()
    except Exception as exc:
        fatal = f"{type(exc).__name__}: {exc}"
        finished = monotonic_ns() if started is not None else None
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for resource in (publisher, client):
            if resource is None:
                continue
            try:
                await bounded(resource.close())
            except Exception as exc:
                fatal = (fatal + "; " if fatal else "") + f"close:{exc}"
        if hasattr(iterator, "aclose"):
            try:
                await bounded(iterator.aclose())
            except Exception as exc:
                fatal = (fatal + "; " if fatal else "") + f"source_close:{exc}"
        for resource in (sampler, setup, writer):
            if resource:
                try:
                    resource.close()
                except Exception as exc:
                    fatal = (fatal + "; " if fatal else "") + f"logger_close:{exc}"
    elapsed_s = (finished - started) / 1e9 if started is not None and finished is not None else 0
    verified = bool(identity.get("video_sha256"))
    if config.placement == "edge":
        verified &= bool(identity.get("artifact_sha256") and identity.get("runtime"))
    else:
        verified &= bool(observed_cloud_identity and observed_cloud_identity.get("artifact_sha256")
                         and observed_cloud_identity.get("runtime")
                         and observed_cloud_identity.get("model_id") == config.model_key)
    try:
        report = summarize(directory / "trace.jsonl", config, elapsed_s, fatal_error=fatal,
                           identity_verified=verified)
    except Exception as exc:
        # Keep damaged raw logs; never publish an apparently valid measurement.
        report = {"schema_version": 1, **config.as_dict(), "elapsed_s": elapsed_s,
                  "completed": 0, "execution_status": "FAIL", "quality_status": "NOT_EVALUATED",
                  "gates": {"execution": False, "measurement": False,
                            "artifact_runtime_identity": verified, "quality": None},
                  "stages_ms": {}, "fatal_error": f"{fatal or ''}; trace:{type(exc).__name__}: {exc}"}
    report.update(measurement_interval={"started_mono_ns": started, "finished_mono_ns": finished,
                                        "clock_scope": config.node_id},
                  cloud_identity=observed_cloud_identity, emitted=emitted,
                  completion_endpoint="cloud_bytetrack",
                  media_mode="rtsp" if publisher is not None else "none",
                  input_pacing="unpaced" if config.profile == "pipeline_capacity" else "realtime",
                  media_padding_encoded=getattr(publisher, "padding_frames_submitted", 0),
                  offered_frames=offered,
                  offered_fps=((offered - 1) * 1e9 / (offer_finished - offer_started)
                               if offered > 1 and offer_finished > offer_started else None),
                  offered_interval={"first_capture_mono_ns": offer_started,
                                    "last_capture_mono_ns": offer_finished})
    # This throughput uses only the Cloud clock at the requested endpoint.
    # Edge effective_fps remains the conservative full-run rate including reply/drain.
    valid_completion_interval = (tracking_count == report["completed"]
                                 and tracking_count > 1 and tracking_ordered)
    report["bytetrack_completion_fps"] = (
        (tracking_count - 1) * 1e9 / (tracking_last - tracking_first)
        if valid_completion_interval else None)
    report["bytetrack_completion_interval"] = {
        "first_mono_ns": tracking_first, "last_mono_ns": tracking_last,
        "count": tracking_count, "clock_scope": "cloud_service"}
    write_report(directory, report)
    return report
