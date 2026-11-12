"""Opt-in global Cloud telemetry and measured tracked-result endpoint."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import replace
from time import monotonic_ns, time_ns
from uuid import uuid4

from contracts.control_v1 import SERVICE, VERSION, encode, decode, finite
from contracts.grpc_codec import result_from_proto, result_to_proto, ticket_from_proto
from contracts.proto import edge_video_platform_pb2 as pb


class CloudTelemetry:
    """Fixed monotonic windows, aggregate all nodes, bounded retry dedup.

    Costs refer to measured *server* capacities, not guessed GPU utilization.
    Unknown model capacity makes a nonempty window's pressure unavailable.
    """
    def __init__(self, service, capacities=None, *, window_ms=1000, dedup_capacity=8192,
                 clock=monotonic_ns):
        self.service = service
        if any(not isinstance(key, str) or not key for key in (capacities or {})):
            raise ValueError("capacity keys must be nonempty artifact model IDs")
        self.capacities = {key: finite(value, key, positive=True)
                           for key, value in (capacities or {}).items()}
        self.window_ms = finite(window_ms, "window_ms", positive=True)
        if dedup_capacity <= 0:
            raise ValueError("dedup_capacity must be positive")
        self.dedup_capacity = dedup_capacity
        self.clock = clock
        self.instance_id = uuid4().hex
        self.sequence = 0
        self.started = clock()
        self.latest = None
        self.seen = OrderedDict()
        self.reset()

    def reset(self):
        self.work_s = 0.0
        self.tickets = self.unknown = 0
        self.inflight_samples = []

    def sample(self):
        now = self.clock()
        self.roll(now)
        slots = self.service.max_cloud_inflight
        self.inflight_samples.append(min(1.0, max(0.0, (slots - self.service._slots._value) / slots)))
        # Event-loop stalls must not grow an unbounded sample array.
        if len(self.inflight_samples) > 4096:
            self.inflight_samples = self.inflight_samples[-4096:]

    def arrival(self, ticket):
        self.roll(self.clock())
        if not ticket.request_id:
            return False
        key = (*ticket.frame.stream_key, ticket.request_id)
        now = self.clock()
        # Keep identities until the ticket deadline; a later replay is expired.
        for old, expiry in list(self.seen.items()):
            if expiry <= now:
                self.seen.pop(old, None)
        if key in self.seen:
            return False
        remaining = ticket.deadline_utc_ns - time_ns()
        if remaining <= 0:
            return False
        if len(self.seen) >= self.dedup_capacity:
            raise RuntimeError("control telemetry dedup capacity exceeded")
        self.seen[key] = now + remaining
        self.tickets += 1
        capacity = self.capacities.get(ticket.model_id)
        if capacity is None:
            self.unknown += 1
        else:
            self.work_s += 1 / capacity
        return True

    def roll(self, now=None):
        now = self.clock() if now is None else now
        elapsed = now - self.started
        if elapsed < self.window_ms * 1e6:
            return
        seconds = elapsed / 1e9
        raw = None if self.unknown else self.work_s / seconds
        self.sequence += 1
        self.latest = {
            "control_v1": VERSION, "cloud_instance_id": self.instance_id,
            "sequence": self.sequence, "window_ms": elapsed / 1e6,
            "measurement_to_send_ms": 0.0,
            "cloud_inflight_norm": (sum(self.inflight_samples) / len(self.inflight_samples)
                                     if self.inflight_samples else None),
            "arrival_rate_rps": self.tickets / seconds,
            "cloud_arrival_pressure_raw": raw,
            "cloud_arrival_pressure_norm": None if raw is None else min(1.0, raw),
            "unknown_capacity_tickets": self.unknown,
        }
        self.latest_end_ns = now
        self.started = now
        self.reset()

    def snapshot(self):
        self.roll(self.clock())
        if self.latest is None:
            return None
        return {**self.latest, "measurement_to_send_ms":
                (self.clock() - self.latest_end_ns) / 1e6}


class ControlService:
    """Delegate inference/media to existing service, preserve native work on RPC cancellation."""
    def __init__(self, core, telemetry, *, max_tasks=64, registry=None, scheduler=None):
        self.core, self.telemetry = core, telemetry
        # docs/PLAN_CONTROL_V3_VI.md: latest window state per stream, and the per-stream command source.
        self.registry, self.scheduler = registry, scheduler
        self.max_tasks = max_tasks
        self.tasks = set()

    async def _owned(self, operation):
        if len(self.tasks) >= self.max_tasks:
            operation.close()
            raise RuntimeError("control request capacity exceeded")
        task = asyncio.create_task(operation)
        self.tasks.add(task)
        def done(task):
            self.tasks.discard(task)
            if not task.cancelled():
                task.exception()
        task.add_done_callback(done)
        return await asyncio.shield(task)

    async def handle_media_reference(self, reference):
        return await self.core.handle_media_reference(reference)

    def finish(self, result, started, completion="result_hub_accepted"):
        if result is None:
            return None
        return replace(result, attributes={**result.attributes,
            "control_v1": VERSION, "completion": completion,
            "cloud_handling_ms": (monotonic_ns() - started) / 1e6})

    async def handle_detection_result(self, result):
        async def execute():
            started = monotonic_ns()
            if not result.request_id or result.model_mode.placement.value != "edge":
                return "invalid", "requires Edge result and request identity", None
            if result.decision_id.startswith("benchmark/"):
                return "invalid", "benchmark uses its own endpoint", None
            deadline = result.attributes.get("deadline_utc_ns")
            if type(deadline) is not int or deadline <= result.frame.capture_utc_ns:
                return "invalid", "requires a valid source-frame deadline", None
            outcome = await self.core.hub.submit(result, deadline_utc_ns=deadline)
            status = outcome.status.value
            return status, "", self.finish(outcome.result, started) if status == "accepted" else None
        return await self._owned(execute())

    async def handle_cloud_ticket(self, ticket):
        async def execute():
            started = monotonic_ns()
            adapter = self.core.cloud_adapters.get(ticket.model_mode)
            if (not ticket.request_id or adapter is None or adapter.model_id != ticket.model_id
                    or ticket.decision_id.startswith("benchmark/")):
                return "invalid", None, "unsupported model/artifact/request identity"
            if ticket.decision_id.startswith("probe/"):  # measurement, not policy demand: no arrival telemetry
                status, result, detail = await self.core.handle_cloud_ticket(ticket)
                return status, self.finish(result, started, "probe_reply") if status == "accepted" else None, detail
            self.telemetry.arrival(ticket)  # includes valid demand later rejected as overloaded
            status, result, detail = await self.core.handle_cloud_ticket(ticket)
            return status, self.finish(result, started) if status == "accepted" else None, detail
        return await self._owned(execute())

    async def close(self):
        if self.tasks:
            done, pending = await asyncio.wait(tuple(self.tasks), timeout=5)
            if pending:
                print(f"[Control v1] {len(pending)} owned Cloud jobs still draining", flush=True)


def install_grpc(server, service):
    import grpc
    async def edge(request, context):
        try:
            status, detail, result = await service.handle_detection_result(result_from_proto(request))
            reply = pb.DetectionReply(status=status, message=detail)
            if result is not None:
                reply.result.CopyFrom(result_to_proto(result))
            return reply
        except (TypeError, ValueError, RuntimeError) as exc:
            return pb.DetectionReply(status="invalid", message=str(exc))
    async def cloud(request, context):
        try:
            status, result, detail = await service.handle_cloud_ticket(ticket_from_proto(request))
            reply = pb.DetectionReply(status=status, message=detail)
            if result is not None:
                reply.result.CopyFrom(result_to_proto(result))
            return reply
        except (TypeError, ValueError, RuntimeError) as exc:
            return pb.DetectionReply(status="invalid", message=str(exc))
    async def feedback(request, context):
        value = service.telemetry.snapshot()
        key = request.get("stream_key") if isinstance(request, dict) else None
        if value is not None and key and service.scheduler is not None:
            value = {**value, "command": service.scheduler.command_for(key)}
        return value
    async def window_state(request, context):
        if service.registry is None:
            return {"status": "no_scheduler"}
        try:
            service.registry.update(request)
            return {"status": "accepted"}
        except (TypeError, ValueError) as exc:
            return {"status": "invalid", "message": str(exc)[:200]}
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE, {
        "SubmitEdgeDetection": grpc.unary_unary_rpc_method_handler(edge,
            request_deserializer=pb.DetectionResult.FromString, response_serializer=pb.DetectionReply.SerializeToString),
        "SubmitCloudInference": grpc.unary_unary_rpc_method_handler(cloud,
            request_deserializer=pb.CloudInferenceTicket.FromString, response_serializer=pb.DetectionReply.SerializeToString),
        "GetFeedback": grpc.unary_unary_rpc_method_handler(feedback,
            request_deserializer=decode, response_serializer=encode),
        "SubmitWindowState": grpc.unary_unary_rpc_method_handler(window_state,
            request_deserializer=decode, response_serializer=encode),
    }),))


async def sample_forever(telemetry, *, kafka_worker=None, log=None):
    """log: optional text file; one JSON line per closed window (global Cloud state for CTDE)."""
    from contracts.control_v1 import FEEDBACK_TOPIC
    previous = logged = 0
    while True:
        telemetry.sample()
        feedback = telemetry.snapshot()
        if log is not None and feedback is not None and feedback["sequence"] != logged:
            log.write(encode({**feedback, "logged_utc_ns": time_ns()}).decode() + "\n")
            log.flush()
            logged = feedback["sequence"]
        if kafka_worker is not None and feedback is not None and feedback["sequence"] != previous:
            try:
                await asyncio.wait_for(kafka_worker._producer.send_and_wait(
                    FEEDBACK_TOPIC, key=telemetry.instance_id.encode(), value=encode(feedback)), timeout=3)
                previous = feedback["sequence"]
            except Exception as exc:
                print(f"[Control v1 feedback] {type(exc).__name__}: {exc}; retrying", flush=True)
        await asyncio.sleep(0.05)
