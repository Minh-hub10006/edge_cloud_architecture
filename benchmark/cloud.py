"""Opt-in Cloud measurement reply metadata and additive gRPC RPC.

The normal data plane and Kafka's existing Cloud-result reply remain intact.
No new protobuf schema or production RPC replacement is needed.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from time import monotonic_ns

from contracts.grpc_codec import result_from_proto, result_to_proto
from contracts.proto import edge_video_platform_pb2 as pb

RPC_SERVICE = "edge_video_platform.benchmark.MeasuredDataPlane"
RPC_EDGE = f"/{RPC_SERVICE}/SubmitEdgeDetection"
DECISION_PREFIX = "benchmark/"


def requested(value) -> bool:
    return value.decision_id.startswith(DECISION_PREFIX)


def finish_result(result, *, started_ns: int, stages: dict, tracking_completed_ns: int,
                  identity: dict | None = None):
    return replace(result, attributes={**result.attributes,
        "benchmark_schema": 1, "benchmark_completion": "bytetrack_tracked",
        "benchmark_tracking_completed_mono_ns": tracking_completed_ns,
        "benchmark_cloud_service_ms": (monotonic_ns() - started_ns) / 1e6,
        "benchmark_stages_ms": stages, "benchmark_model_identity": identity or {}})


def configure(service) -> None:
    """Only called when CLOUD_BENCHMARK=1; hashing is outside measured runs."""
    import os
    from .model import sha256, source_revision
    service.benchmark_enabled = True
    sample_path = os.getenv("CLOUD_BENCHMARK_RESOURCES", "")
    if sample_path and getattr(service, "benchmark_sampler", None) is None:
        # Cloud monotonic clock, same as benchmark_tracking_completed_mono_ns.
        from .resources import ResourceSampler
        service.benchmark_sampler = ResourceSampler(Path(sample_path), "cloud", 1.0)
        service.benchmark_sampler.start()
    for mode, adapter in service.cloud_adapters.items():
        detector = getattr(adapter, "detector", None)
        path = getattr(adapter, "artifact_path", None) or getattr(detector, "onnx_path", None)
        session = getattr(adapter, "session", None) or getattr(detector, "sess", None)
        service.benchmark_identities[mode] = {
            "model_id": adapter.model_id,
            "artifact_sha256": sha256(Path(path)) if path else None,
            "runtime": getattr(adapter, "runtime", None),
            "providers": session.get_providers() if session else None,
            "code": source_revision(), "max_cloud_inflight": service.max_cloud_inflight,
            "settings": getattr(adapter, "settings", None) or {key: os.environ.get(key, default) for key, default in
                         (("DFINE_TOP_K", "20"), ("DFINE_CONF_THR", "0.10"), ("DFINE_BOX_FORMAT", "cxcywh"))},
        }


def install_grpc(server, service) -> None:
    import grpc
    async def submit(request, context):
        try:
            result = result_from_proto(request)
            if not service.benchmark_enabled or not requested(result):
                return pb.DetectionReply(status="benchmark_disabled", message="CLOUD_BENCHMARK=1 required")
            status, message, tracked = await service.handle_detection_result(result)
            reply = pb.DetectionReply(status=status, message=message)
            if tracked is not None:
                reply.result.CopyFrom(result_to_proto(tracked))
            return reply
        except (TypeError, ValueError, KeyError) as exc:
            return pb.DetectionReply(status="invalid", message=str(exc))
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(RPC_SERVICE, {
        "SubmitEdgeDetection": grpc.unary_unary_rpc_method_handler(
            submit, request_deserializer=pb.DetectionResult.FromString,
            response_serializer=pb.DetectionReply.SerializeToString),
    }),))
