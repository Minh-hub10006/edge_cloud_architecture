"""Async Edge client for the versioned protobuf inference data plane."""

from __future__ import annotations

import grpc
from contracts.errors import CloudInferenceRejected

from contracts import CloudInferenceTicket, DetectionResult, MediaFrameRef
from contracts.grpc_codec import reference_to_proto, result_from_proto, result_to_proto, ticket_to_proto
from contracts.proto import edge_video_platform_pb2 as pb
from contracts.proto import edge_video_platform_pb2_grpc as pb_grpc


class GrpcCloudClient:
    def __init__(self, target: str, *, timeout_s: float = 5.0) -> None:
        if not target or timeout_s <= 0:
            raise ValueError("gRPC target and positive timeout_s are required")
        self.timeout_s = timeout_s
        self.channel = grpc.aio.insecure_channel(target, options=(
            ("grpc.max_send_message_length", 8 * 1024 * 1024),
            ("grpc.max_receive_message_length", 8 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 10_000),
            ("grpc.keepalive_timeout_ms", 3_000),
        ))
        self.stub = pb_grpc.EdgeCloudDataPlaneStub(self.channel)

    async def publish_media_reference(self, reference: MediaFrameRef) -> str:
        reply = await self.stub.PublishMediaFrameRef(
            reference_to_proto(reference), timeout=self.timeout_s,
        )
        if reply.status != "accepted":
            raise RuntimeError(f"Cloud rejected media reference: {reply.status} {reply.message}")
        return reply.status

    async def submit_detection(self, result: DetectionResult) -> str:
        reply = await self.stub.SubmitDetectionResult(
            result_to_proto(result), timeout=self.timeout_s,
        )
        if reply.status not in {"accepted", "duplicate", "out_of_order"}:
            raise RuntimeError(f"Cloud rejected detection result: {reply.status} {reply.message}")
        return reply.status

    async def submit_cloud_ticket(self, ticket: CloudInferenceTicket) -> DetectionResult:
        reply = await self.stub.SubmitCloudInference(
            ticket_to_proto(ticket), timeout=self.timeout_s,
        )
        if reply.status != "accepted" or not reply.HasField("result"):
            raise CloudInferenceRejected(reply.status, reply.message)
        return result_from_proto(reply.result)

    async def close(self) -> None:
        await self.channel.close(grace=1.0)


from contracts.reference_sink import MediaReferenceSink

# Backward-compatible export name for earlier Phase 7 scripts.
GrpcMediaReferenceSink = MediaReferenceSink
