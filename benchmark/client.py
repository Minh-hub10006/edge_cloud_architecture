"""Return actual hub-completed results over either existing transport."""
from __future__ import annotations

import asyncio

from contracts.errors import CloudInferenceRejected
from contracts.grpc_codec import result_from_proto, result_to_proto
from contracts.proto import edge_video_platform_pb2 as pb
from .cloud import RPC_EDGE


class MeasuredClient:
    def __init__(self, raw, transport: str) -> None:
        self.raw = raw
        self.transport = transport

    async def publish_media_reference(self, reference):
        return await self.raw.publish_media_reference(reference)

    @staticmethod
    def check(request, reply):
        if (reply.request_id != request.request_id or reply.frame != request.frame
                or reply.decision_id != request.decision_id or reply.model_id != request.model_id
                or reply.model_mode != request.model_mode or reply.preview_pts != request.preview_pts):
            raise ValueError("benchmark reply identity/frame/PTS/model mismatch")
        if (reply.attributes.get("benchmark_schema") != 1
                or reply.attributes.get("benchmark_completion") != "bytetrack_tracked"):
            raise ValueError("Cloud benchmark completion unavailable; enable CLOUD_BENCHMARK=1")
        completed_ns = reply.attributes.get("benchmark_tracking_completed_mono_ns")
        if type(completed_ns) is not int or completed_ns <= 0:
            raise ValueError("Cloud ByteTrack completion timestamp unavailable; update Cloud benchmark code")
        return reply

    async def submit_detection(self, result):
        if self.transport == "grpc":
            method = self.raw.channel.unary_unary(RPC_EDGE,
                request_serializer=pb.DetectionResult.SerializeToString,
                response_deserializer=pb.DetectionReply.FromString)
            reply = await method(result_to_proto(result), timeout=self.raw.timeout_s)
            if reply.status != "accepted" or not reply.HasField("result"):
                raise CloudInferenceRejected(reply.status, reply.message)
            domain = result_from_proto(reply.result)
        else:
            from contracts.kafka_transport import EDGE_RESULTS_TOPIC
            # Existing Kafka client already waits for Cloud's result envelope.
            domain = await self.raw._request_result(result.request_id, self.raw.topics[EDGE_RESULTS_TOPIC],
                                                    result_to_proto(result))
        return self.check(result, domain)

    async def submit_cloud_ticket(self, ticket):
        return self.check(ticket, await self.raw.submit_cloud_ticket(ticket))

    async def close(self):
        await self.raw.close()


async def connect(transport: str, target: str, node_id: str, timeout_s: float,
                  topic_namespace: str = "benchmark"):
    if transport == "grpc":
        from contracts.grpc_client import GrpcCloudClient
        raw = GrpcCloudClient(target, timeout_s=timeout_s)
        try:
            await asyncio.wait_for(raw.channel.channel_ready(), timeout=timeout_s)
        except BaseException:
            await raw.close()
            raise
    else:
        from contracts.kafka_transport import KafkaCloudClient
        raw = KafkaCloudClient(target, node_id=node_id, timeout_s=timeout_s,
                               topic_namespace=topic_namespace)
        await raw.start()
    return MeasuredClient(raw, transport)
