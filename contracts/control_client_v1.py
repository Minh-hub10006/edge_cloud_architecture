"""Opt-in tracked-completion clients; no modification of legacy clients."""
from __future__ import annotations

import asyncio
import json
from time import monotonic
from uuid import uuid4

from contracts.control_v1 import (SERVICE, FEEDBACK_TOPIC, KAFKA_NAMESPACE,
                                  encode, decode, validate_feedback, validate_result)
from contracts.errors import CloudInferenceRejected
from contracts.grpc_codec import result_to_proto, result_from_proto, ticket_to_proto
from contracts.proto import edge_video_platform_pb2 as pb


class GrpcControlClient:
    def __init__(self, target, *, timeout_s=6):
        from contracts.grpc_client import GrpcCloudClient
        self.raw = GrpcCloudClient(target, timeout_s=timeout_s)
        self.timeout_s = timeout_s
        self.channel = self.raw.channel
        self.edge_rpc = self.channel.unary_unary(f"/{SERVICE}/SubmitEdgeDetection",
            request_serializer=pb.DetectionResult.SerializeToString,
            response_deserializer=pb.DetectionReply.FromString)
        self.cloud_rpc = self.channel.unary_unary(f"/{SERVICE}/SubmitCloudInference",
            request_serializer=pb.CloudInferenceTicket.SerializeToString,
            response_deserializer=pb.DetectionReply.FromString)
        self.feedback_rpc = self.channel.unary_unary(f"/{SERVICE}/GetFeedback",
            request_serializer=encode, response_deserializer=decode)
        self.window_rpc = self.channel.unary_unary(f"/{SERVICE}/SubmitWindowState",
            request_serializer=encode, response_deserializer=decode)

    async def start(self):
        await asyncio.wait_for(self.channel.channel_ready(), self.timeout_s)
        # Fail immediately on a wrong/old server, before starting video/model work.
        await self.get_feedback()

    async def publish_media_reference(self, reference):
        return await self.raw.publish_media_reference(reference)

    async def _reply(self, rpc, wire, request):
        reply = await rpc(wire, timeout=self.timeout_s)
        if reply.status != "accepted" or not reply.HasField("result"):
            raise CloudInferenceRejected(reply.status, reply.message)
        return validate_result(result_from_proto(reply.result), request)

    async def submit_detection(self, result):
        return await self._reply(self.edge_rpc, result_to_proto(result), result)

    async def submit_cloud_ticket(self, ticket):
        return await self._reply(self.cloud_rpc, ticket_to_proto(ticket), ticket)

    async def get_feedback(self, stream_key=None):
        """stream_key: also return this stream's scheduler command (docs/PLAN_CONTROL_V3_VI.md)."""
        request = {} if stream_key is None else {"stream_key": list(stream_key)}
        value = await self.feedback_rpc(request, timeout=min(3, self.timeout_s))
        return None if value is None else validate_feedback(value)

    async def submit_window_state(self, value):
        reply = await self.window_rpc(value, timeout=min(3, self.timeout_s))
        return isinstance(reply, dict) and reply.get("status") == "accepted"

    async def close(self):
        await self.raw.close()


class KafkaControlClient:
    def __init__(self, bootstrap, *, node_id, timeout_s=6):
        from contracts.kafka_transport import KafkaCloudClient
        self.raw = KafkaCloudClient(bootstrap, node_id=node_id, timeout_s=timeout_s,
                                   topic_namespace=KAFKA_NAMESPACE)
        self.bootstrap = bootstrap
        self.feedback = None
        self.throttles = {}   # P1: Cloud slow-lane worker -> (throttle, monotonic time); see slow_throttle
        self.consumer = self.listener = None
        self.error = None

    async def start(self):
        from contracts.kafka_transport import _kafka_types
        Consumer, _ = _kafka_types()
        try:
            await self.raw.start()
            from contracts.kafka_transport import SLOW_FEEDBACK_TOPIC
            self.consumer = Consumer(FEEDBACK_TOPIC, SLOW_FEEDBACK_TOPIC, bootstrap_servers=self.bootstrap,
                group_id=f"control-v1-feedback-{self.raw.node_id}-{uuid4().hex}",
                auto_offset_reset="latest", enable_auto_commit=True)
            await self.consumer.start()
            self.listener = asyncio.create_task(self._listen())
        except BaseException:
            await self.close()
            raise

    async def _listen(self):
        try:
            async for message in self.consumer:
                if message.topic != FEEDBACK_TOPIC:   # slow-lane throttle: {"source", "throttle": 1..8}
                    try:
                        m = json.loads(message.value)
                        self.throttles[str(m.get("source", ""))] = (min(8.0, max(1.0, float(m["throttle"]))),
                                                                    monotonic())
                    except (ValueError, KeyError, TypeError):
                        pass   # a malformed throttle message must not stop Cloud feedback
                    continue
                self.feedback = validate_feedback(decode(message.value))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = exc

    @property
    def slow_throttle(self) -> float:
        """The slowest any Cloud slow-lane worker asked for in the last 30 s (a worker that stopped expires)."""
        now = monotonic()
        return max([v for v, t in self.throttles.values() if now - t < 30] or [1.0])

    async def submit_window_state(self, value):
        return False  # the Cloud scheduler is gRPC-only

    async def get_feedback(self, stream_key=None):
        if self.error is not None:
            raise RuntimeError("Kafka feedback listener failed") from self.error
        return self.feedback

    async def publish_media_reference(self, reference):
        return await self.raw.publish_media_reference(reference)

    async def submit_detection(self, result):
        from contracts.kafka_transport import EDGE_RESULTS_TOPIC
        reply = await self.raw._request_result(result.request_id,
            self.raw.topics[EDGE_RESULTS_TOPIC], result_to_proto(result))
        return validate_result(reply, result)

    async def submit_cloud_ticket(self, ticket):
        return validate_result(await self.raw.submit_cloud_ticket(ticket), ticket)

    async def publish_tracked_result(self, result):
        await self.raw.publish_tracked_result(result)

    async def publish_slow(self, kind, payload, wait=False):
        await self.raw.publish_slow(kind, payload, wait)

    async def close(self):
        if self.listener is not None:
            self.listener.cancel()
            await asyncio.gather(self.listener, return_exceptions=True)
        if self.consumer is not None:
            await self.consumer.stop()
            self.consumer = None
        await self.raw.close()
