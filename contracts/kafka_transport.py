"""Kafka implementation of the same compact Edge/Cloud contract as gRPC.

Video is deliberately not sent through Kafka. The topics carry protobuf
identity sidecars, detections and cloud inference tickets; H.264 stays on RTSP.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from uuid import uuid4
from contracts.errors import CloudInferenceRejected
from typing import Any

from contracts import CloudInferenceTicket, DetectionResult, MediaFrameRef
from contracts.grpc_codec import (
    reference_from_proto, reference_to_proto, result_from_proto, result_to_proto,
    ticket_from_proto, ticket_to_proto,
)
from contracts.proto import edge_video_platform_pb2 as pb


MEDIA_REFS_TOPIC = "edge.media-frame-ref.v1"
EDGE_RESULTS_TOPIC = "edge.detection-result.v1"
CLOUD_TICKETS_TOPIC = "edge.cloud-inference-ticket.v1"
CLOUD_RESULTS_TOPIC = "cloud.detection-result.v1"
# v4 mode A: the Edge tracks its own detections and publishes them here (results writer -> database); no Cloud reply.
EDGE_TRACKED_TOPIC = "edge.tracked-result.v1"
# v4 slow lane (8/11): counts every few seconds (UC3) and events (UC2 ...; the start message is the alert) -> database
EDGE_COUNTS_TOPIC = "edge.count.v1"
EDGE_EVENTS_TOPIC = "edge.event.v1"
EDGE_PLATE_CROPS_TOPIC = "edge.plate-crop.v1"   # UC1 crops -> cloud_core/plate_worker (OCR)
SLOW_FEEDBACK_TOPIC = "cloud.slow-feedback.v1"
EDGE_ACTION_CLIPS_TOPIC = "edge.action-clip.v1"   # P3 UC4: fight clips -> cloud_core/uc4_disorder (RTMO-s + ST-GCN++)
EDGE_CRASH_FRAMES_TOPIC = "edge.crash-frame.v1"   # P3 UC4: fallen-bike crops -> cloud_core/uc4_disorder (RTMO-s)
EDGE_CROWD_FRAMES_TOPIC = "edge.crowd-frame.v1"   # P2 UC3: dense scenes -> cloud_core/uc3_crowd (DM-Count)   # P1: the Cloud slow-lane worker asks Edges to space their sends


def namespaced_topics(namespace: str) -> dict[str, str]:
    if namespace and (len(namespace) > 100 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in namespace)):
        raise ValueError("Kafka namespace must be <=100 ASCII alphanumeric/-/_ characters")
    prefix = namespace + "." if namespace else ""
    return {topic: prefix + topic for topic in
            (MEDIA_REFS_TOPIC, EDGE_RESULTS_TOPIC, CLOUD_TICKETS_TOPIC, CLOUD_RESULTS_TOPIC)}


def stream_key(value: Any) -> bytes:
    frame = value.frame
    return f"{frame.node_id}/{frame.camera_id}/{frame.stream_id}/{frame.stream_epoch}".encode()


def node_results_topic(results_topic: str, node_id: str) -> str:
    """Per-Edge copy of the results topic (8/11): every Edge Agent used to read every Edge's results and drop the others;
    with 246 and 252 together (8 cameras) 252's results arrived up to ~1 s late. The shared topic stays for the
    results writer (database); each Edge reads only its own copy."""
    if not node_id or len(node_id) > 100 or not all(c.isascii() and (c.isalnum() or c in "-_.") for c in node_id):
        raise ValueError("node_id must be <=100 ASCII alphanumeric/-/_/. characters to name its Kafka results topic")
    return f"{results_topic}.{node_id}"


def _kafka_types():
    try:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    except ImportError as exc:
        raise RuntimeError("Kafka transport requires the optional dependency: pip install '.[kafka]'") from exc
    return AIOKafkaConsumer, AIOKafkaProducer


class RenewingProducer:
    """An idempotent AIOKafkaProducer that is replaced when the broker no longer knows it.

    11/11 P5 (Kafka stopped 40 s on 249): after the broker came back every send of the old idempotent producers failed
    with OutOfOrderSequenceNumber (Edge 2002-2092 per stream, the Cloud feedback loop 7060 times) and B/C/D never
    recovered (23-25 % of frames in 4 min). aiokafka does not recover from these; a new producer does. send_and_wait
    renews and retries once; send() (no wait) marks the producer broken from its future, the next send renews it."""

    FATAL = ("OutOfOrderSequenceNumber", "UnknownProducerId", "InvalidProducerEpoch", "ProducerFenced")

    def __init__(self, factory) -> None:
        self._factory, self._producer, self._broken, self.renewed = factory, None, False, 0
        self._lock = asyncio.Lock()

    @classmethod
    def fatal(cls, exc) -> bool:
        return exc is not None and any(name in type(exc).__name__ for name in cls.FATAL)

    async def start(self) -> None:
        self._producer = self._factory()
        await self._producer.start()

    async def stop(self) -> None:
        if self._producer is not None:
            await self._producer.stop()

    async def _renew(self) -> None:
        async with self._lock:
            if not self._broken:
                return   # another caller already renewed it
            old, self._producer = self._producer, self._factory()
            await self._producer.start()
            self._broken = False
            self.renewed += 1
            print(f"[kafka] idempotent producer renewed ({self.renewed}) after the broker forgot it", flush=True)
            try:
                await asyncio.wait_for(old.stop(), 5)
            except Exception:
                pass   # the old one is unusable anyway

    def _watch(self, future) -> None:
        if not future.cancelled() and self.fatal(future.exception()):
            self._broken = True

    async def send_and_wait(self, *args, **kwargs):
        if self._broken:
            await self._renew()
        try:
            return await self._producer.send_and_wait(*args, **kwargs)
        except Exception as exc:
            if not self.fatal(exc):
                raise
            self._broken = True
            await self._renew()
            return await self._producer.send_and_wait(*args, **kwargs)

    async def send(self, *args, **kwargs):
        if self._broken:
            await self._renew()
        future = await self._producer.send(*args, **kwargs)
        future.add_done_callback(self._watch)
        return future


class KafkaCloudClient:
    """Edge client; every Edge has a distinct consumer group for result fanout."""

    def __init__(self, bootstrap_servers: str, *, node_id: str, timeout_s: float = 6.0,
                 max_pending: int = 64, topic_namespace: str = "") -> None:
        if not bootstrap_servers or not node_id or timeout_s <= 0 or max_pending <= 0:
            raise ValueError("bootstrap_servers, node_id, positive timeout_s/max_pending required")
        self.bootstrap_servers = bootstrap_servers
        self.node_id = node_id
        self.timeout_s = timeout_s
        self.max_pending = max_pending
        self.topics = namespaced_topics(topic_namespace)
        self.results_topic = node_results_topic(self.topics[CLOUD_RESULTS_TOPIC], node_id)
        self.consumer_group = f"edge-video-{node_id}-{uuid4().hex}"
        self.producer = None
        self.consumer = None
        self._listener: asyncio.Task | None = None
        self._slots = asyncio.Semaphore(max_pending)
        self._pending: dict[str, asyncio.Future[DetectionResult]] = {}
        self._background: set[asyncio.Task] = set()
        self.background_dropped = 0

    # 11/11 P5: aiokafka's send() waits for the broker's metadata. With Kafka stopped 54 s the Edge-only stream (mode N,
    # nothing to wait for from the Cloud) lost 781 frames because each frame awaited the database copy of its result;
    # "no network -> the Edge keeps running" (v3/v4). Messages nobody waits for now go out in the background, at most
    # BACKGROUND_MAX at a time, each given BACKGROUND_TIMEOUT_S; beyond that they are dropped and counted.
    BACKGROUND_MAX, BACKGROUND_TIMEOUT_S = 64, 10.0

    def _send_in_background(self, topic: str, key: bytes, value: bytes) -> None:
        if len(self._background) >= self.BACKGROUND_MAX:
            self.background_dropped += 1
            return

        async def send():
            try:
                await asyncio.wait_for(self.producer.send(topic, key=key, value=value), self.BACKGROUND_TIMEOUT_S)
            except Exception:
                self.background_dropped += 1
        task = asyncio.create_task(send())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def start(self) -> None:
        if self.producer is not None:
            return
        AIOKafkaConsumer, AIOKafkaProducer = _kafka_types()
        self.producer = RenewingProducer(lambda: AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers, acks="all",
            enable_idempotence=True,
        ))
        self.consumer = AIOKafkaConsumer(
            self.results_topic, bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            enable_auto_commit=True, auto_offset_reset="latest",
            value_deserializer=lambda value: value,
        )
        await self.producer.start()
        try:
            await self.consumer.start()
        except BaseException:
            await self.producer.stop()
            self.producer = None
            raise
        self._listener = asyncio.create_task(self._consume_results())

    async def _send_proto(self, topic: str, message: Any) -> None:
        if self.producer is None:
            raise RuntimeError("Kafka client has not been started")
        await asyncio.wait_for(self.producer.send_and_wait(
            topic, key=stream_key(message), value=message.SerializeToString()), self.timeout_s)

    async def publish_media_reference(self, reference: MediaFrameRef) -> str:
        await self._send_proto(self.topics[MEDIA_REFS_TOPIC], reference_to_proto(reference))
        return "accepted"

    async def publish_tracked_result(self, result: DetectionResult) -> None:
        """v4 mode A: queue an Edge-tracked result for the database; nothing waits for a reply."""
        if self.producer is None:
            raise RuntimeError("Kafka client has not been started")
        value = json.dumps({"schema_version": 1, "request_id": result.request_id, "status": "accepted", "message": "",
                            "result": result_to_proto(result).SerializeToString().hex()}, separators=(",", ":")).encode()
        self._send_in_background(EDGE_TRACKED_TOPIC, stream_key(result), value)

    async def publish_slow(self, kind: str, payload: dict, wait: bool = False) -> None:
        """Slow lane output: kind "count" or "event"; keyed by camera so one camera's messages stay in order.
        wait: return only once Kafka has the message (events: the Edge outbox acks after that)."""
        if self.producer is None:
            raise RuntimeError("Kafka client has not been started")
        topic = {"count": EDGE_COUNTS_TOPIC, "event": EDGE_EVENTS_TOPIC, "plate": EDGE_PLATE_CROPS_TOPIC,
                 "crowd": EDGE_CROWD_FRAMES_TOPIC, "action": EDGE_ACTION_CLIPS_TOPIC,
                 "crash": EDGE_CRASH_FRAMES_TOPIC}[kind]
        key = f"{payload['node_id']}/{payload['camera_id']}/{payload['stream_id']}/{payload['stream_epoch']}".encode()
        value = json.dumps({"schema_version": 1, "kind": kind, **payload}, separators=(",", ":")).encode()
        if wait:
            await asyncio.wait_for(self.producer.send_and_wait(topic, key=key, value=value), self.timeout_s)
        else:
            self._send_in_background(topic, key, value)

    async def submit_detection(self, result: DetectionResult) -> str:
        if not result.request_id:
            raise ValueError("Kafka detection results require request_id")
        await self._request_result(
            result.request_id, self.topics[EDGE_RESULTS_TOPIC], result_to_proto(result),
        )
        return "accepted"

    async def submit_cloud_ticket(self, ticket: CloudInferenceTicket) -> DetectionResult:
        request_id = ticket.request_id
        if not request_id:
            raise ValueError("Kafka cloud tickets require request_id")
        return await self._request_result(
            request_id, self.topics[CLOUD_TICKETS_TOPIC], ticket_to_proto(ticket),
        )

    async def _request_result(self, request_id: str, topic: str, message: Any) -> DetectionResult:
        if self.producer is None:
            raise RuntimeError("Kafka client has not been started")
        async with self._slots:
            future = asyncio.get_running_loop().create_future()
            if request_id in self._pending:
                raise ValueError(f"duplicate pending request_id: {request_id}")
            self._pending[request_id] = future
            try:
                async def send_and_receive():
                    await self._send_proto(topic, message)
                    return await asyncio.shield(future)
                return await asyncio.wait_for(send_and_receive(), timeout=self.timeout_s)
            finally:
                self._pending.pop(request_id, None)
                if not future.done():
                    future.cancel()

    async def _consume_results(self) -> None:
        assert self.consumer is not None
        async for message in self.consumer:
            try:
                envelope = json.loads(message.value)
                request_id = envelope.get("request_id")
                future = self._pending.get(request_id)
                if future is None or future.done():
                    continue
                if envelope.get("status") != "accepted" or not envelope.get("result"):
                    future.set_exception(CloudInferenceRejected(
                        str(envelope.get("status", "invalid_reply")), envelope.get("message", ""),
                    ))
                    continue
                result = pb.DetectionResult.FromString(bytes.fromhex(envelope["result"]))
                domain_result = result_from_proto(result)
                if domain_result.request_id != request_id:
                    raise ValueError("Kafka result request_id does not match envelope")
                future.set_result(domain_result)
            except Exception as exc:
                # A malformed unrelated event must not terminate result delivery.
                request_id = None
                try:
                    request_id = json.loads(message.value).get("request_id")
                except Exception:
                    pass
                future = self._pending.get(request_id)
                if future is not None and not future.done():
                    future.set_exception(exc)

    async def close(self) -> None:
        if self._background:   # let queued database copies go out, briefly
            await asyncio.wait(set(self._background), timeout=5)
            for task in set(self._background):
                task.cancel()
        if self._listener is not None:
            self._listener.cancel()
            await asyncio.gather(self._listener, return_exceptions=True)
            self._listener = None
        if self.consumer is not None:
            await self.consumer.stop()
            self.consumer = None
        if self.producer is not None:
            await self.producer.stop()
            self.producer = None


class KafkaCloudWorker:
    """Cloud-side topic worker that delegates business logic to the shared service."""

    def __init__(self, service: Any, bootstrap_servers: str, *, topic_namespace: str = "") -> None:
        if not bootstrap_servers:
            raise ValueError("Kafka bootstrap_servers is required")
        self.service = service
        self.bootstrap_servers = bootstrap_servers
        self.topics = namespaced_topics(topic_namespace)
        self._producer = None
        self._consumers = []
        self._tasks: list[asyncio.Task] = []
        self._lanes: dict[bytes, asyncio.Task] = {}

    async def start(self) -> None:
        AIOKafkaConsumer, AIOKafkaProducer = _kafka_types()
        self._producer = RenewingProducer(lambda: AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers, acks="all",
            enable_idempotence=True,
        ))
        await self._producer.start()
        try:
            for topic in (MEDIA_REFS_TOPIC, EDGE_RESULTS_TOPIC, CLOUD_TICKETS_TOPIC):
                def make(topic=topic):
                    return AIOKafkaConsumer(
                        self.topics[topic], bootstrap_servers=self.bootstrap_servers,
                        group_id=f"edge-video-cloud-{self.topics[topic]}", enable_auto_commit=False,
                        auto_offset_reset="latest", value_deserializer=lambda value: value,
                    )
                consumer = make()
                await consumer.start()
                self._consumers.append(consumer)
                self._tasks.append(asyncio.create_task(self._supervise(topic, make, consumer)))
        except BaseException:
            await self.close()
            raise

    async def _supervise(self, topic: str, make, consumer, backoff_s: float = 1.0) -> None:
        """Keep one topic consumed. _consume stops at the first failed message so nothing is committed past unfinished
        work; here a new consumer then resumes from the last committed offset. 11/11 P5: with Kafka stopped 55 s, the
        reply publishes timed out, the ticket/reference loops ended for good and modes C/D got nothing back for the rest
        of the run (186-246 of 3570 frames) while the worker looked alive."""
        while True:
            try:
                await self._consume(topic, consumer)
                return   # the consumer was stopped (close)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[Cloud Kafka] {topic}: consumer restarts after {type(exc).__name__}: {exc}", flush=True)
            if consumer in self._consumers:
                self._consumers.remove(consumer)
            try:
                await asyncio.wait_for(consumer.stop(), 10)
            except Exception:
                pass
            while True:
                await asyncio.sleep(backoff_s)
                consumer = make()
                try:
                    await consumer.start()
                    break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:   # broker still down: try again
                    print(f"[Cloud Kafka] {topic}: consumer start failed: {type(exc).__name__}: {exc}", flush=True)
            self._consumers.append(consumer)

    async def _publish_result(self, request_id: str | None, status: str,
                              result: DetectionResult | None, message: str = "", source: Any = None) -> None:
        """source = the ticket / Edge result being answered: its node gets its own copy (rejections included)."""
        assert self._producer is not None
        encoded = result_to_proto(result).SerializeToString().hex() if result is not None else None
        origin = result if result is not None else source
        key = stream_key(origin) if origin is not None else (request_id or "unknown").encode()
        value = json.dumps({
            "schema_version": 1, "request_id": request_id, "status": status,
            "message": message, "result": encoded,
        }, separators=(",", ":")).encode()
        shared = self.topics[CLOUD_RESULTS_TOPIC]
        topics = [shared] + ([node_results_topic(shared, origin.frame.node_id)] if origin is not None else [])
        await asyncio.wait_for(asyncio.gather(*(self._producer.send_and_wait(t, key=key, value=value) for t in topics)),
                               timeout=6.0)

    async def _handle(self, topic, value):
        if topic == MEDIA_REFS_TOPIC:
            await self.service.handle_media_reference(value)
        elif topic == EDGE_RESULTS_TOPIC:
            status, detail, result = await self.service.handle_detection_result(value)
            await self._publish_result(value.request_id, status, result, detail, source=value)
        else:
            status, result, detail = await self.service.handle_cloud_ticket(value)
            await self._publish_result(value.request_id, status, result, detail, source=value)

    async def _consume(self, topic: str, consumer: Any) -> None:
        from aiokafka import TopicPartition
        # Ordered per stream, concurrent across streams, bounded in memory.
        slots = asyncio.Semaphore(64)
        running = set()
        offsets = {}
        commit_lock = asyncio.Lock()
        failed = asyncio.get_running_loop().create_future()
        decoder = {
            MEDIA_REFS_TOPIC: lambda b: reference_from_proto(pb.MediaFrameRef.FromString(b)),
            EDGE_RESULTS_TOPIC: lambda b: result_from_proto(pb.DetectionResult.FromString(b)),
            CLOUD_TICKETS_TOPIC: lambda b: ticket_from_proto(pb.CloudInferenceTicket.FromString(b)),
        }[topic]

        async def process(message, value, previous, lane):
            partition = TopicPartition(message.topic, message.partition)
            try:
                if previous is not None:
                    await previous
                if value is not None:
                    await self._handle(topic, value)
                async with commit_lock:
                    entries = offsets[partition]
                    for entry in entries:
                        if entry[0] == message.offset:
                            entry[1] = True
                            break
                    checkpoint = None
                    released = 0
                    while entries and entries[0][1]:
                        checkpoint = entries.popleft()[0] + 1
                        released += 1
                    if checkpoint is not None:
                        await consumer.commit({partition: checkpoint})
                        for _ in range(released):
                            slots.release()
            except Exception as exc:
                # Never commit past unfinished work or a failed reply publish.
                if not failed.done():
                    failed.set_result(exc)
                print(f"[Cloud Kafka] {topic}: processing stopped: {type(exc).__name__}: {exc}", flush=True)
                raise
            finally:
                if self._lanes.get(lane) is asyncio.current_task():
                    self._lanes.pop(lane, None)

        next_message = None
        try:
            iterator = consumer.__aiter__()
            while True:
                next_message = asyncio.create_task(iterator.__anext__())
                done, _ = await asyncio.wait((next_message, failed), return_when=asyncio.FIRST_COMPLETED)
                if failed in done:
                    raise failed.result()
                try:
                    message = next_message.result()
                except StopAsyncIteration:
                    break
                admission = asyncio.create_task(slots.acquire())
                try:
                    ready, _ = await asyncio.wait((admission, failed), return_when=asyncio.FIRST_COMPLETED)
                    if failed in ready:
                        raise failed.result()
                finally:
                    if not admission.done():
                        admission.cancel()
                    await asyncio.gather(admission, return_exceptions=True)
                try:
                    value = decoder(message.value)
                    # References have an independent lane: tickets must never
                    # prevent the sidecar that resolves their cache lookup.
                    lane = (b"refs:" if topic == MEDIA_REFS_TOPIC else b"results:") + stream_key(value)
                    if topic == CLOUD_TICKETS_TOPIC and getattr(self.service, "benchmark_enabled", False):
                        from benchmark.ordering import ordinal
                        if value.decision_id.startswith("benchmark/") and ordinal(value) is not None:
                            # The benchmark service reorders before ByteTrack;
                            # do not wait for the previous frame's inference.
                            lane = b"benchmark-ticket:" + stream_key(value) + value.request_id.encode()
                except Exception as exc:
                    print(f"[Cloud Kafka] invalid {topic} payload: {exc}", flush=True)
                    value, lane = None, b"invalid"
                partition = TopicPartition(message.topic, message.partition)
                offsets.setdefault(partition, deque()).append([message.offset, False])
                task = asyncio.create_task(process(message, value, self._lanes.get(lane), lane))
                self._lanes[lane] = task
                running.add(task)
                def complete(task):
                    running.discard(task)
                    if not task.cancelled():
                        task.exception()  # failure is relayed through failed
                task.add_done_callback(complete)
            if running:
                await asyncio.gather(*running)
            if failed.done():
                raise failed.result()
        finally:
            if next_message is not None and not next_message.done():
                next_message.cancel()
                await asyncio.gather(next_message, return_exceptions=True)
            for task in running:
                task.cancel()
            await asyncio.gather(*running, return_exceptions=True)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        for consumer in self._consumers:
            await consumer.stop()
        self._consumers.clear()
        if self._producer is not None:
            await self._producer.stop()
            self._producer = None
