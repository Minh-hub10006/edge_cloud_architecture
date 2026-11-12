"""Async Kafka request/result transport used only by the standardized benchmark."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any, Callable
from uuid import uuid4

from benchmark.runners.dfine_runner import EncodedFrame, TransportResult
from runtime.kafka_dfine import (
    REQUEST_TOPIC,
    RESULT_TOPIC,
    KafkaDfineRequestV1,
    decode_kafka_dfine_result_v1,
)


@dataclass(frozen=True)
class _Delivery:
    result: Any
    envelope_bytes: int
    decode_ms: float


class KafkaDfineBenchmarkTransport:
    """Bounded async Kafka D-FINE transport with per-frame benchmark evidence.

    It deliberately measures only edge-local clock intervals.  Kafka broker,
    cloud queue, and cloud inference are not derived from unsynchronised clocks.
    """

    name = "kafka"

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        edge_id: str,
        stream_id: str,
        run_id: str,
        timeout_s: float,
        request_topic: str = REQUEST_TOPIC,
        result_topic: str = RESULT_TOPIC,
        consumer_group: str,
        consumer_factory: Callable[..., Any] | None = None,
        producer_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not all(str(value).strip() for value in (bootstrap_servers, edge_id, stream_id, run_id, consumer_group)):
            raise ValueError("Kafka benchmark transport identifiers are required")
        if timeout_s <= 0:
            raise ValueError("Kafka benchmark timeout must be positive")
        self.bootstrap_servers = str(bootstrap_servers).strip()
        self.edge_id = str(edge_id).strip()
        self.stream_id = str(stream_id).strip()
        self.run_id = str(run_id).strip()
        self.timeout_s = float(timeout_s)
        self.request_topic = str(request_topic).strip() or REQUEST_TOPIC
        self.result_topic = str(result_topic).strip() or RESULT_TOPIC
        self.consumer_group = str(consumer_group).strip()
        self._consumer_factory = consumer_factory
        self._producer_factory = producer_factory
        self._consumer = self._producer = None
        self._result_task: asyncio.Task | None = None
        self._pending: dict[str, asyncio.Future] = {}
        self._pending_frames: dict[str, str] = {}
        self._closed = False
        self._submitted = self._published = self._completed = 0
        self._timeout = self._dropped = self._invalid = self._unknown_results = 0
        self._ignored_results = 0

    async def prepare(self) -> None:
        if self._result_task is not None:
            return
        consumer_factory, producer_factory = self._consumer_factory, self._producer_factory
        if consumer_factory is None or producer_factory is None:
            try:
                from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
            except ImportError as exc:
                raise RuntimeError("aiokafka is required for --transport kafka") from exc
            consumer_factory = consumer_factory or AIOKafkaConsumer
            producer_factory = producer_factory or AIOKafkaProducer
        self._consumer = consumer_factory(
            self.result_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            enable_auto_commit=True,
            auto_offset_reset="latest",
        )
        self._producer = producer_factory(
            bootstrap_servers=self.bootstrap_servers,
            acks="all",
        )
        try:
            await self._consumer.start()
            await self._producer.start()
        except BaseException:
            await self.aclose()
            raise
        self._closed = False
        self._result_task = asyncio.create_task(
            self._consume_results(), name="benchmark-kafka-dfine-results"
        )

    async def send_and_wait(
        self,
        encoded: EncodedFrame,
        frame_id: int | str,
        trace: Any,
        *,
        timeout_s: float = 12.0,
    ) -> TransportResult:
        if self._producer is None or self._result_task is None or self._closed:
            raise RuntimeError("Kafka benchmark transport is not prepared")
        if encoded.layout != "mono_jpeg" or not isinstance(encoded.payload, (bytes, bytearray, memoryview)):
            raise ValueError("Kafka benchmark transport requires mono_jpeg bytes")

        timeout = float(timeout_s)
        if timeout <= 0:
            raise ValueError("transport timeout must be positive")
        started_mono = time.perf_counter()
        deadline_mono = started_mono + timeout
        request_id = uuid4().hex
        jpeg = bytes(encoded.payload)
        now = time.time()
        request = KafkaDfineRequestV1(
            run_id=self.run_id,
            request_id=request_id,
            edge_id=self.edge_id,
            stream_id=self.stream_id,
            frame_id=str(frame_id),
            jpeg_bytes=jpeg,
            created_at_unix_s=now,
            deadline_unix_s=now + timeout,
        )
        trace.attributes.update(
            {
                "transport": self.name,
                "transport_timeout_s": timeout,
                "transport_request_id": request_id,
                "transport_pending_before": len(self._pending),
                "payload_bytes": len(jpeg),
                "uplink_layout": "mono_jpeg",
            }
        )
        future: asyncio.Future | None = None
        try:
            with trace.span("transport_pack"):
                envelope = request.to_json_bytes()
            trace.attributes["transport_request_envelope_bytes"] = len(envelope)
            trace.start("transport_enqueue")
            future = asyncio.get_running_loop().create_future()
            self._pending[request_id] = future
            self._pending_frames[request_id] = request.frame_id
            trace.stop("transport_enqueue")
            self._submitted += 1
            trace.start("transport_rtt")

            trace.start("transport_publish")
            try:
                await asyncio.wait_for(
                    self._producer.send_and_wait(
                        self.request_topic,
                        key=request.kafka_key.encode("utf-8"),
                        value=envelope,
                    ),
                    timeout=self._remaining(deadline_mono),
                )
            finally:
                if trace.is_running("transport_publish"):
                    trace.stop("transport_publish")
            self._published += 1
            trace.attributes["transport_broker_ack"] = True

            delivery = await asyncio.wait_for(
                future, timeout=self._remaining(deadline_mono)
            )
        except asyncio.TimeoutError:
            self._timeout += 1
            trace.attributes.setdefault("transport_broker_ack", False)
            return TransportResult(
                feedback=None,
                timed_out=True,
                status="timeout",
                error=f"Kafka D-FINE timeout after {timeout:g}s",
            )
        except Exception as exc:
            self._dropped += 1
            trace.attributes.setdefault("transport_broker_ack", False)
            return TransportResult(
                feedback=None,
                timed_out=False,
                status="dropped",
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            self._pending.pop(request_id, None)
            self._pending_frames.pop(request_id, None)
            if trace.is_running("transport_rtt"):
                trace.stop("transport_rtt")
            trace.attributes["transport_pending_after"] = len(self._pending)

        trace.observe_ms("transport_result_decode", delivery.decode_ms)
        trace.attributes.update(
            {
                "transport_result_envelope_bytes": delivery.envelope_bytes,
                "transport_result_status": delivery.result.status,
                "cloud_matched_id": delivery.result.matched_id,
                "cloud_usage": float(delivery.result.cloud_usage),
                "num_detections": len(delivery.result.boxes),
            }
        )
        if delivery.result.decode_ms is not None:
            trace.observe_ms("cloud_decode", float(delivery.result.decode_ms))
        if delivery.result.status != "completed":
            is_timeout = delivery.result.status in {"timeout", "expired"}
            if is_timeout:
                self._timeout += 1
            else:
                self._dropped += 1
            return TransportResult(
                feedback=None,
                timed_out=is_timeout,
                status="timeout" if is_timeout else "dropped",
                error=delivery.result.error_code or delivery.result.status,
            )

        self._completed += 1
        return TransportResult(
            feedback={
                "frame_id": str(delivery.result.frame_id),
                "matched_id": delivery.result.matched_id,
                "boxes": [dict(box) for box in delivery.result.boxes],
                "num_boxes": len(delivery.result.boxes),
                "cloud_usage": float(delivery.result.cloud_usage),
                "cloud_decode_ms": float(delivery.result.decode_ms),
            },
            timed_out=False,
        )

    async def _consume_results(self) -> None:
        try:
            async for record in self._consumer:
                started = time.perf_counter_ns()
                try:
                    result = decode_kafka_dfine_result_v1(record.value)
                except Exception:
                    self._invalid += 1
                    continue
                decode_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                if (
                    result.run_id != self.run_id
                    or result.edge_id != self.edge_id
                    or result.stream_id != self.stream_id
                ):
                    self._ignored_results += 1
                    continue
                future = self._pending.get(result.request_id)
                if future is None:
                    self._unknown_results += 1
                    continue
                if result.frame_id != self._pending_frames.get(result.request_id):
                    self._invalid += 1
                    continue
                if not future.done():
                    raw = record.value
                    future.set_result(
                        _Delivery(
                            result=result,
                            envelope_bytes=len(raw) if isinstance(raw, (bytes, bytearray, memoryview)) else 0,
                            decode_ms=decode_ms,
                        )
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Individual frame tasks turn this into bounded timeout/error status.
            self._dropped += 1

    async def aclose(self) -> None:
        self._closed = True
        task, self._result_task = self._result_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for future in tuple(self._pending.values()):
            if not future.done():
                future.cancel()
        self._pending.clear()
        self._pending_frames.clear()
        for name in ("_consumer", "_producer"):
            client = getattr(self, name)
            setattr(self, name, None)
            if client is not None:
                await client.stop()

    def reset_benchmark_metrics(self) -> None:
        self._submitted = self._published = self._completed = 0
        self._timeout = self._dropped = self._invalid = 0
        self._unknown_results = self._ignored_results = 0

    def benchmark_metrics(self) -> dict[str, int | str]:
        return {
            "mode": self.name,
            "submitted": self._submitted,
            "broker_acked": self._published,
            "completed": self._completed,
            "timeout": self._timeout,
            "dropped": self._dropped,
            "invalid_results": self._invalid,
            "unknown_results": self._unknown_results,
            "ignored_results": self._ignored_results,
            "pending": len(self._pending),
        }

    @staticmethod
    def _remaining(deadline_mono: float) -> float:
        remaining = deadline_mono - time.perf_counter()
        if remaining <= 0:
            raise asyncio.TimeoutError()
        return remaining
