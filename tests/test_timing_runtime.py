from __future__ import annotations
import asyncio
from collections import namedtuple
from contextlib import redirect_stdout
import io
import sys
import time
from time import time_ns
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from contracts import CloudInferenceTicket, ModelMode, FrameIdentity, MediaFrameRef
from contracts.grpc_codec import ticket_to_proto
from contracts.kafka_transport import KafkaCloudWorker, KafkaCloudClient, CLOUD_TICKETS_TOPIC
from edge_agent import BoundedLatestQueue, CapturedFrame
from inference.edge_worker import run_edge_inference_worker
from inference.frame_refs import MediaReferenceMap
from cloud_core import CloudFrameCache, DecodedMediaFrame, ResultHub
from cloud_core.grpc_service import CloudDataPlaneServicer

TopicPartition = namedtuple("TopicPartition", "topic partition")


class Consumer:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.commits = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.queue.get()
        if value is None:
            raise StopAsyncIteration
        return value

    async def commit(self, value):
        self.commits.append(value)


def message(node, offset):
    frame = FrameIdentity(node, "cam", "preview", 1, offset, time_ns(), offset)
    ticket = CloudInferenceTicket(frame, ModelMode.DFINE_CLOUD, "dfine", "fixed", offset,
                                  time_ns()+10**9, f"{node}-{offset}")
    return SimpleNamespace(topic=CLOUD_TICKETS_TOPIC, partition=0, offset=offset,
                           value=ticket_to_proto(ticket).SerializeToString())


class TimingRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_reference_shutdown_cancels_hung_sender(self):
        from contracts.reference_sink import MediaReferenceSink
        async def hang(reference):
            await asyncio.Event().wait()
        sink = MediaReferenceSink(SimpleNamespace(publish_media_reference=hang))
        sink.start()
        frame = FrameIdentity("a", "b", "c", 1, 1, time_ns(), 1)
        await sink.submit(MediaFrameRef(frame, 1, 1, time_ns()))
        await asyncio.wait_for(sink.close(timeout_s=.02), .5)
        self.assertIsNone(sink._task)
        await asyncio.wait_for(sink.queue.join(), .5)

    async def test_kafka_uncommitted_backlog_is_bounded(self):
        release = asyncio.Event()
        completed = []
        async def handle(ticket):
            if ticket.frame.source_frame_id == 0:
                await release.wait()
            completed.append(ticket.request_id)
            return "accepted", None, ""
        worker = KafkaCloudWorker(SimpleNamespace(handle_cloud_ticket=handle), "unused")
        async def publish(*args, **kwargs):
            pass
        worker._publish_result = publish
        consumer = Consumer()
        for offset in range(150):
            consumer.queue.put_nowait(message(str(offset), offset))
        consumer.queue.put_nowait(None)
        with patch.dict(sys.modules, {"aiokafka": SimpleNamespace(TopicPartition=TopicPartition)}):
            task = asyncio.create_task(worker._consume(CLOUD_TICKETS_TOPIC, consumer))
            try:
                await asyncio.sleep(.15)
                self.assertLessEqual(len(completed), 63)
                self.assertEqual(consumer.commits, [])
                release.set()
                await asyncio.wait_for(task, 3)
                self.assertEqual(len(completed), 150)
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_deadline_expiring_during_inference_does_not_publish(self):
        class Detector:
            def infer(self, pixels):
                time.sleep(.04)
                return [], 40
        class Tracker:
            def update(self, result):
                return result
            def reset(self, key):
                pass
        cache, hub = CloudFrameCache(), ResultHub(tracker=Tracker())
        _, queue = hub.subscribe()
        now = time_ns()
        frame = FrameIdentity("a", "b", "c", 1, 1, now, 1)
        await cache.announce(MediaFrameRef(frame, 1, 1, now))
        await cache.ingest(DecodedMediaFrame(frame.stream_key, 1, 1, "pixels", now))
        service = CloudDataPlaneServicer(hub=hub, frame_cache=cache, frame_taps=None, dfine_adapter=Detector())
        ticket = CloudInferenceTicket(frame, ModelMode.DFINE_CLOUD, "dfine", "fixed", 1, time_ns()+20_000_000, "expires")
        status, result, _ = await service.handle_cloud_ticket(ticket)
        self.assertEqual(status, "deadline_exceeded")
        self.assertIsNone(result)
        self.assertTrue(queue.empty())

    async def test_kafka_parallel_streams_ordered_lane_contiguous_commits(self):
        started, other_done, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        completed = []

        class Service:
            async def handle_cloud_ticket(self, ticket):
                if ticket.request_id == "a-0":
                    started.set()
                    await release.wait()
                completed.append(ticket.request_id)
                if ticket.frame.node_id == "b":
                    other_done.set()
                return "accepted", None, ""

        worker = KafkaCloudWorker(Service(), "unused")
        async def publish(*args, **kwargs):
            pass
        worker._publish_result = publish
        consumer = Consumer()
        for node, offset in [("a", 0), ("a", 1), ("b", 2)]:
            consumer.queue.put_nowait(message(node, offset))
        with patch.dict(sys.modules, {"aiokafka": SimpleNamespace(TopicPartition=TopicPartition)}):
            task = asyncio.create_task(worker._consume(CLOUD_TICKETS_TOPIC, consumer))
            try:
                await asyncio.wait_for(other_done.wait(), 1)
                self.assertTrue(started.is_set())
                self.assertEqual(completed, ["b-2"])
                self.assertEqual(consumer.commits, [])
                release.set()
                consumer.queue.put_nowait(None)
                await asyncio.wait_for(task, 1)
                self.assertEqual(completed, ["b-2", "a-0", "a-1"])
                self.assertEqual(consumer.commits[-1], {TopicPartition(CLOUD_TICKETS_TOPIC, 0): 3})
                self.assertEqual(worker._lanes, {})
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_failed_kafka_reply_is_not_committed(self):
        async def handle(ticket):
            return "accepted", None, ""
        worker = KafkaCloudWorker(SimpleNamespace(handle_cloud_ticket=handle), "unused")
        async def publish(*args, **kwargs):
            raise ConnectionError("broker disconnected")
        worker._publish_result = publish
        consumer = Consumer()
        consumer.queue.put_nowait(message("a", 0))
        with patch.dict(sys.modules, {"aiokafka": SimpleNamespace(TopicPartition=TopicPartition)}), redirect_stdout(io.StringIO()):
            with self.assertRaises(ConnectionError):
                await asyncio.wait_for(worker._consume(CLOUD_TICKETS_TOPIC, consumer), 1)
        self.assertEqual(consumer.commits, [])

    async def test_two_processes_on_same_node_get_distinct_reply_groups(self):
        a, b = KafkaCloudClient("unused", node_id="edge-a"), KafkaCloudClient("unused", node_id="edge-a")
        self.assertNotEqual(a.consumer_group, b.consumer_group)

    async def test_latest_only_drops_backlog_and_capture_age_is_bounded(self):
        queue, refs = BoundedLatestQueue(8), MediaReferenceMap()
        for seq in range(8):
            frame = FrameIdentity("a", "b", "c", 1, seq, time_ns(), seq)
            queue.put_latest(CapturedFrame(frame, None, seq))
            await refs.publish(MediaFrameRef(frame, seq, seq, time_ns()))
        submitted = []
        async def submit(ticket):
            submitted.append(ticket)
            return SimpleNamespace(boxes=[])
        stop = asyncio.Event(); stop.set()
        stats = await run_edge_inference_worker(queue, mode=ModelMode.DFINE_CLOUD,
            client=SimpleNamespace(submit_cloud_ticket=submit), references=refs, stop=stop)
        self.assertEqual([t.frame.source_frame_id for t in submitted], [7])
        self.assertEqual(queue.dropped, 7)
        self.assertEqual(stats["frames_inferred"], 1)
        stale = FrameIdentity("a", "b", "c", 1, 9, time_ns()-10**9, 9)
        queue.put_latest(CapturedFrame(stale, None, 9))
        stats = await run_edge_inference_worker(queue, mode=ModelMode.DFINE_CLOUD,
            client=SimpleNamespace(submit_cloud_ticket=submit), references=refs, stop=stop)
        self.assertEqual(stats["stale_frames_dropped"], 1)
        self.assertEqual(len(submitted), 1)
