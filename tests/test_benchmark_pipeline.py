import asyncio
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from time import monotonic_ns, time_ns
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import grpc
from benchmark.client import MeasuredClient, connect
from benchmark.cloud import finish_result, install_grpc
from benchmark.config import RunConfig
from benchmark.pipeline import PipelineOptions, measure_pipeline
from cloud_core import CloudFrameCache, DecodedMediaFrame, ResultHub
from cloud_core.grpc_service import CloudDataPlaneServicer
from contracts import Box, DetectionResult, FrameIdentity, MediaFrameRef, ModelMode
from contracts.errors import CloudInferenceRejected
from contracts.grpc_codec import result_from_proto
from contracts.kafka_transport import KafkaCloudClient, KafkaCloudWorker, EDGE_RESULTS_TOPIC, CLOUD_RESULTS_TOPIC, namespaced_topics
from contracts.proto import edge_video_platform_pb2 as pb
from contracts.proto import edge_video_platform_pb2_grpc as pb_grpc
from edge_agent import CapturedFrame


class Tracker:
    def update(self, result):
        return replace(result, boxes=tuple(replace(box, track_id=42) for box in result.boxes))
    def reset(self, key):
        pass


class Detector:
    model_id = "test-model"
    def infer(self, pixels):
        return (Box(.1, .1, .5, .5, .9, 2, "car"),), 1.5


class Publisher:
    def __init__(self):
        self.closed = False
        self.frames_submitted = self.padding_frames_submitted = 0
    async def start(self):
        pass
    async def publish(self, frame):
        seq = self.frames_submitted + self.padding_frames_submitted
        self.frames_submitted += 1
        return MediaFrameRef(frame.identity, seq, seq * 33_333_333, time_ns())
    async def publish_padding(self, frame):
        self.padding_frames_submitted += 1
    async def close(self):
        self.closed = True


class Client:
    def __init__(self, delay=0, corrupt=False, reject=False):
        self.delay, self.corrupt, self.reject = delay, corrupt, reject
        self.closed = False
    async def publish_media_reference(self, ref):
        pass
    async def submit_detection(self, request):
        await asyncio.sleep(self.delay)
        if self.reject:
            raise CloudInferenceRejected("deadline_exceeded")
        result = finish_result(request, started_ns=monotonic_ns(), stages={"cloud_hub_submit_ms": .2},
                               tracking_completed_ns=monotonic_ns())
        return replace(result, preview_pts=99) if self.corrupt else result
    async def submit_cloud_ticket(self, ticket):
        result = DetectionResult(ticket.frame, ticket.model_mode, ticket.model_id, ticket.decision_id,
            (), 1, time_ns(), request_id=ticket.request_id, preview_pts=ticket.preview_pts)
        return finish_result(result, started_ns=monotonic_ns(), stages={"cloud_media_join_ms": 2},
                             tracking_completed_ns=monotonic_ns(),
                             identity={"model_id": "test-model", "artifact_sha256": "hash", "runtime": "cpu"})
    async def close(self):
        self.closed = True


async def frames(count=100, node="edge", stream="bench", epoch=1):
    for i in range(count):
        yield CapturedFrame(FrameIdentity(node, "camera", stream, epoch, i, time_ns(), i * 33_333_333),
                            "pixels", monotonic_ns())


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def run_pipeline(self, *, profile="pipeline_capacity", placement="edge", warmup=2,
                           count=4, client=None, source=None, options=None, detector=None, media=True):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "run"
        self.publisher = Publisher()
        self.client = client or Client()
        cfg = RunConfig("campaign", "run", "edge", "test-model", placement, count, warmup, profile, "grpc")
        return await measure_pipeline(cfg, source or frames(), self.publisher if media else None, self.client, self.path,
            options=options or PipelineOptions(fps=240, queue_depth=2, priming_frames=0,
                                               padding_frames=0, resource_sample_s=0),
            mode=ModelMode.YOLO_EDGE if placement == "edge" else ModelMode.DFINE_CLOUD,
            detector=(detector or Detector()) if placement == "edge" else None,
            identity={"video_sha256": "hash", "artifact_sha256": "hash", "runtime": "test"})

    async def test_capacity_every_frame_warmup_excluded_and_resources_closed(self):
        report = await self.run_pipeline()
        self.assertEqual(report["execution_status"], "PASS")
        self.assertEqual(report["completed"], 4)
        self.assertEqual(report["warmup_observations"], 2)
        self.assertEqual(report["stages_ms"]["request_rtt_ms"]["count"], 4)
        self.assertNotIn("cloud_decode_ms", report["stages_ms"])
        self.assertEqual(report["offered_frames"], 4)
        self.assertTrue(self.publisher.closed and self.client.closed)

    async def test_no_media_capacity_is_unpaced_even_with_nominal_fps_one(self):
        # Neither warmup nor measurement may introduce a positive sleep in capacity mode.
        real_sleep = asyncio.sleep
        with patch("benchmark.pipeline.asyncio.sleep", wraps=real_sleep) as sleep:
            report = await self.run_pipeline(media=False, options=PipelineOptions(
                fps=1, priming_frames=30, padding_frames=15, resource_sample_s=0))
        self.assertTrue(all(call.args[0] == 0 for call in sleep.call_args_list))
        self.assertEqual(report["execution_status"], "PASS")
        self.assertEqual(report["completed"], 4)
        self.assertEqual(report["media_mode"], "none")
        self.assertEqual(report["media_padding_encoded"], 0)
        self.assertEqual(self.publisher.frames_submitted, 0)
        self.assertNotIn("ffmpeg_write_ms", report["stages_ms"])
        self.assertGreater(report["bytetrack_completion_fps"], 0)
        rows = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["preview_pts"] is None for row in rows))
        self.assertEqual([r["frame_id"] for r in rows if r["phase"] == "measure"], [2, 3, 4, 5])

    async def test_cloud_cannot_run_without_media_input(self):
        with self.assertRaisesRegex(ValueError, "requires RTSP"):
            await self.run_pipeline(media=False, placement="cloud")

    async def test_endpoint_fps_uses_only_cloud_completion_clock(self):
        class CloudClockClient(Client):
            tick = 900_000_000_000_000
            async def submit_detection(inner, request):
                result = await super().submit_detection(request)
                inner.tick += 10_000_000
                return replace(result, attributes={**result.attributes,
                    "benchmark_tracking_completed_mono_ns": inner.tick})
        report = await self.run_pipeline(media=False, warmup=0, client=CloudClockClient())
        self.assertEqual(report["completed"], 4)
        self.assertAlmostEqual(report["bytetrack_completion_fps"], 100)
        self.assertEqual(report["bytetrack_completion_interval"]["count"], 4)

    async def test_warmup_failure_stops_before_measurement(self):
        report = await self.run_pipeline(client=Client(reject=True))
        self.assertEqual(report["execution_status"], "FAIL")
        self.assertEqual(report["completed"], 0)
        self.assertIn("warmup failed", report["fatal_error"])

    async def test_timeout_keeps_failed_request_duration(self):
        report = await self.run_pipeline(warmup=0, client=Client(reject=True))
        self.assertEqual(report["frames"]["timeout"], 4)
        self.assertEqual(report["failed_stages_ms"]["request_rtt_ms"]["count"], 4)

    async def test_cross_pts_reply_not_counted(self):
        report = await self.run_pipeline(warmup=0, client=Client(corrupt=True))
        self.assertEqual(report["frames"]["error"], 4)
        self.assertEqual(report["completed"], 0)

    async def test_source_eof_report_incomplete_not_deadlock(self):
        report = await self.run_pipeline(warmup=0, source=frames(1))
        self.assertEqual(report["execution_status"], "FAIL")
        self.assertIn("source EOF", report["fatal_error"])
        self.assertTrue(self.publisher.closed and self.client.closed)

    async def test_realtime_drops_visible_not_called_lossless(self):
        report = await self.run_pipeline(profile="realtime", warmup=0, count=12, client=Client(delay=.03))
        self.assertEqual(report["execution_status"], "PASS")
        self.assertGreater(report["frames"]["dropped"], 0)
        self.assertEqual(sum(report["frames"].values()), 12)
        self.assertFalse(report["lossless_required"])

    async def test_cloud_identity_returned_not_guessed_from_edge(self):
        report = await self.run_pipeline(placement="cloud")
        self.assertEqual(report["execution_status"], "PASS")
        self.assertEqual(report["cloud_identity"]["runtime"], "cpu")

    async def test_padding_during_tail_drain_does_not_deadlock_full_queue(self):
        class NeedsPadding(Client):
            async def submit_detection(inner, request):
                if request.frame.source_frame_id >= 2:
                    while self.publisher.padding_frames_submitted == 0:
                        await asyncio.sleep(.001)
                return await super().submit_detection(request)
        report = await self.run_pipeline(warmup=0, client=NeedsPadding(),
            options=PipelineOptions(fps=240, queue_depth=2, priming_frames=0, padding_frames=4,
                                    rpc_timeout_s=.2, phase_timeout_s=1, resource_sample_s=0))
        self.assertEqual(report["execution_status"], "PASS")
        self.assertGreater(report["media_padding_encoded"], 0)

    async def test_priming_and_padding_not_measured(self):
        report = await self.run_pipeline(options=PipelineOptions(fps=240, queue_depth=2, priming_frames=2,
                                               padding_frames=1, resource_sample_s=0))
        self.assertEqual(report["completed"], 4)
        self.assertEqual(self.publisher.frames_submitted, 8)
        rows = [json.loads(line) for line in (self.path / "trace.jsonl").read_text().splitlines()]
        self.assertEqual([r["frame_id"] for r in rows if r["phase"] == "measure"], [4, 5, 6, 7])

    async def test_pipeline_timeout_is_bounded(self):
        report = await self.run_pipeline(warmup=0, client=Client(delay=.1), options=PipelineOptions(
            fps=240, priming_frames=0, padding_frames=0, rpc_timeout_s=.01, resource_sample_s=0))
        self.assertEqual(report["frames"]["timeout"], 4)


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_cloud_cap_queues_two_nodes_and_releases_slots(self):
        from contracts import CloudInferenceTicket
        import time
        service = await self.service()
        service.max_cloud_inflight = 1
        service._slots = asyncio.Semaphore(1)
        def infer(pixels):
            time.sleep(.025)
            return (), 25
        service.cloud_adapters[ModelMode.DFINE_CLOUD].infer = infer
        tickets = []
        for node in ("edge-a", "edge-b"):
            frame = (await anext(frames(node=node))).identity
            await service.handle_media_reference(MediaFrameRef(frame, 0, 0, time_ns()))
            tickets.append(CloudInferenceTicket(frame, ModelMode.DFINE_CLOUD, "test-model", "benchmark/c/r",
                           0, time_ns() + 1_000_000_000, request_id=node))
        replies = await asyncio.gather(*(service.handle_cloud_ticket(t) for t in tickets))
        self.assertTrue(all(row[0] == "accepted" for row in replies))
        self.assertGreater(replies[1][1].attributes["benchmark_stages_ms"]["cloud_admission_ms"], 10)
        self.assertEqual(service._slots._value, 1)
        self.assertFalse(service._benchmark_streams)

    async def test_live_tickets_queue_for_the_gpu_slot_and_never_hold_it_while_waiting_for_media(self):
        from contracts import CloudInferenceTicket
        import time
        service = await self.service()
        service.max_cloud_inflight = 1
        service._slots = asyncio.Semaphore(1)
        def infer(pixels):
            time.sleep(.025)
            return (), 25
        service.cloud_adapters[ModelMode.DFINE_CLOUD].infer = infer
        live = lambda frame, node: CloudInferenceTicket(frame, ModelMode.DFINE_CLOUD, "test-model", f"live/{node}",  # noqa: E731
                                                        0, time_ns() + 1_000_000_000, request_id=node)
        # edge-w's frame has not arrived yet: its ticket waits for media without taking the only slot.
        waiting = (await anext(frames(node="edge-w"))).identity
        parked = asyncio.create_task(service.handle_cloud_ticket(live(waiting, "edge-w")))
        await asyncio.sleep(.01)
        self.assertEqual(service._slots._value, 1)
        ready = []
        for node in ("edge-a", "edge-b"):
            frame = (await anext(frames(node=node))).identity
            await service.handle_media_reference(MediaFrameRef(frame, 0, 0, time_ns()))
            ready.append(live(frame, node))
        # Two ready tickets, one slot: the second queues instead of being rejected as "overloaded".
        replies = await asyncio.gather(*(service.handle_cloud_ticket(t) for t in ready))
        self.assertEqual([r[0] for r in replies], ["accepted", "accepted"])
        await service.handle_media_reference(MediaFrameRef(waiting, 0, 0, time_ns()))
        self.assertEqual((await parked)[0], "accepted")
        self.assertEqual(service._slots._value, 1)

    async def test_live_ticket_past_its_deadline_in_the_slot_queue_is_dropped_and_slot_kept(self):
        from contracts import CloudInferenceTicket
        service = await self.service()
        service.max_cloud_inflight = 1
        service._slots = asyncio.Semaphore(1)
        frame = (await anext(frames())).identity
        await service.handle_media_reference(MediaFrameRef(frame, 0, 0, time_ns()))
        await service._slots.acquire()  # GPU busy for longer than the ticket's budget
        ticket = CloudInferenceTicket(frame, ModelMode.DFINE_CLOUD, "test-model", "live/x",
                                      0, time_ns() + 30_000_000, request_id="late")
        self.assertEqual((await service.handle_cloud_ticket(ticket))[0], "deadline_exceeded")
        service._slots.release()
        self.assertEqual(service._slots._value, 1)

    async def test_caller_cancel_does_not_release_running_native_slot(self):
        from contracts import CloudInferenceTicket
        from threading import Event
        started, release = Event(), Event()
        service = await self.service()
        service.max_cloud_inflight = 1
        service._slots = asyncio.Semaphore(1)
        def infer(pixels):
            started.set()
            release.wait(timeout=1)
            return (), 1
        service.cloud_adapters[ModelMode.DFINE_CLOUD].infer = infer
        frame = (await anext(frames())).identity
        await service.handle_media_reference(MediaFrameRef(frame, 0, 0, time_ns()))
        ticket = CloudInferenceTicket(frame, ModelMode.DFINE_CLOUD, "test-model", "benchmark/c/r",
                                     0, time_ns() + 2_000_000_000, request_id="request")
        task = asyncio.create_task(service.handle_cloud_ticket(ticket))
        try:
            while not started.is_set():
                await asyncio.sleep(.001)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(service._slots._value, 0)
            self.assertEqual((await service.handle_cloud_ticket(ticket))[0], "benchmark_overloaded")
        finally:
            release.set()
            for _ in range(100):
                if not service._benchmark_streams:
                    break
                await asyncio.sleep(.005)
        self.assertEqual(service._slots._value, 1)

    async def test_real_publisher_sequence_stays_unique_after_midstream_padding(self):
        from media.ffmpeg_publisher import FFmpegH264Publisher
        from media.profile import MediaProfile
        from unittest.mock import AsyncMock
        publisher = FFmpegH264Publisher(profile=MediaProfile(), output="fake",
                                       stream_key=("edge", "camera", "bench", 1))
        publisher._process = SimpleNamespace(returncode=None, stdin=SimpleNamespace(write=lambda b: None, drain=AsyncMock()))
        frame = await anext(frames())
        with patch("media.ffmpeg_publisher._frame_bytes", return_value=b"pixels"):
            first = await publisher.publish(frame)
            await publisher.publish_padding(frame)
            second = await publisher.publish(replace(frame, identity=replace(frame.identity, source_frame_id=1)))
        self.assertEqual((first.encoded_seq, second.encoded_seq), (0, 2))

    async def service(self, enabled=True):
        cache = CloudFrameCache()
        class Taps:
            async def publish_reference(self, ref):
                await cache.announce(ref)
                await cache.ingest(DecodedMediaFrame(ref.frame.stream_key, ref.encoded_seq, ref.media_pts_ns,
                                                    "pixels", time_ns()))
        service = CloudDataPlaneServicer(hub=ResultHub(tracker=Tracker()), frame_cache=cache,
            frame_taps=Taps(), benchmark_enabled=enabled,
            dfine_adapter=SimpleNamespace(model_id="test-model", infer=Detector().infer, runtime="cpu"))
        service.benchmark_identities[ModelMode.DFINE_CLOUD] = {"model_id": "test-model", "artifact_sha256": "hash", "runtime": "cpu"}
        return service

    async def test_localhost_real_grpc_measured_edge_and_cloud(self):
        service = await self.service()
        _, browser_queue = service.hub.subscribe()
        server = grpc.aio.server()
        pb_grpc.add_EdgeCloudDataPlaneServicer_to_server(service, server)
        install_grpc(server, service)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        client = await connect("grpc", f"127.0.0.1:{port}", "edge", 1)
        try:
            cfg = RunConfig("campaign", "run", "edge", "test-model", "edge", 3, 1, "pipeline_capacity", "grpc")
            with tempfile.TemporaryDirectory() as tmp:
                report = await measure_pipeline(cfg, frames(), None, client, Path(tmp) / "edge", options=PipelineOptions(
                    fps=240, priming_frames=0, padding_frames=0, resource_sample_s=0), mode=ModelMode.YOLO_EDGE,
                    detector=Detector(), identity={"artifact_sha256": "hash", "runtime": "fake", "video_sha256": "hash"})
                self.assertEqual(report["execution_status"], "PASS")
                self.assertEqual(report["stages_ms"]["cloud_hub_submit_ms"]["count"], 3)
                self.assertEqual(report["stages_ms"]["cloud_bytetrack_update_ms"]["count"], 3)
                self.assertEqual(report["bytetrack_completion_interval"]["count"], 3)
                self.assertGreater(report["bytetrack_completion_fps"], 0)
                self.assertTrue(browser_queue.empty())
                self.assertNotIn("reference_ack_ms", report["stages_ms"])
                self.assertTrue(all(json.loads(line)["detections"] == 1
                                    for line in (Path(tmp) / "edge" / "trace.jsonl").read_text().splitlines()))
            # New stream separates frame IDs from the previous run's hub lane.
            client = await connect("grpc", f"127.0.0.1:{port}", "edge", 1)
            cfg = replace(cfg, run_id="cloud", placement="cloud")
            with tempfile.TemporaryDirectory() as tmp:
                report = await measure_pipeline(cfg, frames(stream="cloud"), Publisher(), client, Path(tmp) / "cloud", options=PipelineOptions(
                    fps=240, priming_frames=0, padding_frames=0, resource_sample_s=0), mode=ModelMode.DFINE_CLOUD,
                    identity={"video_sha256": "hash"})
                self.assertEqual(report["execution_status"], "PASS")
                self.assertEqual(report["stages_ms"]["cloud_adapter_call_ms"]["count"], 3)
                self.assertEqual(report["stages_ms"]["cloud_bytetrack_update_ms"]["count"], 3)
                self.assertGreater(report["bytetrack_completion_fps"], 0)
                self.assertTrue(browser_queue.empty())
                self.assertFalse(service.last_inference)
        finally:
            await client.close()
            await server.stop(0)

    async def test_disabled_cloud_rejects_benchmark_but_accepts_live(self):
        service = await self.service(enabled=False)
        frame = (await anext(frames())).identity
        result = DetectionResult(frame, ModelMode.YOLO_EDGE, "test-model", "benchmark/c/r", (), 1, time_ns())
        self.assertEqual((await service.handle_detection_result(result))[0], "benchmark_disabled")
        status, _, tracked = await service.handle_detection_result(replace(result, decision_id="live"))
        self.assertEqual(status, "accepted")
        self.assertNotIn("benchmark_schema", tracked.attributes)

    async def test_kafka_waits_for_cloud_reply_with_namespace(self):
        service = await self.service()
        worker = KafkaCloudWorker(service, "fake", topic_namespace="bench")
        raw = KafkaCloudClient("fake", node_id="edge", topic_namespace="bench")
        measured = MeasuredClient(raw, "kafka")
        frame = (await anext(frames())).identity
        result = DetectionResult(frame, ModelMode.YOLO_EDGE, "test-model", "benchmark/c/r",
                                 Detector().infer(None)[0], 1, time_ns(), request_id="request", preview_pts=0)
        completed = asyncio.Event()
        class Producer:
            topics = []
            async def send_and_wait(self, topic, *, key, value):
                self.topics.append(topic)
                if topic != raw.results_topic:   # the shared copy is for the results writer
                    return
                envelope = json.loads(value)
                domain = result_from_proto(pb.DetectionResult.FromString(bytes.fromhex(envelope["result"])))
                completed.set()
                raw._pending[envelope["request_id"]].set_result(domain)
        producer = Producer()
        worker._producer = producer
        async def broker_ack(topic, value):
            self.assertEqual(topic, "bench." + EDGE_RESULTS_TOPIC)
            # Broker ACK alone is intentionally not a completion event.
            asyncio.get_running_loop().call_later(.02,
                lambda: asyncio.create_task(worker._handle(EDGE_RESULTS_TOPIC, result)))
        raw.producer = object()
        with patch.object(raw, "_send_proto", side_effect=broker_ack):
            task = asyncio.create_task(measured.submit_detection(result))
            await asyncio.sleep(.005)
            self.assertFalse(task.done())
            reply = await asyncio.wait_for(task, .5)
        self.assertTrue(completed.is_set())
        self.assertEqual(reply.boxes[0].track_id, 42)
        # shared topic for the database writer + this Edge's own copy, which is the one the Edge reads (8/11)
        self.assertEqual(sorted(producer.topics), ["bench." + CLOUD_RESULTS_TOPIC, "bench." + CLOUD_RESULTS_TOPIC + ".edge"])
        self.assertEqual(raw.results_topic, "bench." + CLOUD_RESULTS_TOPIC + ".edge")

    async def test_same_frame_id_two_nodes_isolated(self):
        service = await self.service()
        for node in ("edge-a", "edge-b"):
            frame = (await anext(frames(node=node))).identity
            result = DetectionResult(frame, ModelMode.YOLO_EDGE, "test-model", "benchmark/c/r", (), 1, time_ns())
            self.assertEqual((await service.handle_detection_result(result))[0], "accepted")

    def test_namespace_default_backward_compatible_and_invalid_rejected(self):
        self.assertEqual(namespaced_topics("")[EDGE_RESULTS_TOPIC], EDGE_RESULTS_TOPIC)
        for value in ("bad/topic", "../live", "bênch", "x" * 101):
            with self.assertRaises(ValueError):
                namespaced_topics(value)
