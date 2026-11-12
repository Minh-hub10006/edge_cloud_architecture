import asyncio
from dataclasses import replace
from pathlib import Path
import tempfile
from time import time_ns
from types import SimpleNamespace
import unittest

from benchmark.ordering import BenchmarkOrdering
from benchmark.pipeline import PipelineOptions, measure_pipeline
from benchmark.config import RunConfig
from contracts import ModelMode
import test_benchmark_pipeline as fixtures
from test_benchmark_pipeline import Client, Detector, frames, Tracker


def request(sequence, node="edge", phase="measure"):
    return SimpleNamespace(request_id=f"bp1-{sequence}-" + "a" * 32,
                           frame=SimpleNamespace(stream_key=(node, "camera", "stream", 1)),
                           decision_id="benchmark/c/run/" + phase)


class OrderingTests(unittest.IsolatedAsyncioTestCase):
    async def test_reverse_completion_waits_for_earlier_frame(self):
        gate = BenchmarkOrdering()
        second = gate.reserve(request(1))
        waiting = asyncio.create_task(second.wait(time_ns() + 1_000_000_000))
        await asyncio.sleep(.01)
        self.assertFalse(waiting.done())
        first = gate.reserve(request(0))
        await first.wait(time_ns() + 1_000_000_000)
        first.finish()
        await waiting
        second.finish()
        self.assertFalse(second.lane.pending)

    async def test_failed_earlier_releases_later_and_nodes_are_isolated(self):
        gate = BenchmarkOrdering()
        first = gate.reserve(request(0))
        second = gate.reserve(request(1))
        other = gate.reserve(request(0, node="other"))
        await other.wait(time_ns() + 100_000_000)
        first.finish()  # Inference failure still retires the dispatch ordinal.
        await second.wait(time_ns() + 100_000_000)
        second.finish()
        other.finish()

    async def test_missing_ordinal_times_out_and_capacity_is_bounded(self):
        gate = BenchmarkOrdering(capacity=2)
        second = gate.reserve(request(1))
        with self.assertRaises(TimeoutError):
            await second.wait(time_ns() + 5_000_000)
        second.finish()
        with self.assertRaisesRegex(ValueError, "window full"):
            gate.reserve(request(2))
        first = gate.reserve(request(0))
        first.finish()
        next_frame = gate.reserve(request(2))
        await next_frame.wait(time_ns() + 100_000_000)
        next_frame.finish()
        with self.assertRaisesRegex(ValueError, "stale"):
            gate.reserve(request(0))

    async def test_warmup_and_measure_have_separate_dense_sequences(self):
        gate = BenchmarkOrdering()
        for phase in ("warmup", "measure"):
            token = gate.reserve(request(0, phase=phase))
            await token.wait(time_ns() + 100_000_000)
            token.finish()
        self.assertEqual(len(gate.lanes), 2)


class OverlapTests(unittest.IsolatedAsyncioTestCase):
    async def test_cloud_inference_finishes_reverse_but_tracker_is_fifo(self):
        import time
        from contracts import CloudInferenceTicket, MediaFrameRef
        from cloud_core.result_hub import ResultHub
        service = await fixtures.CompletionTests.service(self)
        tracked = []
        class RecordingTracker(Tracker):
            def update(self, result):
                tracked.append(result.frame.source_frame_id)
                return super().update(result)
        service.hub = ResultHub(tracker=RecordingTracker())
        class ReverseDetector:
            model_id = "test-model"
            runtime = "cpu"
            def infer(self, pixels):
                time.sleep(.03 if pixels == 0 else .001)
                return Detector().infer(None)
        service.cloud_adapters[ModelMode.DFINE_CLOUD] = ReverseDetector()
        from cloud_core import DecodedMediaFrame
        tickets = []
        iterator = frames(count=2)
        for sequence in range(2):
            frame = (await anext(iterator)).identity
            ref = MediaFrameRef(frame, sequence, frame.source_pts, time_ns())
            await service.frame_cache.announce(ref)
            await service.frame_cache.ingest(DecodedMediaFrame(frame.stream_key,
                sequence, frame.source_pts, sequence, time_ns()))
            tickets.append(CloudInferenceTicket(frame, ModelMode.DFINE_CLOUD, "test-model",
                "benchmark/c/r/measure", frame.source_pts, time_ns() + 2_000_000_000,
                request_id=request(sequence).request_id))
        replies = await asyncio.gather(*(service.handle_cloud_ticket(t) for t in tickets))
        self.assertEqual([r[0] for r in replies], ["accepted", "accepted"])
        self.assertEqual(tracked, [0, 1])
        self.assertEqual(service._benchmark_pending, 0)

    async def test_requests_overlap_without_losing_trace_frames(self):
        class ConcurrentClient(Client):
            active = peak = 0
            async def submit_detection(self, result):
                self.active += 1
                self.peak = max(self.peak, self.active)
                try:
                    return await super().submit_detection(result)
                finally:
                    self.active -= 1
        client = ConcurrentClient(delay=.02)
        cfg = RunConfig("c", "r", "edge", "test-model", "edge", 12, 0,
                        "pipeline_capacity", "grpc")
        with tempfile.TemporaryDirectory() as tmp:
            report = await measure_pipeline(cfg, frames(), None, client, Path(tmp) / "run",
                options=PipelineOptions(queue_depth=8, max_inflight=4, priming_frames=0,
                                        padding_frames=0, resource_sample_s=0),
                mode=ModelMode.YOLO_EDGE, detector=Detector(),
                identity={"artifact_sha256": "hash", "runtime": "fake", "video_sha256": "hash"})
        self.assertEqual(report["completed"], 12)
        self.assertEqual(report["execution_status"], "PASS")
        self.assertGreater(client.peak, 1)
        self.assertLessEqual(client.peak, 4)

    def test_inflight_rejects_unbounded_or_larger_than_queue(self):
        for inflight in (0, 33):
            with self.assertRaises(ValueError):
                PipelineOptions(max_inflight=inflight)
        with self.assertRaises(ValueError):
            PipelineOptions(queue_depth=2, max_inflight=3)

    def test_small_rgb_conversion_preserves_pixels_and_prepared_tensor_is_owned(self):
        import cv2
        import numpy as np
        from inference.adapters import YoloSnpeAdapter
        from legacy.adapters.yolo.yolo_snpe import YOLOSnpeDetector
        native = YOLOSnpeDetector.__new__(YOLOSnpeDetector)
        native.imgsz, native.input_layout = 320, "nchw"
        native._pad = native._tensor = None
        adapter = YoloSnpeAdapter.__new__(YoloSnpeAdapter)
        adapter.detector = native
        image = np.random.default_rng(42).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        first = adapter.prepare_frame(image)
        old_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        old_small = cv2.resize(old_rgb, (320, 180), interpolation=cv2.INTER_LINEAR)
        expected = np.full((320, 320, 3), 114, dtype=np.uint8)
        expected[:180] = old_small
        expected_tensor = np.empty((1, 3, 320, 320), dtype=np.float32)
        np.multiply(expected.transpose(2, 0, 1), 1 / 255., out=expected_tensor[0], casting="unsafe")
        np.testing.assert_array_equal(first.tensor, expected_tensor)
        saved = first.tensor.copy()
        adapter.prepare_frame(np.zeros_like(image))
        np.testing.assert_array_equal(first.tensor, saved)
        self.assertFalse(np.shares_memory(first.tensor, native._tensor))
