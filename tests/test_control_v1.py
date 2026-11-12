"""Control-v1 tests: real loopback gRPC, no GPU/RTSP hardware required."""
import asyncio
from dataclasses import replace
from time import time_ns, monotonic_ns
from types import SimpleNamespace
import unittest

import grpc
import numpy as np

from contracts import Box, CloudInferenceTicket, DetectionResult, FrameIdentity, MediaFrameRef, ModelMode
from contracts.control_v1 import validate_feedback, validate_result, finite
from contracts.control_client_v1 import GrpcControlClient
from contracts.proto import edge_video_platform_pb2_grpc as pb_grpc
from contracts.errors import CloudInferenceRejected
from cloud_core import CloudFrameCache, DecodedMediaFrame, ResultHub
from cloud_core.grpc_service import CloudDataPlaneServicer
from cloud_core.control_v1 import CloudTelemetry, ControlService, install_grpc, sample_forever
from edge_agent.state_v1 import FeedbackTracker, TimeWindowMonitor, RuleController, Action, FEATURES

Y, D = ModelMode.YOLO_EDGE, ModelMode.DFINE_CLOUD
KEY = ('edge-a', 'camera', 'control', 1)


class Clock:
    def __init__(self): self.ns = 0
    def __call__(self): return self.ns
    def advance(self, ms): self.ns += round(ms * 1e6)


def frame(i=0, node='edge-a'):
    return FrameIdentity(node, 'camera', 'control', 1, i, time_ns(), i * 33_333_333)


def ticket(i=0, node='edge-a', model='dfine_full'):
    f = frame(i, node)
    return CloudInferenceTicket(frame=f, model_mode=D, model_id=model,
        decision_id='control-v1:test:0', request_id=f'{node}-{i}', preview_pts=f.source_pts,
        deadline_utc_ns=f.capture_utc_ns + 5_000_000_000)


def feedback(sequence=1, instance='cloud', pressure=.2):
    return dict(control_v1=1, cloud_instance_id=instance, sequence=sequence,
        window_ms=1000., measurement_to_send_ms=5., cloud_inflight_norm=.3,
        cloud_arrival_pressure_raw=pressure, cloud_arrival_pressure_norm=min(pressure, 1),
        unknown_capacity_tickets=0, arrival_rate_rps=5.)


def event(i=0, decision='decision', status='accepted', timely=True):
    return dict(stream_key=list(KEY), frame_id=i, model_id='yolo', decision_id=decision,
        status=status, timely=timely, transport_overhead_ms=20., active_path_latency_ms=50.,
        raw_confidences=[.8], calibrated_confidences=[], box_change_mean=.2)


class StateTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.monitor = TimeWindowMonitor(KEY, clock=self.clock)
        self.action = Action(Y, 15, 'decision', 0)

    def test_fixed_time_fps_reward_and_missing(self):
        self.monitor.observation(event())
        self.monitor.queue(2, 8)
        self.clock.advance(1000)
        result = self.monitor.finish(self.action)
        self.assertEqual(len(result['state']), 10)
        self.assertEqual(set(result['state']), set(FEATURES))
        self.assertEqual(result['raw']['effective_fps'], 1)
        self.assertAlmostEqual(result['reward'], 1 / 30)
        self.assertEqual(result['state']['edge_backlog_norm'], .25)
        self.assertEqual(result['confidence_source'], 'raw_debug')
        self.assertIsNone(result['raw']['calibrated_confidence_mean'])
        self.assertFalse(result['availability_mask']['cloud_inflight_norm'])
        self.assertFalse(result['training_ready'])

    def test_failures_late_and_switch_penalty(self):
        self.monitor.observation(event(0, timely=False))
        self.monitor.observation(event(1, status='deadline_exceeded'))
        self.clock.advance(1000)
        result = self.monitor.finish(self.action, switches=1)
        self.assertEqual(result['raw']['effective_fps'], 1.)
        self.assertEqual(result['raw']['timely_fps'], 0.)
        self.assertAlmostEqual(result['reward'], -1.05)

    def test_previous_action_not_relabelled(self):
        self.monitor.observation(event(0, decision='old'))
        self.clock.advance(1000)
        result = self.monitor.finish(self.action)
        self.assertEqual(result['raw']['effective_fps'], 1)
        self.assertIsNone(result['raw']['active_path_latency_ms'])
        self.assertIsNone(result['state']['confidence_mean'])
        self.assertEqual(result['action_results']['old']['accepted'], 1)

    def test_order_dedup_and_bounded_samples(self):
        self.assertTrue(self.monitor.observation(event(100)))
        self.assertFalse(self.monitor.observation(event(100)))
        self.assertFalse(self.monitor.observation(event(99)))
        self.assertTrue(self.monitor.observation(event(101)))
        self.assertEqual(self.monitor.duplicates, 2)
        small = TimeWindowMonitor(KEY, clock=self.clock, max_samples=1)
        small.add('laplacian_variance', 100)
        small.add('laplacian_variance', 200)
        self.assertEqual(small.overflow, 1)

    def test_other_stream_and_invalid_numeric_rejected(self):
        wrong = event()
        wrong['stream_key'][0] = 'edge-b'
        with self.assertRaises(ValueError): self.monitor.observation(wrong)
        for value in [None, float('nan'), float('inf'), -1, True]:
            with self.assertRaises(ValueError): finite(value, 'sample')
        with self.assertRaises(ValueError): self.monitor.add('raw_confidence', 1.1)

    def test_empty_window_is_zero_fps_not_zero_latency(self):
        self.clock.advance(1000)
        result = self.monitor.finish(self.action)
        self.assertEqual(result['raw']['effective_fps'], 0)
        self.assertIsNone(result['state']['active_path_latency_norm'])

    def test_age_growth_duplicates_restart_and_no_fake_zero(self):
        tracker = self.monitor.feedback
        self.assertIsNone(tracker.age(40))
        tracker.accept(feedback())
        self.assertFalse(tracker.accept(feedback()))
        self.clock.advance(200)
        self.assertEqual(tracker.age(40)['estimated_ms'], 225)
        self.assertIsNone(tracker.age(None)['norm'])
        self.clock.advance(3000)
        self.assertEqual(tracker.age(40)['norm'], 1)
        self.assertTrue(tracker.accept(feedback(instance='new')))
        self.assertFalse(tracker.accept(feedback(sequence=100)))

    def test_validate_feedback_rejects_nonfinite(self):
        for name in ['window_ms', 'measurement_to_send_ms', 'cloud_inflight_norm']:
            value = feedback()
            value[name] = float('nan')
            with self.assertRaises(ValueError): validate_feedback(value)

    def test_policy_cooldown_fps_and_stale_fallback(self):
        controller = RuleController(Y, 15, allowed_modes=(Y, D), fps_levels=(5, 15, 30),
            stream_key=KEY, policy='rules', cooldown_s=1, clock=self.clock)
        self.monitor.feedback.accept(feedback())
        self.monitor.observation(event())
        self.monitor.queue(0, 8)
        self.clock.advance(1000)
        snapshot = self.monitor.finish(controller.action)
        # Event must belong to the current decision to provide active latency.
        snapshot['raw']['active_path_latency_ms'] = 10
        action, reason = controller.decide(snapshot)
        self.assertEqual((action.fps, reason), (30, 'increase_fps'))
        self.assertEqual(controller.decide(snapshot)[1], 'cooldown')
        controller.action = Action(D, 15, 'cloud', self.clock())
        snapshot['cloud_feedback'] = None
        action, reason = controller.decide(snapshot)
        self.assertEqual((action.mode, reason), (Y, 'cloud_feedback_stale'))

    def test_observe_does_not_switch(self):
        controller = RuleController(D, 15, allowed_modes=(Y, D), fps_levels=(15,),
            stream_key=KEY, clock=self.clock)
        self.clock.advance(1000)
        snapshot = self.monitor.finish(controller.action)
        self.assertEqual(controller.decide(snapshot)[0], controller.action)


class PressureTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.core = SimpleNamespace(max_cloud_inflight=2, _slots=SimpleNamespace(_value=1))
        self.telemetry = CloudTelemetry(self.core, {'dfine_full': 25, 'rtdetr': 10}, clock=self.clock)

    def test_weighted_multi_node_pressure_and_dedup(self):
        for i in range(8): self.telemetry.arrival(ticket(i, 'node-a'))
        for i in range(4): self.telemetry.arrival(ticket(i, 'node-b', 'rtdetr'))
        self.assertFalse(self.telemetry.arrival(ticket(0, 'node-a')))
        self.telemetry.sample()
        self.clock.advance(1000)
        result = self.telemetry.snapshot()
        self.assertAlmostEqual(result['cloud_arrival_pressure_raw'], .72)
        self.assertEqual(result['cloud_inflight_norm'], .5)
        self.assertEqual(result['arrival_rate_rps'], 12)

    def test_over_capacity_and_actual_elapsed(self):
        for i in range(20): self.telemetry.arrival(ticket(i))
        for i in range(5): self.telemetry.arrival(ticket(i, 'node-b', 'rtdetr'))
        self.clock.advance(1000)
        self.assertAlmostEqual(self.telemetry.snapshot()['cloud_arrival_pressure_raw'], 1.3)
        self.assertEqual(self.telemetry.snapshot()['cloud_arrival_pressure_norm'], 1)
        for i in range(25): self.telemetry.arrival(ticket(i, 'node-c'))
        self.clock.advance(2000)
        self.assertAlmostEqual(self.telemetry.snapshot()['cloud_arrival_pressure_raw'], .5)

    def test_unknown_capacity_unavailable_empty_zero(self):
        self.telemetry.arrival(ticket(model='unknown'))
        self.clock.advance(1000)
        self.assertIsNone(self.telemetry.snapshot()['cloud_arrival_pressure_norm'])
        self.clock.advance(1000)
        self.assertEqual(self.telemetry.snapshot()['cloud_arrival_pressure_norm'], 0)

    def test_invalid_capacity_and_expired_ticket(self):
        for capacity in [0, -1, float('nan')]:
            with self.assertRaises(ValueError): CloudTelemetry(self.core, {'dfine': capacity})
        t = ticket()
        expired = replace(t, deadline_utc_ns=t.frame.capture_utc_ns + 1)
        self.assertFalse(self.telemetry.arrival(expired))


class Tracker:
    def update(self, result):
        return replace(result, boxes=tuple(replace(box, track_id=1) for box in result.boxes))
    def reset(self, key): pass


class Detector:
    model_id = 'dfine_full'
    runtime = 'fake-test'
    def infer(self, pixels):
        return (Box(.1, .1, .5, .6, .9, 2, 'car'),), 1.


class GrpcIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cache = CloudFrameCache()
        cache = self.cache
        class Tap:
            async def publish_reference(self, ref):
                await cache.announce(ref)
                await cache.ingest(DecodedMediaFrame(stream_key=ref.frame.stream_key,
                    encoded_seq=ref.encoded_seq, media_pts_ns=ref.media_pts_ns,
                    pixels=np.zeros((8, 8, 3), dtype=np.uint8), received_utc_ns=time_ns()))
        self.core = CloudDataPlaneServicer(hub=ResultHub(tracker=Tracker()), frame_cache=self.cache,
            frame_taps=Tap(), dfine_adapter=Detector(), max_cloud_inflight=2)
        self.telemetry = CloudTelemetry(self.core, {'dfine_full': 100}, window_ms=100)
        self.core.control_ticket_observer = self.telemetry
        self.service = ControlService(self.core, self.telemetry)
        self.server = grpc.aio.server()
        pb_grpc.add_EdgeCloudDataPlaneServicer_to_server(self.core, self.server)
        install_grpc(self.server, self.service)
        port = self.server.add_insecure_port('127.0.0.1:0')
        await self.server.start()
        self.client = GrpcControlClient(f'127.0.0.1:{port}', timeout_s=2)
        await self.client.start()
        self.sampler = asyncio.create_task(sample_forever(self.telemetry))

    async def asyncTearDown(self):
        self.sampler.cancel()
        await asyncio.gather(self.sampler, return_exceptions=True)
        await self.client.close()
        await self.server.stop(0)
        await self.service.close()

    async def test_tracked_completion_both_routes_multiple_nodes(self):
        self.telemetry.window_ms = 10_000  # keep all arrivals in one test window
        for node in ['node-1', 'node-2', 'node-3', 'node-4']:
            t = ticket(0, node)
            await self.client.publish_media_reference(MediaFrameRef(t.frame, 0, t.preview_pts, time_ns()))
            result = await self.client.submit_cloud_ticket(t)
            self.assertEqual(result.frame, t.frame)
            self.assertEqual(result.boxes[0].track_id, 1)
            f = frame(1, node)
            request = DetectionResult(frame=f, model_mode=Y, model_id='yolo', decision_id='control-v1:test:1',
                request_id=f'{node}-yolo', preview_pts=f.source_pts, boxes=Detector().infer(None)[0],
                inference_ms=1, produced_utc_ns=time_ns(),
                attributes={'deadline_utc_ns': f.capture_utc_ns + 5_000_000_000})
            result = await self.client.submit_detection(request)
            self.assertEqual(result.boxes[0].track_id, 1)
            validate_result(result, request)
            with self.assertRaises(ValueError): validate_result(replace(result, request_id='wrong'), request)
        self.assertEqual(self.telemetry.tickets, 4)

    async def test_cancel_keeps_owned_native_work_and_slot(self):
        import threading
        started, release = threading.Event(), threading.Event()
        class Blocking(Detector):
            def infer(self, pixels):
                started.set()
                release.wait(3)
                return super().infer(pixels)
        self.core.cloud_adapters[D] = Blocking()
        t = ticket()
        await self.client.publish_media_reference(MediaFrameRef(t.frame, 0, 0, time_ns()))
        task = asyncio.create_task(self.service.handle_cloud_ticket(t))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self.assertEqual(self.core._slots._value, 1)
            self.assertEqual(len(self.service.tasks), 1)
        finally:
            release.set()
            await self.service.close()
        self.assertEqual(self.core._slots._value, 2)

    async def test_wrong_model_and_missing_deadline_rejected(self):
        with self.assertRaises(CloudInferenceRejected):
            await self.client.submit_cloud_ticket(ticket(model='wrong'))
        f = frame()
        r = DetectionResult(frame=f, model_mode=Y, model_id='yolo', decision_id='decision',
            request_id='r', preview_pts=0, boxes=(), inference_ms=1, produced_utc_ns=time_ns())
        with self.assertRaises(CloudInferenceRejected): await self.client.submit_detection(r)

    async def test_publisher_failure_closes_source_and_does_not_hang(self):
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        closed = asyncio.Event()
        class Source:
            async def __aiter__(self):
                try:
                    while True:
                        yield CapturedFrame(identity=frame(), pixels=np.zeros((8, 8, 3), dtype=np.uint8),
                                            capture_mono_ns=monotonic_ns())
                        await asyncio.sleep(.05)
                finally:
                    closed.set()
        class Publisher:
            profile = SimpleNamespace(fps=30)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured): raise RuntimeError('publisher test failure')
        controller = RuleController(D, 15, allowed_modes=(D,), fps_levels=(15,), stream_key=KEY)
        with self.assertRaisesRegex(RuntimeError, 'publisher test failure'):
            await asyncio.wait_for(run_pipeline(Source(), Publisher(), self.client, {}, controller,
                TimeWindowMonitor(KEY), emit=lambda value: None), timeout=2)
        self.assertTrue(closed.is_set())

    async def test_legacy_endpoint_still_operates(self):
        f = frame()
        r = DetectionResult(frame=f, model_mode=Y, model_id='yolo', decision_id='legacy',
            request_id='legacy', preview_pts=0, boxes=(), inference_ms=1, produced_utc_ns=time_ns())
        self.assertEqual(await self.client.raw.submit_detection(r), 'accepted')

    async def test_actual_executor_switches_without_changing_frame_identity(self):
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        key = KEY
        class Source:
            async def __aiter__(self):
                for i in range(36):
                    yield CapturedFrame(identity=frame(i), pixels=np.zeros((16, 16, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(1 / 30)
        class Publisher:
            profile = SimpleNamespace(fps=30)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                ref = MediaFrameRef(captured.identity, self.frames_submitted,
                    self.frames_submitted * 33_333_333, time_ns())
                self.frames_submitted += 1
                await self.on_frame_submitted(ref)
            async def publish_padding(self, captured): self.padding_frames_submitted += 1
        class Yolo(Detector): model_id = 'yolo'
        controller = RuleController(Y, 30, allowed_modes=(Y, D), fps_levels=(15, 30),
            stream_key=key, policy='cycle', cooldown_s=.2)
        monitor = TimeWindowMonitor(key, window_ms=200)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback())
        windows = []
        summary = await run_pipeline(Source(), Publisher(), self.client, {Y: Yolo()}, controller, monitor,
            priming_frames=0, padding_frames=0, blur_width=16, emit=windows.append)
        self.assertGreaterEqual(summary['stats']['action_changes'], 2)
        self.assertGreater(summary['stats']['frames_inferred'], 10)
        self.assertEqual(summary['stats'].get('processing_errors', 0), 0)
        modes = {e['model_mode'] for w in windows for e in w['frames'] if e['status'] == 'accepted'}
        self.assertEqual(modes, {'yolo_edge', 'dfine_cloud'})
        self.assertTrue(all(tuple(w['stream_key']) == key for w in windows))

    async def test_jpeg_tickets_leave_in_frame_order_with_two_in_flight(self):
        import time
        from edge_agent import control_v1
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        delays = iter([.12, .01] * 20)
        real = control_v1.encode_cloud_jpeg

        def uneven(pixels, size, quality):
            time.sleep(next(delays))
            return real(pixels, size, quality)
        order = []
        real_submit = self.client.submit_cloud_ticket

        async def spy(ticket):
            order.append(ticket.frame.source_frame_id)
            return await real_submit(ticket)
        self.client.submit_cloud_ticket = spy
        key = ('edge-j', 'camera', 'control', 1)

        class Source:
            async def __aiter__(self):
                for i in range(8):
                    yield CapturedFrame(identity=frame(i, 'edge-j'), pixels=np.zeros((16, 16, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(.05)

        class Publisher:
            profile = SimpleNamespace(fps=15)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                ref = MediaFrameRef(captured.identity, self.frames_submitted, self.frames_submitted * 66_666_667, time_ns())
                self.frames_submitted += 1
                await self.on_frame_submitted(ref)
            async def publish_padding(self, captured): self.padding_frames_submitted += 1

        controller = RuleController(D, 15, allowed_modes=(D,), fps_levels=(15,), stream_key=key)
        monitor = TimeWindowMonitor(key, window_ms=200)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback())
        control_v1.encode_cloud_jpeg = uneven
        try:
            summary = await run_pipeline(Source(), Publisher(), self.client, {}, controller, monitor, priming_frames=0,
                padding_frames=0, blur_width=8, emit=lambda value: None, cloud_jpeg_quality=85, cloud_inflight=2)
        finally:
            control_v1.encode_cloud_jpeg = real
        self.assertEqual(summary['error_reasons'].get('out_of_order', 0), 0, summary)
        self.assertGreaterEqual(len(order), 3)
        self.assertEqual(order, sorted(order))

    async def test_mode_a_tracks_on_the_edge_and_publishes_without_a_cloud_reply(self):
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import EDGE_TRACK_ID_BASE, run_pipeline
        from edge_agent.slow_lane import build_slow_lane
        published, tracked = [], []

        class Tracker:  # stands in for cloud_core.tracking.CloudByteTrack
            def update(self, result):
                tracked.append(result.frame.source_frame_id)
                return replace(result, boxes=tuple(replace(b, track_id=7) for b in result.boxes))

        async def publish(result):
            published.append(result)
        self.client.publish_tracked_result = publish

        async def no_round_trip(result):
            raise AssertionError("mode A with Edge tracking must not wait for the Cloud")
        self.client.submit_detection = no_round_trip

        class Source:
            async def __aiter__(self):
                for i in range(6):
                    yield CapturedFrame(identity=frame(i), pixels=np.zeros((16, 16, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(1 / 15)

        class Publisher:
            profile = SimpleNamespace(fps=15)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                ref = MediaFrameRef(captured.identity, self.frames_submitted, self.frames_submitted * 66_666_667, time_ns())
                self.frames_submitted += 1
                await self.on_frame_submitted(ref)
            async def publish_padding(self, captured): self.padding_frames_submitted += 1

        class Yolo(Detector): model_id = 'yolo'
        controller = RuleController(Y, 15, allowed_modes=(Y,), fps_levels=(15,), stream_key=KEY)
        monitor = TimeWindowMonitor(KEY, window_ms=200)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback())
        summary = await run_pipeline(Source(), Publisher(), self.client, {Y: Yolo()}, controller, monitor,
            priming_frames=0, padding_frames=0, blur_width=16, emit=lambda value: None, edge_tracker=Tracker(),
            slow_lane=build_slow_lane('check'))
        self.assertEqual(summary['stats'].get('processing_errors', 0), 0, summary)
        self.assertGreaterEqual(len(published), 3)
        # slow lane (8/11): every result reached its BUS task, with the tracked IDs, and the fast lane did not wait
        slow = summary['slow_lane']
        self.assertEqual(slow['published'], summary['stats']['frames_inferred'])
        self.assertEqual((slow['tasks']['check']['events'], slow['tasks']['check']['dropped']), (slow['published'], 0))
        self.assertEqual(tracked, sorted(tracked))
        self.assertTrue(all(b.track_id == 7 + EDGE_TRACK_ID_BASE for r in published for b in r.boxes))
        self.assertEqual(published[0].attributes['tracker'], 'edge')

    async def test_plate_branch_sees_every_published_frame_whatever_the_route(self):
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        from edge_agent.plate_branch import PlateBranch
        from edge_agent.plate_policy import PlateReadPolicy
        class Source:
            async def __aiter__(self):
                for i in range(30):
                    yield CapturedFrame(identity=frame(i), pixels=np.zeros((64, 64, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(1 / 30)
        class Publisher:
            profile = SimpleNamespace(fps=30)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                ref = MediaFrameRef(captured.identity, self.frames_submitted,
                    self.frames_submitted * 33_333_333, time_ns())
                self.frames_submitted += 1
                await self.on_frame_submitted(ref)
            async def publish_padding(self, captured): self.padding_frames_submitted += 1
        class Ocr:
            async def read(self, header, jpeg): return {"text": "29E74852", "calls": 3}
        seen = []
        def detect(pixels):
            seen.append(1)
            return [(10, 10, 40, 30, .9)]
        plate = PlateBranch(detect, PlateReadPolicy(every=3, cap=5, confirm=2), Ocr(), stream_key=KEY,
                            queue_capacity=32)
        controller = RuleController(D, 10, allowed_modes=(D,), fps_levels=(10,), stream_key=KEY)
        monitor = TimeWindowMonitor(KEY, window_ms=200)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback())
        summary = await run_pipeline(Source(), Publisher(), self.client, {}, controller, monitor,
            priming_frames=0, padding_frames=0, blur_width=16, emit=lambda value: None, plate=plate)
        # analytics run at 10 FPS on the Cloud, plates on all 30 published frames
        self.assertEqual(summary['plate']['frames'], 30)
        self.assertEqual(len(seen), 30)
        self.assertEqual(summary['plate']['ends'], {'confirmed': 1})
        self.assertEqual(summary['plate']['replies'], 2)

    async def test_two_cloud_tickets_in_flight_double_the_rate_and_keep_tracker_order(self):
        # Cloud call alternates 160 ms / 60 ms (serial: ~9 FPS): with two tickets in flight the second often finishes first,
        # and the per-stream order gate must hold it so ByteTrack never sees an older frame after a newer one.
        import time
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        calls = []
        def slow_infer(pixels):
            calls.append(1)
            time.sleep(.16 if len(calls) % 2 else .06)
            return (Box(.1, .1, .5, .6, .9, 2, 'car'),), 1.
        self.core.cloud_adapters[D].infer = slow_infer

        async def run(inflight, node):
            key = (node, 'camera', 'control', 1)
            class Source:
                async def __aiter__(self):
                    for i in range(30):
                        yield CapturedFrame(identity=frame(i, node), pixels=np.zeros((8, 8, 3), dtype=np.uint8),
                                            capture_mono_ns=monotonic_ns())
                        await asyncio.sleep(1 / 15)
            class Publisher:
                profile = SimpleNamespace(fps=15)
                frames_submitted = padding_frames_submitted = 0
                async def __aenter__(self): return self
                async def __aexit__(self, *args): pass
                async def publish(self, captured):
                    ref = MediaFrameRef(captured.identity, self.frames_submitted,
                                        self.frames_submitted * 66_666_667, time_ns())
                    self.frames_submitted += 1
                    await self.on_frame_submitted(ref)
                async def publish_padding(self, captured): self.padding_frames_submitted += 1
            controller = RuleController(D, 15, allowed_modes=(D,), fps_levels=(15,), stream_key=key)
            monitor = TimeWindowMonitor(key, window_ms=200)
            await asyncio.sleep(.12)  # first telemetry window
            monitor.feedback.accept(await self.client.get_feedback())
            return await run_pipeline(Source(), Publisher(), self.client, {}, controller, monitor,
                priming_frames=0, padding_frames=0, blur_width=8, emit=lambda value: None, cloud_inflight=inflight)

        serial = await run(1, 'edge-s')
        paired = await run(2, 'edge-p')
        self.assertEqual(paired['error_reasons'].get('out_of_order', 0), 0)
        self.assertEqual(paired['stats'].get('processing_errors', 0), 0)
        # Order gate: a fast ticket completes with the slow one before it, so pairs finish every ~160 ms here.
        self.assertGreaterEqual(paired['stats']['frames_inferred'], serial['stats']['frames_inferred'] * 1.2)
        self.assertEqual(self.core._live_pending, {})

    async def test_path_b_ticket_carries_the_head_tensor_and_reaches_the_split_adapter(self):
        from contracts.cut_tensor import pack, unpack
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        S = ModelMode.RTDETR_SPLIT_CLOUD
        t = np.ones((1, 4, 2, 2), np.float32)
        encoded, seen = [], []

        class Head:  # the Edge half on the NPU
            def encode(self, bgr):
                encoded.append(bgr.shape)
                return pack(t, width=bgr.shape[1], height=bgr.shape[0], head_id='h'), {
                    'prepare_ms': 1., 'npu_ms': 2., 'pack_ms': 3.}

        class Tail:  # the Cloud half
            model_id, runtime = 'split', 'fake'

            def infer(self, payload):
                seen.append(unpack(payload, expected_head_id='h')[1:])
                return (Box(.1, .1, .5, .6, .9, 2, 'car'),), 1.
        self.core.cloud_adapters[S] = Tail()

        class Source:
            async def __aiter__(self):
                for i in range(5):
                    yield CapturedFrame(identity=frame(i), pixels=np.zeros((54, 96, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(.1)

        class Publisher:
            profile = SimpleNamespace(fps=15)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                ref = MediaFrameRef(captured.identity, self.frames_submitted, self.frames_submitted * 66_666_667, time_ns())
                self.frames_submitted += 1
                await self.on_frame_submitted(ref)
            async def publish_padding(self, captured): self.padding_frames_submitted += 1

        controller = RuleController(S, 15, allowed_modes=(S,), fps_levels=(15,), stream_key=KEY)
        monitor = TimeWindowMonitor(KEY, window_ms=200)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback())
        events = []
        summary = await run_pipeline(Source(), Publisher(), self.client, {}, controller, monitor, priming_frames=0,
            padding_frames=0, blur_width=8, emit=events.append, cloud_model_ids={S: 'split'}, cloud_jpeg_quality=85,
            split_head=Head())
        self.assertEqual(summary['stats'].get('processing_errors', 0), 0, summary)
        self.assertGreaterEqual(summary['stats']['frames_inferred'], 3)
        self.assertEqual(seen[0], (96, 54))                 # the source size travels in the tensor header
        self.assertEqual(encoded[0], (54, 96, 3))
        self.assertEqual(self.core.last_inference, {})      # no pixels to keep for the web snapshot
        # two tickets in flight: a slow encode must not let the next frame leave first
        import time
        delays = iter([.12, .01] * 20)
        real_encode = Head.encode

        def uneven(self_, bgr):
            time.sleep(next(delays))
            return real_encode(self_, bgr)
        Head.encode = uneven
        order = []
        real_submit = self.client.submit_cloud_ticket

        async def spy(ticket):
            order.append(ticket.frame.source_frame_id)
            return await real_submit(ticket)
        self.client.submit_cloud_ticket = spy
        key2 = ('edge-b', 'camera', 'control', 1)

        class Source2:
            async def __aiter__(self):
                for i in range(8):
                    yield CapturedFrame(identity=frame(i, 'edge-b'), pixels=np.zeros((54, 96, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(.05)
        controller2 = RuleController(S, 15, allowed_modes=(S,), fps_levels=(15,), stream_key=key2)
        monitor2 = TimeWindowMonitor(key2, window_ms=200)
        await asyncio.sleep(.12)
        monitor2.feedback.accept(await self.client.get_feedback())
        paired = await run_pipeline(Source2(), Publisher(), self.client, {}, controller2, monitor2, priming_frames=0,
            padding_frames=0, blur_width=8, emit=lambda value: None, cloud_model_ids={S: 'split'}, cloud_jpeg_quality=85,
            split_head=Head(), cloud_inflight=2)
        self.assertEqual(paired['error_reasons'].get('out_of_order', 0), 0, paired)
        self.assertGreaterEqual(len(order), 3)
        self.assertEqual(order, sorted(order))
        with self.assertRaisesRegex(ValueError, 'split_head'):
            await run_pipeline(Source(), Publisher(), self.client, {}, controller, monitor,
                cloud_model_ids={S: 'split'}, emit=lambda value: None)

    async def test_yolo_executor_publishes_web_overlay_identity_contract(self):
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        from media.frame_identity import frame_pts_ns
        class Source:
            async def __aiter__(self):
                for i in range(3):
                    yield CapturedFrame(identity=frame(i), pixels=np.zeros((8, 8, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(.04)
        class Publisher:
            profile = SimpleNamespace(fps=30)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                seq = self.frames_submitted
                self.frames_submitted += 1
                await self.on_frame_submitted(MediaFrameRef(captured.identity, seq, frame_pts_ns(seq, 30), time_ns()))
        class Yolo(Detector): model_id = 'yolo'
        received = []
        real_submit = self.client.submit_detection
        async def record(request):
            result = await real_submit(request)
            # Mirrors live_sync.js pushResult/select preconditions on the
            # actual tracked result returned from the production Hub path.
            self.assertEqual(result.attributes.get('media_identity'), 'pixel-v1')
            seq = result.attributes['media_encoded_seq']
            fps = result.attributes['media_fps']
            self.assertEqual(fps, 30)
            self.assertEqual(result.preview_pts, frame_pts_ns(seq, fps))
            self.assertTrue(result.boxes)
            received.append(result)
            return result
        self.client.submit_detection = record
        controller = RuleController(Y, 30, allowed_modes=(Y,), fps_levels=(30,), stream_key=KEY)
        summary = await run_pipeline(Source(), Publisher(), self.client, {Y: Yolo()}, controller,
            TimeWindowMonitor(KEY), priming_frames=0, padding_frames=0, blur_width=8,
            emit=lambda value: None)
        self.assertGreater(len(received), 0)
        self.assertEqual(summary['stats'].get('processing_errors', 0), 0)

    async def test_executor_uses_file_selection_grid_not_decode_completion_clock(self):
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        from media.frame_identity import frame_pts_ns
        class Source:
            async def __aiter__(self):
                for i in range(12):
                    # All decodes occur almost immediately. Source grid still
                    # spans400ms. Capture times remain REAL for latency checks.
                    yield CapturedFrame(identity=frame(i), pixels=np.zeros((8, 8, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns(), cadence_ns=round(i * 1e9 / 30))
                    await asyncio.sleep(.001)
        class Publisher:
            profile = SimpleNamespace(fps=30)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                seq = self.frames_submitted
                self.frames_submitted += 1
                await self.on_frame_submitted(MediaFrameRef(captured.identity, seq, frame_pts_ns(seq, 30), time_ns()))
        class Yolo(Detector): model_id = 'yolo'
        controller = RuleController(Y, 15, allowed_modes=(Y,), fps_levels=(15,), stream_key=KEY)
        summary = await run_pipeline(Source(), Publisher(), self.client, {Y: Yolo()}, controller,
            TimeWindowMonitor(KEY), priming_frames=0, padding_frames=0, blur_width=8,
            emit=lambda value: None)
        self.assertEqual(summary['media_encoded'], 12)
        self.assertEqual(summary['stats']['analytics_selected'], 6)
        self.assertEqual(summary['stats'].get('media_fps_skipped', 0), 0)
        self.assertEqual(summary['media_queue_dropped'], 0)
        self.assertEqual(summary['media_ref_send_failed'], 0)

    async def test_analytics_keeps_every_media_frame_when_fps_match_on_a_25fps_source(self):
        # 25 fps source -> 15 fps media (uneven 80/40 ms spacing). Analytics at 15 starting on media
        # frame 1 (priming) used to lock onto 10 fps; it must take every media frame after priming.
        from edge_agent.capture import CapturedFrame
        from edge_agent.control_v1 import run_pipeline
        from media.frame_identity import frame_pts_ns
        class Source:
            async def __aiter__(self):
                for i in range(50):
                    yield CapturedFrame(identity=frame(i), pixels=np.zeros((8, 8, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns(), cadence_ns=round(i * 1e9 / 25))
                    await asyncio.sleep(.001)
        class Publisher:
            profile = SimpleNamespace(fps=15)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                seq = self.frames_submitted
                self.frames_submitted += 1
                await self.on_frame_submitted(MediaFrameRef(captured.identity, seq, frame_pts_ns(seq, 15), time_ns()))
        class Yolo(Detector): model_id = 'yolo'
        controller = RuleController(Y, 15, allowed_modes=(Y,), fps_levels=(15,), stream_key=KEY)
        summary = await run_pipeline(Source(), Publisher(), self.client, {Y: Yolo()}, controller,
            TimeWindowMonitor(KEY), priming_frames=1, padding_frames=0, blur_width=8,
            emit=lambda value: None)
        self.assertEqual(summary['media_encoded'], 30)
        self.assertEqual(summary['stats']['analytics_selected'], 29)
        self.assertEqual(summary['media_queue_dropped'], 0)

    async def test_kafka_actual_codec_worker_feedback_four_nodes_fake_broker(self):
        # Only broker I/O is fake; exercise production serializers, lane ordering,
        # reply correlation, topic namespace and feedback consumer.
        from unittest.mock import patch
        from collections import namedtuple
        from contracts.kafka_transport import KafkaCloudWorker
        from contracts.control_client_v1 import KafkaControlClient
        from contracts.control_v1 import KAFKA_NAMESPACE
        import sys
        consumers, offsets = [], {}
        class Consumer:
            def __init__(self, *topics, **options):
                self.topics, self.group, self.queue = topics, options['group_id'], asyncio.Queue()
                self.commits = []
            async def start(self): consumers.append(self)
            async def stop(self):
                if self in consumers: consumers.remove(self)
            def __aiter__(self): return self
            async def __anext__(self): return await self.queue.get()
            async def commit(self, values): self.commits.append(values)
        class Producer:
            def __init__(self, **options): pass
            async def start(self): pass
            async def stop(self): pass
            async def send_and_wait(self, topic, *, value, key=None):
                offset = offsets.get(topic, 0)
                offsets[topic] = offset + 1
                groups = set()
                for c in consumers:
                    if topic in c.topics and c.group not in groups:
                        groups.add(c.group)
                        c.queue.put_nowait(SimpleNamespace(topic=topic, partition=0, offset=offset,
                                                          key=key, value=value))
        stub = SimpleNamespace(TopicPartition=namedtuple('TopicPartition', 'topic partition'))
        with patch('contracts.kafka_transport._kafka_types', return_value=(Consumer, Producer)), \
             patch.dict(sys.modules, {'aiokafka': stub}):
            worker = KafkaCloudWorker(self.service, 'fake-broker', topic_namespace=KAFKA_NAMESPACE)
            clients = [KafkaControlClient('fake-broker', node_id=f'kafka-{i}', timeout_s=2) for i in range(4)]
            await worker.start()
            sampler = asyncio.create_task(sample_forever(self.telemetry, kafka_worker=worker))
            try:
                for c in clients: await c.start()
                for i, c in enumerate(clients):
                    t = ticket(0, f'kafka-{i}')
                    await c.publish_media_reference(MediaFrameRef(t.frame, 0, 0, time_ns()))
                    result = await c.submit_cloud_ticket(t)
                    self.assertEqual(result.frame.node_id, f'kafka-{i}')
                    self.assertEqual(result.boxes[0].track_id, 1)
                    f = frame(1, f'kafka-{i}')
                    r = DetectionResult(frame=f, model_mode=Y, model_id='yolo',
                        decision_id='control-v1:test:1', request_id=f'kafka-{i}-yolo', preview_pts=f.source_pts,
                        boxes=Detector().infer(None)[0], inference_ms=1, produced_utc_ns=time_ns(),
                        attributes={'deadline_utc_ns': f.capture_utc_ns + 5_000_000_000})
                    reply = await c.submit_detection(r)
                    self.assertEqual(reply.boxes[0].track_id, 1)
                    validate_result(reply, r)
                await asyncio.sleep(.15)
                for c in clients:
                    self.assertIsNotNone(await c.get_feedback())
                    self.assertFalse(c.raw._pending)
                self.assertTrue(all(not task.done() for task in worker._tasks))
            finally:
                sampler.cancel()
                await asyncio.gather(sampler, return_exceptions=True)
                for c in clients: await c.close()
                await worker.close()
            self.assertFalse(consumers)
