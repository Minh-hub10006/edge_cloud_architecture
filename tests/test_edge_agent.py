from __future__ import annotations

import asyncio
from dataclasses import replace
import unittest

from contracts import DEFAULT_MODEL_REGISTRY, ModelMode, WindowStateFeedback
from edge_agent import (
    EdgeFanout, SyntheticFrameSource, WindowScheduler, capture_to_fanout,
)


def source(frame_count: int, payload_factory=None) -> SyntheticFrameSource:
    return SyntheticFrameSource(
        node_id="edge-01", camera_id="cam-01", stream_id="main",
        stream_epoch=3, frame_count=frame_count,
        start_utc_ns=1_790_000_000_000_000_000,
        start_mono_ns=10_000_000_000,
        frame_interval_ns=100_000_000,
        payload_factory=payload_factory,
    )


def feedback(stream_key, window_id: int, last_frame_id: int) -> WindowStateFeedback:
    return WindowStateFeedback(
        stream_key=stream_key, window_id=window_id,
        decision_id=":".join(map(str, (*stream_key, window_id))),
        model_id=DEFAULT_MODEL_REGISTRY[ModelMode.YOLO_EDGE].model_id,
        first_frame_id=window_id * 3, last_frame_id=last_frame_id,
        feedback_utc_ns=1_790_000_000_000_000_001,
        blur_difficulty_mean=0.4, calibrated_confidence_mean=0.7,
        box_change_mean=0.2, tracking_stability=0.9,
        cloud_queue_depth=0, cloud_p95_inference_ms=10.0,
        cloud_gpu_utilization_pct=50.0, transport_rtt_ms=5.0,
        drop_ratio=0.0, media_bitrate_mbps=2.0,
    )


class EdgeCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_capture_fans_out_the_same_frame_to_both_branches(self) -> None:
        calls = []
        payloads = []

        def make_payload(frame_id: int):
            calls.append(frame_id)
            payload = object()
            payloads.append(payload)
            return payload

        fanout = EdgeFanout(media_capacity=3, analytics_capacity=3)
        stats = await capture_to_fanout(source(3, make_payload), fanout)

        self.assertEqual(calls, [0, 1, 2])
        self.assertEqual(stats.captured, 3)
        self.assertEqual((stats.media_dropped, stats.analytics_dropped), (0, 0))
        for payload in payloads:
            media = await fanout.media.get()
            analytics = await fanout.analytics.get()
            self.assertIs(media, analytics)
            self.assertIs(media.pixels, payload)
            self.assertEqual(media.identity, analytics.identity)

    async def test_each_queue_drops_its_own_oldest_frame(self) -> None:
        fanout = EdgeFanout(media_capacity=2, analytics_capacity=3)
        stats = await capture_to_fanout(source(5), fanout)

        self.assertEqual((stats.media_dropped, stats.analytics_dropped), (3, 2))
        self.assertEqual(
            [fanout.media.get_nowait().identity.source_frame_id for _ in range(2)],
            [3, 4],
        )
        self.assertEqual(
            [fanout.analytics.get_nowait().identity.source_frame_id for _ in range(3)],
            [2, 3, 4],
        )

    async def test_media_consumer_can_keep_up_when_analytics_lags(self) -> None:
        fanout = EdgeFanout(media_capacity=1, analytics_capacity=1)
        media_seen = []

        async def live_source():
            async for frame in source(4):
                yield frame
                await asyncio.sleep(0)

        async def consume_media():
            while len(media_seen) < 4:
                media_seen.append((await fanout.media.get()).identity.source_frame_id)

        consumer = asyncio.create_task(consume_media())
        stats = await capture_to_fanout(live_source(), fanout)
        await asyncio.wait_for(consumer, timeout=1.0)

        self.assertEqual(media_seen, [0, 1, 2, 3])
        self.assertEqual(stats.media_dropped, 0)
        self.assertEqual(stats.analytics_dropped, 3)


class WindowSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_feedback_from_window_zero_applies_only_at_next_boundary(self) -> None:
        frames = [frame async for frame in source(6)]
        key = frames[0].identity.stream_key
        scheduler = WindowScheduler(
            stream_key=key, window_size=3,
            initial_mode=ModelMode.YOLO_EDGE, target_fps=10.0,
            policy=lambda _feedback, _current: (ModelMode.DFINE_CLOUD, 5.0),
        )

        for frame in frames[:3]:
            result = scheduler.schedule(frame)
            self.assertIsNotNone(result)
            self.assertEqual(result.route.model_mode, ModelMode.YOLO_EDGE)
            self.assertEqual(result.route.window_id, 0)

        next_route = scheduler.accept_feedback(feedback(key, 0, 2))
        self.assertEqual(next_route.effective_from_frame_id, 3)
        self.assertEqual(scheduler.current_route.model_mode, ModelMode.YOLO_EDGE)

        next_frame = scheduler.schedule(frames[3])
        self.assertIsNotNone(next_frame)
        self.assertEqual(next_frame.route.model_mode, ModelMode.DFINE_CLOUD)
        self.assertEqual(next_frame.route.window_id, 1)
        with self.assertRaises(ValueError):
            scheduler.accept_feedback(feedback(key, 0, 2))

    async def test_feedback_cannot_change_route_before_window_finishes(self) -> None:
        frames = [frame async for frame in source(3)]
        scheduler = WindowScheduler(
            stream_key=frames[0].identity.stream_key, window_size=3,
            initial_mode=ModelMode.YOLO_EDGE, target_fps=10.0,
        )
        scheduler.schedule(frames[0])
        with self.assertRaisesRegex(ValueError, "not complete"):
            scheduler.accept_feedback(feedback(frames[0].identity.stream_key, 0, 2))

    async def test_wrong_stream_feedback_is_rejected(self) -> None:
        frame = await anext(source(1).__aiter__())
        scheduler = WindowScheduler(
            stream_key=frame.identity.stream_key, window_size=3,
            initial_mode=ModelMode.YOLO_EDGE, target_fps=10.0,
        )
        scheduler.schedule(frame)
        wrong = replace(feedback(frame.identity.stream_key, 0, 0),
                        stream_key=("edge-02", "cam-01", "main", 3))
        with self.assertRaises(ValueError):
            scheduler.accept_feedback(wrong)

    async def test_feedback_from_another_decision_is_rejected(self) -> None:
        frames = [frame async for frame in source(3)]
        scheduler = WindowScheduler(
            stream_key=frames[0].identity.stream_key, window_size=3,
            initial_mode=ModelMode.YOLO_EDGE, target_fps=10.0,
        )
        for frame in frames:
            scheduler.schedule(frame)
        wrong = replace(feedback(frames[0].identity.stream_key, 0, 2),
                        decision_id="old-decision")
        with self.assertRaisesRegex(ValueError, "decision_id"):
            scheduler.accept_feedback(wrong)

    async def test_missing_feedback_holds_mode_and_fps(self) -> None:
        frames = [frame async for frame in source(4)]
        scheduler = WindowScheduler(
            stream_key=frames[0].identity.stream_key, window_size=3,
            initial_mode=ModelMode.RTMDET_EDGE, target_fps=10.0,
        )
        for frame in frames:
            result = scheduler.schedule(frame)
        self.assertIsNotNone(result)
        self.assertEqual(result.route.window_id, 1)
        self.assertEqual(result.route.model_mode, ModelMode.RTMDET_EDGE)
        self.assertEqual(result.route.target_fps, 10.0)

    async def test_target_fps_throttles_only_analytics_frames(self) -> None:
        frames = [frame async for frame in source(5)]
        fanout = EdgeFanout(media_capacity=5, analytics_capacity=5)
        for frame in frames:
            fanout.publish(frame)
        scheduler = WindowScheduler(
            stream_key=frames[0].identity.stream_key, window_size=10,
            initial_mode=ModelMode.YOLO_EDGE, target_fps=5.0,
        )
        scheduled = []
        for _ in frames:
            selected = scheduler.schedule(fanout.analytics.get_nowait())
            if selected is not None:
                scheduled.append(selected.frame.identity.source_frame_id)
        self.assertEqual(scheduled, [0, 2, 4])
        self.assertEqual(fanout.media.pending, 5)


if __name__ == "__main__":
    unittest.main()
