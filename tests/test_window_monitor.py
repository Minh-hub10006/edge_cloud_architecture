from __future__ import annotations

import unittest

from edge_agent.window_monitor import (
    StateObservation, StatePrometheusExporter, WindowStateAggregator,
)


KEY = ("edge-a", "cam-1", "preview", 4)


def observation(frame_id: int, **kwargs) -> StateObservation:
    defaults = dict(
        stream_key=KEY, window_id=2, frame_id=frame_id,
        monotonic_ns=frame_id * 100_000_000,
        media_matched=True, transport_rtt_ms=10.0,
        edge_backlog_norm=0.25, cloud_inflight_norm=0.5,
        edge_inference_ms=8.0, cloud_e2e_ms=30.0,
        blur_difficulty=0.4, calibrated_confidences=(0.8, 0.6),
        box_change=0.2, completed=True, model_id="test-model", transport="grpc",
    )
    defaults.update(kwargs)
    return StateObservation(**defaults)


class WindowStateAggregatorTests(unittest.TestCase):
    def test_aggregates_ten_features_and_calibrated_confidence_per_detection(self) -> None:
        collector = WindowStateAggregator()
        collector.record(observation(11, media_matched=True, transport_rtt_ms=10.0))
        collector.record(observation(10, media_matched=False, transport_rtt_ms=20.0,
                                     calibrated_confidences=(0.2,)))
        state = collector.finish(KEY, 2)

        self.assertEqual(state.first_frame_id, 10)
        self.assertEqual(state.last_frame_id, 11)
        self.assertEqual(state.media_match_rate, 0.5)
        self.assertEqual(state.transport_rtt_ms, 15.0)
        self.assertAlmostEqual(state.calibrated_confidence_mean, 1.6 / 3)
        self.assertEqual(state.effective_fps, 10.0)
        self.assertEqual(state.missing_features, ())
        self.assertEqual(state.sample_counts["calibrated_confidence_mean"], 3)

    def test_unknown_calibration_and_unobserved_media_match_stay_missing(self) -> None:
        collector = WindowStateAggregator()
        collector.record(observation(
            0, media_matched=None, calibrated_confidences=(),
            edge_inference_ms=None, cloud_e2e_ms=None, box_change=None,
        ))
        state = collector.finish(KEY, 2)

        self.assertIsNone(state.media_match_rate)
        self.assertIsNone(state.calibrated_confidence_mean)
        self.assertIn("media_match_rate", state.missing_features)
        self.assertIn("calibrated_confidence_mean", state.missing_features)
        self.assertIn("edge_inference_ms", state.missing_features)

    def test_duplicate_frame_is_rejected_and_counted(self) -> None:
        collector = WindowStateAggregator()
        self.assertTrue(collector.record(observation(1)))
        self.assertFalse(collector.record(observation(1)))
        self.assertEqual(collector.duplicate_observations, 1)

    def test_invalid_values_and_window_capacity_are_visible(self) -> None:
        with self.assertRaisesRegex(ValueError, "edge_backlog_norm"):
            observation(1, edge_backlog_norm=1.2)
        collector = WindowStateAggregator(max_observations_per_window=1)
        collector.record(observation(1))
        self.assertFalse(collector.record(observation(2)))
        self.assertEqual(collector.capacity_drops, 1)

    def test_finished_windows_are_bounded_and_not_reopened(self) -> None:
        collector = WindowStateAggregator(max_windows=1)
        first = observation(0)
        collector.record(first)
        collector.finish(KEY, 2)
        self.assertFalse(collector.record(observation(1)))
        collector.record(observation(30, window_id=3))
        collector.finish(KEY, 3)
        self.assertEqual(len(collector.recent()), 1)
        self.assertEqual(collector.recent()[0].window_id, 3)

    def test_pending_windows_are_bounded_per_stream(self) -> None:
        collector = WindowStateAggregator(max_windows=1)
        collector.record(observation(1, window_id=1))
        collector.record(observation(31, window_id=2))
        self.assertEqual(collector.pending_window_drops, 1)
        self.assertEqual(collector.capacity_drops, 1)
        with self.assertRaisesRegex(ValueError, "no observations"):
            collector.finish(KEY, 1)
        self.assertEqual(collector.finish(KEY, 2).window_id, 2)

    def test_prometheus_marks_missing_values_unavailable_without_zero_fill(self) -> None:
        from prometheus_client import generate_latest

        collector = WindowStateAggregator()
        collector.record(observation(
            0, media_matched=None, calibrated_confidences=(),
        ))
        state = collector.finish(KEY, 2)
        exporter = StatePrometheusExporter(
            node=KEY[0], camera=KEY[1], stream=KEY[2], epoch=KEY[3],
            model="test-model", transport="grpc",
        )
        exporter.publish(state, collector)
        payload = generate_latest(exporter.registry).decode("utf-8")
        self.assertIn('edge_state_feature_available{camera="cam-1",epoch="4",feature="media_match_rate"', payload)
        self.assertIn('feature="media_match_rate",model="test-model",node="edge-a",stream="preview",transport="grpc"} 0.0', payload)
        self.assertNotIn('edge_state_media_match_rate{', payload)


if __name__ == "__main__":
    unittest.main()
