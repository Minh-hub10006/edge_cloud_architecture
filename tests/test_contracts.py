from __future__ import annotations

import unittest

from contracts import (
    Box,
    CloudInferenceTicket,
    DEFAULT_MODEL_REGISTRY,
    DetectionResult,
    FrameIdentity,
    MediaFrameRef,
    ModelMode,
    PlattConfidenceCalibrator,
    RouteState,
    WindowStateFeedback,
    calibrate_detection_result,
)


class ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = FrameIdentity(
            node_id="edge-01",
            camera_id="cam-01",
            stream_id="main",
            stream_epoch=2,
            source_frame_id=42,
            capture_utc_ns=1_790_000_000_000_000_000,
            source_pts=1_260,
        )

    def test_frame_identity_stream_key_is_stable(self) -> None:
        self.assertEqual(self.frame.stream_key, ("edge-01", "cam-01", "main", 2))

    def test_unsupported_schema_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version"):
            CloudInferenceTicket(
                frame=self.frame, model_mode=ModelMode.DFINE_CLOUD,
                model_id="dfine", decision_id="d-3", preview_pts=1_260,
                deadline_utc_ns=1_790_000_000_100_000_000,
                schema_version=99,
            )

    def test_media_reference_requires_valid_sequence_and_timestamp(self) -> None:
        reference = MediaFrameRef(
            frame=self.frame,
            encoded_seq=7,
            media_pts_ns=466_666_667,
            submitted_utc_ns=1_790_000_000_001_000_000,
        )
        self.assertEqual(reference.encoded_seq, 7)
        with self.assertRaisesRegex(ValueError, "encoded_seq"):
            MediaFrameRef(
                frame=self.frame, encoded_seq=-1, media_pts_ns=0,
                submitted_utc_ns=1,
            )

    def test_cloud_ticket_rejects_edge_mode(self) -> None:
        with self.assertRaises(ValueError):
            CloudInferenceTicket(
                frame=self.frame,
                model_mode=ModelMode.YOLO_EDGE,
                model_id="yolov8n_320_int8",
                decision_id="d-3",
                preview_pts=1_260,
                deadline_utc_ns=1_790_000_000_100_000_000,
            )

    def test_detection_accepts_normalized_box(self) -> None:
        result = DetectionResult(
            frame=self.frame,
            model_mode=ModelMode.YOLO_EDGE,
            model_id="yolov8n_320_int8",
            decision_id="d-3",
            boxes=[Box(0.1, 0.2, 0.3, 0.4, 0.8, 2, "car")],
            inference_ms=5.2,
            produced_utc_ns=1_790_000_000_010_000_000,
        )
        self.assertEqual(result.boxes[0].class_name, "car")

    def test_feedback_is_measurement_not_command(self) -> None:
        feedback = WindowStateFeedback(
            stream_key=self.frame.stream_key,
            window_id=8,
            decision_id="d-8",
            model_id="dfine",
            first_frame_id=20,
            last_frame_id=42,
            feedback_utc_ns=1_790_000_000_020_000_000,
            blur_difficulty_mean=0.4,
            calibrated_confidence_mean=0.7,
            box_change_mean=0.2,
            tracking_stability=0.9,
            cloud_queue_depth=1,
            cloud_p95_inference_ms=14.0,
            cloud_gpu_utilization_pct=60.0,
            transport_rtt_ms=9.0,
            drop_ratio=0.0,
            media_bitrate_mbps=2.0,
        )
        self.assertEqual(feedback.window_id, 8)
        self.assertAlmostEqual(feedback.confidence_difficulty_mean, 0.3)

    def test_calibration_is_explicit_and_preserves_raw_confidence(self) -> None:
        result = DetectionResult(
            frame=self.frame, model_mode=ModelMode.YOLO_EDGE,
            model_id="yolov8n_320_int8", decision_id="d-3",
            boxes=[Box(0.1, 0.2, 0.3, 0.4, 0.8, 2, "car")],
            inference_ms=5.2, produced_utc_ns=1_790_000_000_010_000_000,
        )
        self.assertIsNone(result.boxes[0].calibrated_confidence)
        calibrator = PlattConfidenceCalibrator(
            model_id="yolov8n_320_int8", profile_id="traffic-val-v1",
            slope=0.8, intercept=-0.4,
        )
        calibrated = calibrate_detection_result(result, calibrator)
        self.assertEqual(calibrated.boxes[0].confidence, 0.8)
        self.assertNotEqual(calibrated.boxes[0].calibrated_confidence, 0.8)
        self.assertEqual(calibrated.confidence_calibration_id, "traffic-val-v1")

    def test_calibration_rejects_a_profile_for_another_model(self) -> None:
        result = DetectionResult(
            frame=self.frame, model_mode=ModelMode.YOLO_EDGE,
            model_id="yolov8n_320_int8", decision_id="d-3", boxes=[],
            inference_ms=5.2, produced_utc_ns=1_790_000_000_010_000_000,
        )
        calibrator = PlattConfidenceCalibrator(
            model_id="another-model", profile_id="traffic-val-v1",
            slope=1.0, intercept=0.0,
        )
        with self.assertRaisesRegex(ValueError, "another model_id"):
            calibrate_detection_result(result, calibrator)

    def test_route_state_is_edge_decision_for_next_window(self) -> None:
        state = RouteState(
            stream_key=self.frame.stream_key,
            window_id=9,
            decision_id="d-9",
            model_mode=ModelMode.DFINE_CLOUD,
            target_fps=15.0,
            effective_from_frame_id=43,
            committed_utc_ns=1_790_000_000_030_000_000,
        )
        self.assertEqual(state.model_mode.placement.value, "cloud")

    def test_registry_contains_all_candidate_modes(self) -> None:
        self.assertEqual(set(DEFAULT_MODEL_REGISTRY), set(ModelMode))
        self.assertEqual(len(DEFAULT_MODEL_REGISTRY), 8)  # + rtdetr_split_cloud (path B, 5/11)
        for mode, profile in DEFAULT_MODEL_REGISTRY.items():
            self.assertIs(profile.mode, mode)
            self.assertIs(profile.placement, mode.placement)
        with self.assertRaises(TypeError):
            DEFAULT_MODEL_REGISTRY[ModelMode.YOLO_EDGE] = None


if __name__ == "__main__":
    unittest.main()
