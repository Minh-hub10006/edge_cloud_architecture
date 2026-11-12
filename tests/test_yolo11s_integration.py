from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from contracts import ModelMode, Placement, DetectionResult, FrameIdentity
from contracts.grpc_codec import result_to_proto, result_from_proto
from contracts.proto import edge_video_platform_pb2 as pb
from inference.catalog import phase7_edge_catalog
from inference.yolo11s_snpe import Yolo11sSnpeAdapter, DLC_SHA256, MODEL_ID
from yolo11s_contract import SHAPES, prepare, decode


class Yolo11IntegrationTests(unittest.TestCase):
    def make_adapter(self, path, session):
        with patch("inference.yolo11s_snpe.digest", return_value=DLC_SHA256), patch(
                "legacy.adapters.npu.snpe_dlc_session.SnpeDlcSession", return_value=session):
            return Yolo11sSnpeAdapter(path, allow_diagnostic=True)

    def session(self):
        session = MagicMock(runtime="dsp", mode="snpe_daemon")
        outputs = {k: np.zeros(s, np.float32) for k, s in SHAPES.items()}
        outputs["boxes"][0, 0] = [60, 110, 180, 170]
        outputs["scores"][0, 0, 2] = .9
        session.infer_raw.return_value = (.001, {k: v.ravel() for k, v in outputs.items()})
        return session, outputs

    def test_wire_placement_and_roundtrip_preserve_candidate(self):
        self.assertEqual(pb.YOLO11S_EDGE, 6)
        self.assertEqual((pb.YOLO_EDGE, pb.RTMDET_EDGE, pb.LWDETR_CLOUD), (1, 2, 5))
        self.assertIs(ModelMode.YOLO11S_EDGE.placement, Placement.EDGE)
        frame = FrameIdentity("e", "c", "s", 1, 0, 1, 0)
        result = DetectionResult(frame, ModelMode.YOLO11S_EDGE, MODEL_ID, "d", [], 1, 2,
                                 attributes={"diagnostic_only": True, "parity_gate": "FAIL"})
        restored = result_from_proto(result_to_proto(result))
        self.assertEqual(restored.model_mode, result.model_mode)
        self.assertEqual(restored.attributes, result.attributes)

    def test_catalog_is_lazy_and_baseline_unchanged(self):
        self.assertEqual(phase7_edge_catalog().modes(), (ModelMode.YOLO_EDGE,))
        cat = phase7_edge_catalog(yolo11s_dlc="not-loaded-yet")
        self.assertEqual(cat.modes(), (ModelMode.YOLO_EDGE, ModelMode.YOLO11S_EDGE))
        with self.assertRaisesRegex(ValueError, "opt-in"):
            cat.load(ModelMode.YOLO11S_EDGE)

    def test_hash_and_optin_before_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            artifact = Path(folder) / "wrong.dlc"
            artifact.write_bytes(b"not-the-pinned-model")
            with patch("legacy.adapters.npu.snpe_dlc_session.SnpeDlcSession") as session:
                with self.assertRaisesRegex(ValueError, "opt-in"):
                    Yolo11sSnpeAdapter(artifact)
                with self.assertRaisesRegex(ValueError, "checksum"):
                    Yolo11sSnpeAdapter(artifact, allow_diagnostic=True)
                session.assert_not_called()

    def test_non_square_prepare_and_decode_matches_standalone(self):
        with tempfile.NamedTemporaryFile(suffix=".dlc") as file:
            session, outputs = self.session()
            adapter = self.make_adapter(file.name, session)
            image = np.zeros((180, 320, 3), np.uint8)
            image[:, :, 2] = 255
            prepared = adapter.prepare_frame(image)
            self.assertEqual(prepared.tensor.shape, (1, 320, 320, 3))
            self.assertEqual(prepared.tensor.dtype, np.float32)
            self.assertTrue(prepared.tensor.flags.c_contiguous)
            oracle = prepare(image)
            np.testing.assert_array_equal(prepared.tensor, oracle.tensor.transpose(0, 2, 3, 1))
            boxes, _, stages = adapter.infer_prepared(prepared)
            expected = decode(outputs, oracle, nms_iou=.5)
            self.assertEqual(boxes[0].left, expected[0].left)
            self.assertEqual(boxes[0].top, expected[0].top)
            self.assertEqual(boxes[0].class_name, "car")
            self.assertIn("edge_session_call_ms", stages)
            self.assertFalse(adapter.settings["production_ready"])

    def test_invalid_output_not_clipped_or_silently_reshaped(self):
        with tempfile.NamedTemporaryFile(suffix=".dlc") as file:
            session, outputs = self.session()
            adapter = self.make_adapter(file.name, session)
            prepared = adapter.prepare_frame(np.zeros((320, 320, 3), np.uint8))
            for invalid in (np.full(SHAPES["scores"], 1.3, np.float32),
                            np.zeros((1, 80, 2100), np.float32),
                            np.zeros(SHAPES["scores"], np.float64)):
                session.infer_raw.return_value = (.001, {"boxes": outputs["boxes"], "scores": invalid})
                with self.assertRaises(ValueError):
                    adapter.infer_prepared(prepared)

    def test_strict_session_no_subprocess_fallback(self):
        from legacy.adapters.npu.snpe_dlc_session import SnpeDlcSession
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.dlc"
            path.write_bytes(b"fake")
            with patch("legacy.adapters.npu.snpe_daemon_manager.try_get_daemon_client", return_value=None), patch(
                    "legacy.adapters.npu.snpe_dlc_session._snpe_bin") as binary:
                with self.assertRaisesRegex(RuntimeError, "no fallback"):
                    SnpeDlcSession(str(path), require_daemon=True, output_tensors=("boxes", "scores"),
                                   input_bytes=320*320*3*4)
                binary.assert_not_called()

    def test_daemon_contract_is_separate_from_baseline(self):
        from legacy.adapters.npu import snpe_daemon_manager as manager
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {"SNPE_DAEMON_SOCK_DIR": folder}):
            self.assertNotEqual(manager.socket_path_for_dlc("x"),
                                manager.socket_path_for_dlc("x", namespace="explicit-outputs"))
            with self.assertRaises(ValueError):
                manager.get_daemon_handle("x", output_tensors=("boxes", "boxes"), input_bytes=12)

    def test_wrong_runtime_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".dlc") as file:
            session, _ = self.session()
            session.runtime = "cpu"
            with self.assertRaisesRegex(RuntimeError, "no CPU"):
                self.make_adapter(file.name, session)
            session.close.assert_called_once()

    def test_explicit_daemon_outputs_override_ambient_yolo_settings(self):
        from legacy.adapters.npu.snpe_daemon_manager import SnpeDaemonHandle
        with tempfile.TemporaryDirectory() as folder, patch.dict("os.environ", {
                "SNPE_DAEMON_SOCK_DIR": folder, "SNPE_OUTPUT_TENSORS": "legacy_output"}), patch(
                "legacy.adapters.npu.snpe_daemon_manager.SnpeDaemonClient") as client, patch(
                "legacy.adapters.npu.snpe_daemon_manager.subprocess.Popen") as process:
            client.return_value.ping.return_value = True
            handle = SnpeDaemonHandle("model.dlc", socket_path=str(Path(folder) / "candidate.sock"),
                runtime="dsp", perf_profile="burst", bin_path="daemon", output_tensors=("boxes", "scores"),
                input_bytes=320*320*3*4, namespace="new-contract")
            argv = process.call_args.args[0]
            self.assertEqual(argv[argv.index("--outputs") + 1], "boxes,scores")
            self.assertEqual(handle.client, client.return_value)
            self.assertEqual(handle.input_bytes, 1228800)
            self.assertIn('--init-cache', argv)
            self.assertNotIn('--no-init-cache', argv)


class Yolo11SharedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_result_uses_shared_hub_tracking_without_cuda_adapter(self):
        from dataclasses import replace
        from time import time_ns
        from cloud_core import CloudFrameCache, ResultHub
        from cloud_core.grpc_service import CloudDataPlaneServicer
        from contracts import Box
        class Tracker:
            def update(self, result):
                return replace(result, boxes=[replace(b, track_id=42) for b in result.boxes])
            def reset(self, key):
                pass
        service = CloudDataPlaneServicer(hub=ResultHub(tracker=Tracker()), frame_cache=CloudFrameCache(),
                                        frame_taps=object(), cloud_adapters={})
        result = DetectionResult(FrameIdentity("edge", "cam", "y11", 1, 0, time_ns(), 0),
            ModelMode.YOLO11S_EDGE, MODEL_ID, "manual", [Box(.1,.2,.4,.5,.9,2,"car")], 1, time_ns(),
            attributes={"diagnostic_only": True, "production_ready": False, "parity_gate": "FAIL"})
        transported = result_from_proto(pb.DetectionResult.FromString(result_to_proto(result).SerializeToString()))
        status, _, tracked = await service.handle_detection_result(transported)
        self.assertEqual(status, "accepted")
        self.assertEqual(tracked.boxes[0].track_id, 42)
        self.assertEqual(tracked.model_id, MODEL_ID)
        self.assertEqual(tracked.attributes["parity_gate"], "FAIL")
