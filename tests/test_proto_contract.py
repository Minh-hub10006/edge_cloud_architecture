from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from grpc_tools import protoc
from contracts.proto import edge_video_platform_pb2 as module


class ProtobufContractTests(unittest.TestCase):
    def test_proto_compiles_and_round_trips_frame_identity(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        proto_root = project_root / "contracts" / "proto"
        source = proto_root / "edge_video_platform.proto"

        with tempfile.TemporaryDirectory() as generated_dir:
            status = protoc.main([
                "grpc_tools.protoc",
                f"-I{proto_root}",
                f"--python_out={generated_dir}",
                f"--grpc_python_out={generated_dir}",
                str(source),
            ])
            self.assertEqual(status, 0)

            generated = Path(generated_dir) / "edge_video_platform_pb2.py"
            generated_grpc = Path(generated_dir) / "edge_video_platform_pb2_grpc.py"
            self.assertTrue(generated_grpc.is_file())
            original = module.FrameIdentity(
                node_id="edge-01",
                camera_id="cam-01",
                stream_id="main",
                stream_epoch=2,
                source_frame_id=42,
                capture_utc_ns=1_790_000_000_000_000_000,
                source_pts=1_260,
            )
            restored = module.FrameIdentity.FromString(original.SerializeToString())
            self.assertEqual(restored, original)

            reference = module.MediaFrameRef(
                schema_version=2,
                frame=original,
                encoded_seq=7,
                media_pts_ns=466_666_667,
                submitted_utc_ns=1_790_000_000_001_000_000,
            )
            self.assertEqual(module.MediaFrameRef.FromString(reference.SerializeToString()), reference)

            feedback = module.WindowStateFeedback(
                schema_version=2,
                blur_difficulty_mean=0.4,
                calibrated_confidence_mean=0.7,
                box_change_mean=0.2,
            )
            restored_feedback = module.WindowStateFeedback.FromString(feedback.SerializeToString())
            self.assertEqual(restored_feedback.blur_difficulty_mean, 0.4)
            self.assertEqual(restored_feedback.calibrated_confidence_mean, 0.7)

            service = module.DESCRIPTOR.services_by_name["EdgeCloudDataPlane"]
            self.assertEqual(
                [method.name for method in service.methods],
                ["PublishMediaFrameRef", "SubmitDetectionResult", "SubmitCloudInference"],
            )


if __name__ == "__main__":
    unittest.main()
