from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import candidate_artifact as artifact_tools
with patch.dict(sys.modules, {"candidate_artifact": artifact_tools}):
    from scripts.export_cloud_candidate import command


class ExportCommandTests(unittest.TestCase):
    def test_rt_small_exact_sp2_config_and_upstream_cli(self):
        args, cwd = command("rtdetrv2_small", Path("upstream"), Path("checkpoint.pth"), Path("new.onnx"))
        self.assertEqual(cwd, Path("upstream/rtdetrv2_pytorch"))
        self.assertEqual(args[args.index("-s") + 1], "640")
        self.assertIn("configs/rtdetrv2/rtdetrv2_r18vd_sp2_120e_coco.yml", args)
        self.assertIn("--check", args)
        with self.assertRaises(ValueError):
            command("wrong", Path("upstream"), Path("checkpoint.pth"), Path("new.onnx"))

    def test_lw_large_architecture_not_small_defaults(self):
        args, cwd = command("lwdetr_large", Path("upstream"), Path("checkpoint.pth"), Path("new/inference_model.onnx"))
        self.assertEqual(cwd, Path("upstream"))
        self.assertEqual(args[args.index("--hidden_dim") + 1], "384")
        self.assertEqual(args[args.index("--ca_nheads") + 1], "24")
        self.assertEqual(args[args.index("--vit_encoder_num_layers") + 1], "10")
        self.assertEqual(args[-1], "export_model")


class SealGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "weights.pth"
        self.checkpoint.write_bytes(b"test-checkpoint")
        self.artifact = self.root / "candidate.onnx"
        self.artifact.write_bytes(b"test-model")
        config = self.root / artifact_tools.SOURCES["rtmdet_small"][1]
        config.parent.mkdir(parents=True)
        config.write_text("test config", encoding="utf-8")
        (self.root / "LICENSE").write_text("test license", encoding="utf-8")

    def seal(self, path=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return artifact_tools.seal(path or self.artifact, "rtmdet_small", repo=self.root,
                                       checkpoint=self.checkpoint, **kwargs)

    def test_dirty_checkout_rejected_before_onnx_probe(self):
        with patch.object(artifact_tools.subprocess, "check_output", side_effect=["revision", " M config"]), \
                patch.object(artifact_tools, "probe_onnx") as probe:
            with self.assertRaisesRegex(ValueError, "clean"):
                self.seal()
            probe.assert_not_called()

    def test_manifest_sealed_once_hardware_stays_pending(self):
        shapes = {"boxes": [1, 8400, 4], "scores": [1, 8400, 80]}
        with patch.object(artifact_tools.subprocess, "check_output", side_effect=["revision", ""] * 2), \
                patch.object(artifact_tools, "probe_onnx", return_value=shapes):
            data = self.seal()
            self.assertFalse(data["hardware_validated"])
            self.assertEqual(data["quality_status"], "NOT_EVALUATED")
            self.assertEqual(data["sha256"], artifact_tools.sha256(self.artifact))
            with self.assertRaises(FileExistsError):
                self.seal()

    def test_dlc_cannot_bind_to_other_checkpoint_reference(self):
        dlc = self.root / "candidate.dlc"
        dlc.write_bytes(b"test-dlc")
        sidecar = {"kind": "rtmdet_small", "model_id": artifact_tools.MODEL_IDS["rtmdet_small"],
                   "sha256": artifact_tools.sha256(self.artifact), "source_revision": "revision",
                   "checkpoint_sha256": "different-checkpoint"}
        Path(str(self.artifact) + ".json").write_text(json.dumps(sidecar), encoding="utf-8")
        with patch.object(artifact_tools.subprocess, "check_output", side_effect=["revision", ""]), \
                patch.object(artifact_tools, "probe_onnx") as probe:
            with self.assertRaisesRegex(ValueError, "provenance"):
                self.seal(dlc, reference_onnx=self.artifact, layout="nhwc")
            probe.assert_not_called()
