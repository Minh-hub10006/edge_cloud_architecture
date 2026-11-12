import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts import yolo11s_video_trial as trial
from yolo11s_contract import SHAPES, digest


class VideoTrialTests(unittest.TestCase):
    def test_completion_intervals_exclude_warmup_and_reject_coarse_timestamps(self):
        result = trial.completion_timing([0, 1_000_000, 3_000_000, 6_000_000, 10_000_000], 2)
        self.assertEqual(result["interval_ms"]["count"], 3)
        self.assertEqual(result["interval_ms"]["mean"], 3)
        self.assertAlmostEqual(result["fps"], 1000 / 3)
        self.assertIsNone(trial.completion_timing([0, 1, 1], 1)["fps"])

    def exercise_npu(self, invalid=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "input.mp4"
            encoder = trial.writer(video, 30, 64, 48)
            for _ in range(5):
                encoder.write(np.zeros((48, 64, 3), np.uint8))
            encoder.release()
            artifact = root / "artifact"
            artifact.mkdir()
            dlc = artifact / "yolo11s320_w8a8_qcs6490.dlc"
            dlc.write_bytes(b"fake model for mocked SDK test only")
            trial.save_json(artifact / "diagnostic.json", {"diagnostic_only": True,
                            "production_ready": False, "prepared_candidate_sha256": digest(dlc)})
            (artifact / "qcs6490_info.log").write_text("Info of graph: yolo11s320_float\nHTP_V68_QCS6490_2MB")
            sdk = root / "sdk"
            binaries = sdk / "bin/aarch64-oe-linux-gcc11.2"
            binaries.mkdir(parents=True)
            (binaries / "snpe-net-run").write_text("mock")
            def execute(command, **kwargs):
                self.assertIn("--use_dsp", command)
                self.assertNotIn("--enable_cpu_fallback", command)
                output = Path(command[command.index("--output_dir") + 1])
                for index in range(5):
                    result = output / f"Result_{index}"
                    result.mkdir()
                    for name, shape in SHAPES.items():
                        values = np.zeros(shape, np.float32)
                        if invalid and name == "scores":
                            values[0, 0, 0] = 1.3
                        path = result / f"{name}.raw"
                        values.tofile(path)
                        stamp = 1_700_000_000_000_000_000 + index * 10_000_000
                        os.utime(path, ns=(stamp, stamp))
                return SimpleNamespace(returncode=0)
            args = SimpleNamespace(video=video, output_dir=root / "output", nms_iou=.5,
                                   warmup=2, frames=3, local_float_onnx=None, artifact_root=artifact, sdk=sdk)
            with patch.object(trial.subprocess, "run", side_effect=execute) as mocked, \
                    contextlib.redirect_stdout(io.StringIO()):
                code = trial.trial(args)
            self.assertEqual(mocked.call_count, 1)
            report = json.loads((root / "output/report.json").read_text())
            self.assertFalse(report["production_ready"])
            self.assertTrue(report["diagnostic_only"])
            self.assertAlmostEqual(report["snpe_output_completion"]["fps"], 100)
            return code, report

    def test_single_process_npu_video_and_timing(self):
        code, report = self.exercise_npu()
        self.assertEqual(code, 0)
        self.assertEqual(report["rendered_frames"], 3)
        self.assertEqual(report["stages_ms"]["preprocess"]["count"], 3)
        self.assertIsNone(report["stages_ms"]["inference_execute"])

    def test_bad_npu_scores_save_failure_report(self):
        code, report = self.exercise_npu(invalid=True)
        self.assertEqual(code, 2)
        self.assertEqual(report["status"], "VIDEO_TRIAL_FAILED")
        self.assertIn("invalid YOLO output tensor", report["error"])


if __name__ == "__main__":
    unittest.main()
