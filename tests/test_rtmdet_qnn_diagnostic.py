from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from inference import rtmdet_qnn_diagnostic as qnn
from scripts import rtmdet_qnn_smoke as cli


def heads():
    return {name: np.full(shape, zero, np.uint8) if channels == 4 else np.zeros(shape, np.uint8)
            for name, (shape, scale, zero) in qnn.ENCODINGS.items() if name != "image"
            for channels in [shape[1]]}


class QuantAndDecoderTests(unittest.TestCase):
    def test_input_zero_point_not_qnn_negative_offset(self):
        tensor = np.zeros((1, 3, 640, 640), np.float32)
        native = qnn.quantize(tensor)
        self.assertTrue((native == 96).all())
        self.assertTrue((qnn.dequantize(native, "image") == 0).all())
        self.assertTrue(native.flags.c_contiguous)

    def test_quantization_clips_rejects_nan_wrong_shape(self):
        tensor = np.full((1, 3, 640, 640), -100, np.float32)
        self.assertEqual(qnn.quantize(tensor).max(), 0)
        tensor[:] = 100
        self.assertEqual(qnn.quantize(tensor).min(), 255)
        for bad in (np.zeros((3, 640, 640)), np.full(tensor.shape, np.nan)):
            with self.assertRaises(ValueError):
                qnn.quantize(bad)
        with self.assertRaises(ValueError):
            qnn.dequantize(np.zeros(tensor.shape, np.float32), "image")

    def test_default_off_preprocessing_and_decoding(self):
        image = np.zeros((360, 640, 3), np.uint8)
        with self.assertRaises(ValueError):
            qnn.prepare_frame(image)
        prepared = qnn.prepare_frame(image, allow_unverified_contract=True)
        with self.assertRaises(ValueError):
            qnn.decode_heads(heads(), prepared)

    def test_owned_bgr_normalization_bottom_padding_and_scale(self):
        image = np.full((720, 1280, 3), [150, 100, 200], np.uint8)
        prepared = qnn.prepare_frame(image, allow_unverified_contract=True)
        original = prepared.native.copy()
        image[:] = 0
        np.testing.assert_array_equal(prepared.native, original)
        self.assertEqual((prepared.scale_x, prepared.scale_y), (.5, .5))
        _, scale, zero = qnn.ENCODINGS["image"]
        desired = (np.array([150, 100, 200]) - [103.53, 116.28, 123.675]) / [57.375, 57.12, 58.395]
        np.testing.assert_array_equal(prepared.native[0, :, 0, 0], np.clip(np.rint(desired / scale) + zero, 0, 255))
        pad = (np.full(3, 114) - [103.53, 116.28, 123.675]) / [57.375, 57.12, 58.395]
        np.testing.assert_array_equal(prepared.native[0, :, 400, 0], np.clip(np.rint(pad / scale) + zero, 0, 255))

    def test_six_heads_geometry_no_double_stride_or_exp(self):
        prepared = qnn.prepare_frame(np.zeros((720, 1280, 3), np.uint8), allow_unverified_contract=True)
        outputs = heads()
        outputs["output_0"][0, 2, 20, 40] = 255  # logit0 -> score0.5, class car
        outputs["output_3"][0, :, 20, 40] = 33
        boxes = qnn.decode_heads(outputs, prepared, allow_unverified_contract=True)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].class_name, "car")
        self.assertEqual(boxes[0].confidence, .5)
        distance = np.float32(33) * np.float32(qnn.ENCODINGS["output_3"][1])
        self.assertAlmostEqual(boxes[0].left, (320 - distance) / 640)
        self.assertAlmostEqual(boxes[0].top, (160 - distance) / 360)
        self.assertEqual(qnn.decode_heads(heads(), prepared, allow_unverified_contract=True), [])

    def test_qai_rgb01_only_rgb_div255_same_geometry_padding(self):
        image = np.full((720, 1280, 3), [150, 100, 200], np.uint8)
        baseline = qnn.prepare_frame(image, allow_unverified_contract=True)
        rgb = qnn.prepare_frame(image, allow_unverified_contract=True, preprocessing="qai_rgb01")
        self.assertEqual((rgb.width, rgb.height, rgb.scale_x, rgb.scale_y),
                         (baseline.width, baseline.height, baseline.scale_x, baseline.scale_y))
        _, scale, zero = qnn.ENCODINGS["image"]
        expected = np.clip(np.rint(np.array([200, 100, 150], np.float32) / 255 / scale) + zero, 0, 255)
        np.testing.assert_array_equal(rgb.native[0, :, 0, 0], expected)
        np.testing.assert_array_equal(rgb.native[0, :, 400, 0], [123, 123, 123])
        self.assertEqual(rgb.clipped_fraction, 0)
        self.assertFalse(np.array_equal(rgb.native, baseline.native))
        self.assertTrue(rgb.native.flags.c_contiguous)
        contract = qnn.preprocessing_contract("qai_rgb01")
        self.assertEqual(contract["color"], "RGB")
        self.assertIsNone(contract["normalize_mean"])
        self.assertEqual(contract["reference_preprocessing_match"], "UNVERIFIED")
        self.assertIn("defaults to M", contract["reference_variant_warning"])

    def test_unknown_profile_and_qai_without_optin_rejected(self):
        image = np.zeros((360, 640, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "unknown"):
            qnn.prepare_frame(image, allow_unverified_contract=True, preprocessing="guess")
        with self.assertRaisesRegex(ValueError, "opt-in"):
            qnn.prepare_frame(image, preprocessing="qai_rgb01")

    def test_wrong_heads_shapes_and_confidence_fail(self):
        prepared = qnn.prepare_frame(np.zeros((360, 640, 3), np.uint8), allow_unverified_contract=True)
        outputs = heads()
        with self.assertRaises(ValueError):
            qnn.decode_heads({k: v for k, v in outputs.items() if k != "output_5"}, prepared,
                             allow_unverified_contract=True)
        outputs["output_1"] = np.zeros((1, 80, 20, 20), np.uint8)
        with self.assertRaises(ValueError):
            qnn.decode_heads(outputs, prepared, allow_unverified_contract=True)
        with self.assertRaises(ValueError):
            qnn.decode_heads(heads(), prepared, confidence=0, allow_unverified_contract=True)


class NativeDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runtime = self.root / "runtime"
        self.sdk = self.runtime / "qairt" / qnn.SDK_VERSION
        for name in ("bin/aarch64-ubuntu-gcc9.4/qnn-net-run", "lib/aarch64-ubuntu-gcc9.4/libQnnHtp.so",
                     "lib/aarch64-ubuntu-gcc9.4/libQnnSystem.so", "lib/aarch64-ubuntu-gcc9.4/libQnnHtpV68Stub.so",
                     "lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so"):
            path = self.sdk / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture-not-a-runtime")
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        for name in qnn.BUNDLE_HASHES:
            (self.bundle / name).write_bytes(b"fixture-not-a-model")
        self.work = self.root / "work"
        self.work.mkdir()
        self.native = np.full((1, 3, 640, 640), 96, np.uint8)

    def write_outputs(self, root, repeats=1, suffix=""):
        for index in range(repeats):
            result = root / f"Result_{index}"
            result.mkdir(parents=True)
            for name, value in heads().items():
                (result / f"{name}{suffix}.raw").write_bytes(value.tobytes())

    def test_qairt250_native_suffix_all_three_repeats_named_and_byte_valid(self):
        output = self.work / "native_output"
        self.write_outputs(output, 3, suffix="_native")
        for index in range(3):
            actual = qnn.read_native_outputs(output, index)
            self.assertEqual(set(actual), set(heads()))
            for name, expected in heads().items():
                np.testing.assert_array_equal(actual[name], expected)
        bad = output / "Result_2/output_4_native.raw"
        bad.write_bytes(b"wrong-byte-count")
        with self.assertRaisesRegex(ValueError, "output_4 native UINT8 byte count"):
            qnn.read_native_outputs(output, 2)

    def test_native_suffix_duplicate_alias_and_unknown_names_rejected(self):
        output = self.work / "native_output"
        self.write_outputs(output, suffix="_native")
        duplicate = output / "Result_0/output_0.raw"
        duplicate.write_bytes(heads()["output_0"].tobytes())
        with self.assertRaisesRegex(ValueError, "ambiguous native output names"):
            qnn.read_native_outputs(output, 0)
        duplicate.unlink()
        (output / "Result_0/unknown_native.raw").write_bytes(b"unexpected")
        with self.assertRaisesRegex(ValueError, "unknown_native.raw"):
            qnn.read_native_outputs(output, 0)

    def test_unknown_bundle_rejected_without_loading_binary(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            qnn.verify_bundle(self.bundle)

    def test_runtime_missing_and_android_only_rejected(self):
        self.assertEqual(qnn.runtime_paths(self.runtime)[0].name, "qnn-net-run")
        (self.sdk / "lib/aarch64-ubuntu-gcc9.4/libQnnHtpV68Stub.so").unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            qnn.runtime_paths(self.runtime)
        with self.assertRaises(ValueError):
            qnn.runtime_paths(self.root)

    def test_native_output_sizes_names_and_unique_result_directory(self):
        output = self.work / "output"
        self.write_outputs(output)
        self.assertEqual(set(qnn.read_native_outputs(output, 0)), set(heads()))
        bad = output / "Result_0/output_0.raw"
        bad.write_bytes(np.zeros(qnn.ENCODINGS["output_0"][0], np.float32).tobytes())
        with self.assertRaisesRegex(ValueError, "byte count"):
            qnn.read_native_outputs(output, 0)
        bad.unlink()
        with self.assertRaisesRegex(ValueError, "names"):
            qnn.read_native_outputs(output, 0)
        (output / "nested/Result_0").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "one Result"):
            qnn.read_native_outputs(output, 0)

    def run_with(self, executor):
        with patch.object(qnn.platform, "system", return_value="Linux"), \
             patch.object(qnn.platform, "machine", return_value="aarch64"), \
             patch.object(qnn, "verify_bundle", return_value=self.bundle), \
             patch.object(qnn.subprocess, "run", side_effect=executor):
            return qnn.run_native(self.bundle, self.runtime, self.work, self.native, 2)

    def test_native_flags_htp_version_and_child_scoped_environment(self):
        previous_env = os.environ.copy()
        calls = []
        def executor(command, **kwargs):
            calls.append((command, kwargs))
            if "--version" in command:
                return SimpleNamespace(stdout=f"qnn-net-run build v{qnn.SDK_VERSION}", stderr="", returncode=0)
            output = Path(command[command.index("--output_dir") + 1])
            self.write_outputs(output, 2, suffix="_native")
            return SimpleNamespace(returncode=0)
        outputs, evidence = self.run_with(executor)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(os.environ.copy(), previous_env)
        cmd, kwargs = calls[-1]
        self.assertIn("--use_native_input_files", cmd)
        self.assertIn("--use_native_output_files", cmd)
        self.assertTrue(cmd[cmd.index("--backend") + 1].endswith("libQnnHtp.so"))
        self.assertTrue(kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(self.sdk / "lib/aarch64-ubuntu-gcc9.4")))
        self.assertEqual((self.work / "input_uint8.raw").stat().st_size, self.native.size)
        self.assertIn(qnn.SDK_VERSION, evidence["runner_version"])
        with self.assertRaises(FileExistsError):
            self.run_with(executor)

    def test_wrong_sdk_exit_failure_and_timeout_propagate(self):
        with self.assertRaisesRegex(RuntimeError, "expected compatible"):
            self.run_with(lambda *a, **k: SimpleNamespace(stdout="v2.45.40", stderr=""))
        def executor(command, **kwargs):
            if "--version" in command:
                return SimpleNamespace(stdout=qnn.SDK_VERSION, stderr="")
            return SimpleNamespace(returncode=16)
        with self.assertRaisesRegex(RuntimeError, "exit=16"):
            self.run_with(executor)
        fresh = self.root / "timeout"
        fresh.mkdir()
        self.work = fresh
        def timeout(command, **kwargs):
            raise subprocess.TimeoutExpired(command, 1)
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_with(timeout)

    def test_host_platform_rejects_no_cpu_fallback(self):
        with patch.object(qnn.platform, "system", return_value="Windows"), self.assertRaisesRegex(RuntimeError, "Linux ARM64"):
            qnn.run_native(self.bundle, self.runtime, self.work, self.native)

    def test_cli_report_does_not_admit_benchmark_or_quality(self):
        args = SimpleNamespace(allow_unverified_contract=True, repeats=2, frame_index=0,
            confidence=.1, artifact_dir=self.bundle, video=self.root / "traffic.mp4",
            runtime_root=self.runtime, output_parent=self.root, timeout_s=120)
        capture = SimpleNamespace(read=lambda: (True, np.zeros((360, 640, 3), np.uint8)), release=lambda: None)
        with patch.object(cli, "verify_bundle", return_value=self.bundle), \
             patch("cv2.VideoCapture", return_value=capture), \
             patch.object(cli, "run_native", return_value=([heads(), heads()], {"fixture": True})):
            self.assertEqual(cli.smoke(args), 0)
        report_path = next(self.root.glob("rtmdet_qnn_real_*/report.json"))
        report = json.loads(report_path.read_text())
        self.assertFalse(report["pipeline_admitted"])
        self.assertFalse(report["benchmark"])
        self.assertIsNone(report["fps"])
        self.assertEqual(report["contract_gate"], "UNVERIFIED_PREPROCESS_AND_HEAD_SEMANTICS")
        self.assertEqual(report["quality_status"], "NOT_EVALUATED")
        self.assertTrue(Path(report["overlay"]).is_file())
        args.allow_unverified_contract = False
        with self.assertRaises(ValueError):
            cli.smoke(args)

    def ab_args(self):
        return SimpleNamespace(allow_unverified_contract=True, repeats=3, frame_index=0,
            confidence=.1, artifact_dir=self.bundle, video=self.root / "traffic.mp4",
            runtime_root=self.runtime, output_parent=self.root, timeout_s=120,
            preprocessing="mmdet_bgr_normalized", compare_preprocessing=True)

    def ab_runtime(self):
        return {"runner_version": qnn.SDK_VERSION, "sdk_version": qnn.SDK_VERSION, "runner_sha256": "fixture-runner",
                "htp_sha256": "fixture-htp"}

    def test_ab_same_decoded_frame_two_native_inputs_no_winner_or_admission(self):
        args = self.ab_args()
        image = np.full((360, 640, 3), [150, 100, 200], np.uint8)
        capture = SimpleNamespace(read=lambda: (True, image), release=lambda: None)
        native_inputs = []
        def runner(folder, runtime_root, work, native, repeats, timeout_s):
            native_inputs.append(native.copy())
            outputs = heads()
            if len(native_inputs) == 2:
                outputs["output_0"][0, 2, 20, 40] = 255
                outputs["output_3"][0, :, 20, 40] = 33
            runtime = self.ab_runtime()
            runtime["runner_version"] = f"qnn-net-run pid:{100 + len(native_inputs)}\nQNN SDK v{qnn.SDK_VERSION}"
            return [outputs] * repeats, runtime
        with patch.object(cli, "verify_bundle", return_value=self.bundle), \
             patch("cv2.VideoCapture", return_value=capture) as video_mock, \
             patch.object(cli, "run_native", side_effect=runner):
            self.assertEqual(cli.compare_preprocessing(args), 0)
        self.assertEqual(video_mock.call_count, 1)
        self.assertEqual(args.preprocessing, "mmdet_bgr_normalized")
        self.assertEqual(len(native_inputs), 2)
        self.assertFalse(np.array_equal(*native_inputs))
        summary_path = next(self.root.glob("rtmdet_qnn_ab_*/ab_report.json"))
        summary = json.loads(summary_path.read_text())
        self.assertTrue(summary["controls_verified"])
        self.assertIsNone(summary["winner"])
        self.assertFalse(summary["pipeline_admitted"])
        self.assertFalse(summary["benchmark"])
        self.assertEqual([case["detections"] for case in summary["cases"]], [0, 1])
        self.assertNotEqual(*[case["native_input_sha256"] for case in summary["cases"]])
        case_reports = [json.loads(Path(case["report"]).read_text()) for case in summary["cases"]]
        self.assertEqual(case_reports[0]["source"], case_reports[1]["source"])
        self.assertTrue(all(report["repeats_same_frame"] == 3 for report in case_reports))
        self.assertEqual(case_reports[1]["experimental_contract"]["color"], "RGB")
        self.assertIsNone(case_reports[1]["experimental_contract"]["normalize_mean"])
        self.assertEqual(case_reports[0]["experimental_contract"]["color"], "BGR")

    def test_ab_runtime_mismatch_fails_without_summary_or_auto_selection(self):
        args = self.ab_args()
        capture = SimpleNamespace(read=lambda: (True, np.zeros((360, 640, 3), np.uint8)), release=lambda: None)
        first, second = self.ab_runtime(), self.ab_runtime()
        second["htp_sha256"] = "different-htp"
        with patch.object(cli, "verify_bundle", return_value=self.bundle), \
             patch("cv2.VideoCapture", return_value=capture), \
             patch.object(cli, "run_native", side_effect=[([heads()]*3, first), ([heads()]*3, second)]), \
             self.assertRaisesRegex(ValueError, "runtime control mismatch"):
            cli.compare_preprocessing(args)
        self.assertFalse(list(self.root.glob("rtmdet_qnn_ab_*/ab_report.json")))

    def test_ab_native_failure_propagates_no_cpu_fallback_or_summary(self):
        args = self.ab_args()
        capture = SimpleNamespace(read=lambda: (True, np.zeros((360, 640, 3), np.uint8)), release=lambda: None)
        with patch.object(cli, "verify_bundle", return_value=self.bundle), \
             patch("cv2.VideoCapture", return_value=capture), \
             patch.object(cli, "run_native", side_effect=RuntimeError("HTP failed")) as native_mock, \
             self.assertRaisesRegex(RuntimeError, "HTP failed"):
            cli.compare_preprocessing(args)
        self.assertEqual(native_mock.call_count, 1)
        self.assertFalse(list(self.root.glob("rtmdet_qnn_ab_*/ab_report.json")))

    def test_ab_without_optin_fails_before_video_read_or_hardware(self):
        args = self.ab_args()
        args.allow_unverified_contract = False
        with patch.object(cli, "load_frame") as video_mock, patch.object(cli, "run_native") as native_mock, \
             self.assertRaisesRegex(ValueError, "allow-unverified-contract"):
            cli.compare_preprocessing(args)
        video_mock.assert_not_called()
        native_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
