import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from contracts import Box
from scripts import rtmdet_quant_sweep as sweep
from scripts.package_rtmdet_sweep import build, safe_name


class SweepTests(unittest.TestCase):
    def test_agreement_gate_does_not_reward_more_wrong_boxes(self):
        car = Box(.1, .1, .5, .5, .8, 2, "car")
        wrong = Box(.1, .1, .5, .5, .8, 0, "person")
        self.assertTrue(sweep.smoke_pass(sweep.agreement([car], [car])))
        self.assertFalse(sweep.smoke_pass(sweep.agreement([car], [wrong])))
        self.assertFalse(sweep.smoke_pass(sweep.agreement([car], [car, wrong])))
        self.assertFalse(sweep.smoke_pass(sweep.agreement([car], [])))

    def test_score_error_gate(self):
        ref = Box(.1, .1, .5, .5, .8, 2, "car")
        candidate = Box(.1, .1, .5, .5, .2, 2, "car")
        self.assertFalse(sweep.smoke_pass(sweep.agreement([ref], [candidate])))

    def test_three_frames_required(self):
        self.assertFalse(sweep.outputs_pass([{"smoke_gate": "PASS"}] * 2))
        self.assertFalse(sweep.outputs_pass([{"smoke_gate": "PASS"}] * 2 + [{"smoke_gate": "FAIL"}]))
        self.assertTrue(sweep.outputs_pass([{"smoke_gate": "PASS"}] * 3))

    def test_diagnostic_float_weights_and_cpu_only_never_win(self):
        cases = [{"name": "float", "deployable_int8": False, "npu_gate": "PASS"},
                 {"name": "cpu_only", "deployable_int8": True, "cpu_gate": "PASS", "npu_gate": "NOT_RUN"}]
        self.assertIsNone(sweep.select_winner(cases))
        score = {"reference_coverage": .95, "candidate_agreement": .95, "median_iou": .9, "matched_score_error_p95": .03}
        cases.append({"name": "npu", "deployable_int8": True, "npu_gate": "PASS",
                      "npu_results": [{"agreement": score}] * 3})
        self.assertEqual(sweep.select_winner(cases), "npu")
        self.assertFalse(sweep.CASES[0][2])
        self.assertNotIn("--enable_htp", str(sweep.CASES))

    def test_unknown_prepare_grammar_stops(self):
        self.assertEqual(sweep.prepare_output_arg("--set_output_tensors graphA tensorA1,tensorA2"),
                         "rtmdet_small_fp32 boxes,scores")
        self.assertEqual(sweep.prepare_output_arg("--set_output_tensors comma-separated tensors"), "boxes,scores")
        for text in ("no output selection", "--set_output_tensors unknown syntax"):
            with self.assertRaises(ValueError):
                sweep.prepare_output_arg(text)
        with self.assertRaises(ValueError):
            sweep.prepare_output_arg("--set_output_tensors unknown\n  --htp_socs comma separated targets")

    def test_remote_access_failure_stops_whole_sweep(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sweep.subprocess, "run", return_value=subprocess.CompletedProcess(["ssh"], 255)):
                with self.assertRaises(sweep.AccessBlocked):
                    sweep.run(["ssh", "example"], Path(directory) / "ssh.log", {})
            with patch.object(sweep.subprocess, "run", return_value=subprocess.CompletedProcess(["quantizer"], 1)):
                with self.assertRaises(RuntimeError) as caught:
                    sweep.run(["quantizer"], Path(directory) / "quant.log", {})
                self.assertNotIsInstance(caught.exception, sweep.AccessBlocked)

    def test_checkpoint_missing_or_changed_is_not_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.dlc"
            self.assertFalse(sweep.checkpoint_unchanged({path: "bad"}))
            path.write_bytes(b"baseline")
            value = hashlib.sha256(b"baseline").hexdigest()
            self.assertTrue(sweep.checkpoint_unchanged({path: value}))
            self.assertFalse(sweep.checkpoint_unchanged({path: "bad"}))

    def test_review_excludes_dlc_and_preserves_raw_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("candidate.dlc", "scores.raw", "summary.json", "quant.log", "input_list.txt", "image.png"):
                (folder / name).write_bytes(b"test")
            with zipfile.ZipFile(sweep.pack_review(folder)) as archive:
                self.assertNotIn("candidate.dlc", archive.namelist())
                self.assertIn("scores.raw", archive.namelist())
                self.assertIn("quant.log", archive.namelist())
                self.assertIsNone(archive.testzip())

    def test_remote_script_selects_dsp_both_outputs_without_global_edits(self):
        script = sweep.edge_commands("/home/ubuntu/edge_artifacts/new_run/int8", [
            {"id": f"frame_{n:06d}"} for n in (120, 600, 1200)])
        self.assertEqual(script.count("--use_dsp"), 3)
        self.assertEqual(script.count("--set_output_tensors 'rtmdet_small_fp32 boxes,scores'"), 3)
        self.assertEqual(script.count("sha256sum -c TRANSFER_SHA256SUMS"), 2)
        self.assertIn("/home/ubuntu/sigm/qairt/2.43.0.260128", script)
        for unsafe in ("sudo", "rm ", "pip ", ".bashrc", "--use_cpu", "2.50.0"):
            self.assertNotIn(unsafe, script)
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if bash.is_file():
            subprocess.run([str(bash), "-n"], input=script, text=True, check=True)

    def test_package_path_gate(self):
        self.assertEqual(safe_name("scripts/tool.py"), "scripts/tool.py")
        for name in ("../old.dlc", "/etc/file", "C:/file", "scripts\\file", ""):
            with self.assertRaises(ValueError):
                safe_name(name)

    def test_builder_checksum_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.zip"
            data = b"# test"
            with zipfile.ZipFile(source, "x") as archive:
                archive.writestr("scripts/rtmdet_quant_sweep.py", data)
                archive.writestr("RELEASE_MANIFEST.json", json.dumps({"scripts/rtmdet_quant_sweep.py": hashlib.sha256(data).hexdigest()}))
                archive.writestr("RELEASE_CHECKPOINT_DELTA.json", "{}")
            for folder in ("float_reference_frame120_v2", "float_reference_frame600", "float_reference_frame1200"):
                target = root / folder
                target.mkdir()
                names = ("source.png", "reference_input.npy", "reference_outputs.npz", "rebuild_report.json")
                for name in names:
                    (target / name).write_bytes(data)
                (target / "SHA256SUMS").write_text("".join(f"{hashlib.sha256(data).hexdigest()}  {n}\n" for n in names))
            output = root / "bundle.zip"
            build(source, root, output)
            with zipfile.ZipFile(output) as archive:
                self.assertIn("sweep_references/frame_001200/reference_outputs.npz", archive.namelist())
                self.assertIn("SWEEP_BUNDLE_MANIFEST.json", archive.namelist())
            with self.assertRaises(FileExistsError):
                build(source, root, output)
            (root / "float_reference_frame600" / "source.png").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                build(source, root, root / "bad.zip")

    def test_orchestration_cpu_fail_skips_npu_and_float_control_never_prepares(self):
        """Mock subprocesses, not hardware evidence; exercise full decision/pack path."""
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            work, sdk = root / "work", root / "sdk"
            work.mkdir()
            tool_dir = sdk / "bin/x86_64-linux-clang"
            tool_dir.mkdir(parents=True)
            for name in ("snpe-net-run", "snpe-dlc-quantize", "snpe-dlc-graph-prepare", "snpe-dlc-info"):
                (tool_dir / name).write_bytes(b"tool")
            for name in ("rtmdet_small_fp32.dlc", "rtmdet_small_int8.dlc"):
                (work / name).write_bytes(b"baseline")
            baseline_sha = hashlib.sha256(b"baseline").hexdigest()
            frames = [{"id": f"frame_{n:06d}", "input": __import__("numpy").zeros((1,), dtype="<f4")}
                      for n in (120, 600, 1200)]
            calls = []

            def fake_run(cmd, log, env, **kwargs):
                calls.append(cmd)
                log = Path(log)
                if "--help" in cmd:
                    text = " ".join(f for _, flags, _ in sweep.CASES for f in flags)
                    text += "\n--set_output_tensors graphA tensorA1,tensorA2\n--htp_socs qcs6490"
                elif log.name == "cached_info.txt":
                    text = "Cache Info: HTP_V68_QCS6490_2MB boxes scores"
                else:
                    text = "mock, not hardware evidence"
                log.write_text(text)
                if "--output_dlc" in cmd:
                    Path(cmd[cmd.index("--output_dlc") + 1]).write_bytes(b"candidate")
                if log.name == "edge_run.log":
                    (log.parent / "npu").mkdir()

            def fake_outputs(folder, _, **kwargs):
                if Path(folder).name == "float_baseline":
                    return [{"strict_fp32_parity": {"gate": "PASS"}}] * 3
                failed = "int8_per_channel" == Path(folder).parent.name
                score = {"reference_coverage": .95, "candidate_agreement": .95,
                         "median_iou": .9, "matched_score_error_p95": .03}
                return [{"smoke_gate": "FAIL" if failed else "PASS", "agreement": score}] * 3

            for name, value in (("FLOAT_SHA", baseline_sha), ("OLD_QUANT_SHA", baseline_sha),
                                ("QUANTIZER_SHA", hashlib.sha256(b"tool").hexdigest()),
                                ("__file__", str(root / "scripts/rtmdet_quant_sweep.py"))):
                stack.enter_context(patch.object(sweep, name, value))
            stack.enter_context(patch.object(sweep.sys, "platform", "linux"))
            stack.enter_context(patch.object(sweep.argparse.ArgumentParser, "parse_args",
                                             return_value=argparse.Namespace(work_root=work, sdk_root=sdk,
                                                                             edge_target="ubuntu@192.168.180.246")))
            stack.enter_context(patch.object(sweep, "verify_bundle"))
            stack.enter_context(patch.object(sweep, "load_frames", return_value=frames))
            stack.enter_context(patch.object(sweep, "check_calibration", return_value=work / "input_list.txt"))
            stack.enter_context(patch.object(sweep.os, "access", return_value=True))
            stack.enter_context(patch.object(sweep, "run", side_effect=fake_run))
            stack.enter_context(patch.object(sweep, "inspect_outputs", side_effect=fake_outputs))
            stack.enter_context(patch.object(sweep, "overlays"))
            stack.enter_context(patch.object(sweep.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)))
            self.assertEqual(sweep.main(), 0)
            with zipfile.ZipFile(root / "sweep_review.zip") as archive:
                report = json.loads(archive.read("summary.json"))
            self.assertEqual(report["winner"], "int8_per_channel_enhanced")
            self.assertTrue(report["original_artifacts_unchanged"])
            self.assertFalse(report["production_ready"])
            prepares = [cmd for cmd in calls if "snpe-dlc-graph-prepare" in str(cmd[0]) and "--output_dlc" in cmd]
            self.assertEqual(len(prepares), 1)
            self.assertIn("int8_per_channel_enhanced", str(prepares[0]))
            self.assertEqual(report["cases"][0]["npu_gate"], "NOT_RUN")
            self.assertEqual(report["cases"][1]["npu_gate"], "NOT_RUN")
            with self.assertRaises(SystemExit):
                sweep.main()  # Fixed evidence path is never overwritten.


if __name__ == "__main__":
    unittest.main()
