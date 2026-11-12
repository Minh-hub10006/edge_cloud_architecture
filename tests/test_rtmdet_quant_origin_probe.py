import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts import rtmdet_quant_origin_probe as probe


class OriginProbeTests(unittest.TestCase):
    def test_command_runner_records_elapsed_time_and_log(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = []
            log = Path(directory) / "command.log"
            probe.run([sys.executable, "-c", "print('probe-ok')"], log,
                      os.environ.copy(), commands, timeout=10)
            self.assertIn("probe-ok", log.read_text())
            self.assertEqual(commands[0]["returncode"], 0)
            self.assertGreaterEqual(commands[0]["elapsed_s"], 0)

    def test_command_failure_keeps_diagnostic_record(self):
        with tempfile.TemporaryDirectory() as directory:
            commands = []
            with self.assertRaises(RuntimeError):
                probe.run([sys.executable, "-c", "raise SystemExit(7)"],
                          Path(directory) / "failed.log", os.environ.copy(), commands, timeout=10)
            self.assertEqual(commands[0]["returncode"], 7)
            self.assertGreaterEqual(commands[0]["elapsed_s"], 0)

    def test_encoding_parser_keeps_actual_bits_scale_offset(self):
        text = ("| /stem/Conv_output_0 encoding : bitwidth 16, min -1.2, max 2.3, "
                "scale 5.3e-5, offset -22641.000000 |\n"
                "| input | 1,640,640,3 | uFxp_16 | bitwidth 16, min -2.0, max 2.0, "
                "scale 0.000061036088, offset -32768.0000 |")
        encoded = probe.encodings(text)
        self.assertEqual(encoded["/stem/Conv_output_0"]["bits"], 16)
        self.assertEqual(encoded["input"]["offset"], -32768)
        self.assertAlmostEqual(encoded["/stem/Conv_output_0"]["scale"], 5.3e-5)

    def test_qdq_offset_and_clipping(self):
        value = np.array([-100., -2., 0., 3., 100.], dtype=np.float32)
        actual = probe.qdq(value, {"bits": 8, "scale": .5, "offset": -4})
        np.testing.assert_array_equal(actual, [-2., -2., 0., 3., 100.])

    def test_trace_layout_does_not_guess_ambiguous_shape(self):
        info = "/stem (data type: Float_32; tensor dimension: [1,320,320,16]; tensor type: NATIVE)"
        self.assertEqual(probe.trace_layout(info, "/stem", (1,16,320,320)), ((1,320,320,16), True))
        with self.assertRaises(ValueError):
            probe.trace_layout("/x (data type: Float_32; tensor dimension: [1,80,80,80];", "/x", (1,80,80,80))

    def test_raw_dtype_and_size_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.array([1,2], dtype="<f4").tofile(path / "x.raw")
            np.testing.assert_array_equal(probe.read_raw(path, "/x", (1,2)), [[1,2]])
            with self.assertRaises(ValueError):
                probe.read_raw(path, "x", (4,))

    def test_instrumentation_changed_final_is_rejected(self):
        baseline = {"boxes": np.zeros((1,1,4)), "scores": np.zeros((1,1,1))}
        self.assertTrue(probe.instrumentation_pass(baseline, baseline))
        changed = {**baseline, "scores": np.ones((1,1,1)) * .02}
        self.assertFalse(probe.instrumentation_pass(baseline, changed))

    def test_replay_separates_upstream_error_from_local_error(self):
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        shape = (1,1,2,2)
        graph = helper.make_graph([helper.make_node("Conv", ["input", "weight"], ["out"])], "test",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, shape)],
            [numpy_helper.from_array(np.array([[[[2.]]]], dtype=np.float32), "weight")])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("",17)])
        model.ir_version = 9
        with patch.object(probe, "STEM_SHAPES", {"out": shape}):
            upstream_changed = np.ones(shape, dtype=np.float32) * 3
            replay = probe.replay_stem(model, {"out": upstream_changed * 2}, upstream_changed, {})
            self.assertEqual(replay[0]["best_tested_rel_mae"], 0)
            wrong_local = probe.replay_stem(model, {"out": upstream_changed * 2 + 1}, upstream_changed, {})
            self.assertGreater(wrong_local[0]["best_tested_rel_mae"], .1)


if __name__ == "__main__":
    unittest.main()
