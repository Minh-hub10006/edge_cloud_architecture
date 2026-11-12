import tempfile
from pathlib import Path
import unittest

import numpy as np
from contracts import Box
from scripts.check_rtmdet_snpe_outputs import native_output, matching, raw_metrics


class NativeOutputTests(unittest.TestCase):
    def test_exact_float_shape_and_bad_size(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tensor.raw"
            np.array([1, 2, 3, 4], dtype="<f4").tofile(path)
            np.testing.assert_array_equal(native_output(path, (1, 4)), [[1, 2, 3, 4]])
            with self.assertRaises(ValueError):
                native_output(path, (1, 5))

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tensor.raw"
            for value in (np.nan, np.inf):
                np.array([value], dtype="<f4").tofile(path)
                with self.assertRaises(ValueError):
                    native_output(path, (1,))

    def test_matching_class_geometry_and_unique_assignment(self):
        car = Box(.1, .1, .5, .5, .8, 2, "car")
        other_class = Box(.1, .1, .5, .5, .8, 0, "person")
        far = Box(.6, .6, .9, .9, .8, 2, "car")
        self.assertEqual(matching([car], [other_class, far])["same_class_matches"], 0)
        result = matching([car, car], [car])
        self.assertEqual(result["same_class_matches"], 1)
        self.assertEqual(result["unmatched_reference"], 1)

    def test_raw_errors_do_not_hide_active_queries(self):
        reference = {"boxes": np.zeros((1, 2, 4), dtype=np.float32),
                     "scores": np.zeros((1, 2, 80), dtype=np.float32)}
        reference["scores"][0, 1, 2] = .8
        native = {name: value.copy() for name, value in reference.items()}
        native["boxes"][0, 1] = 5
        result = raw_metrics(reference, native)
        self.assertEqual(result["boxes"]["absolute_error_mean"], 2.5)
        self.assertEqual(result["boxes"]["active_query_absolute_error_mean"], 5)


if __name__ == "__main__":
    unittest.main()
