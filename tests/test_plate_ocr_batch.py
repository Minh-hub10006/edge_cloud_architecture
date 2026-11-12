import unittest

import numpy as np


class PlateOcrBatchTests(unittest.TestCase):
    def test_greedy_ctc_drops_repeats_and_blanks(self):
        from inference.plate_ocr import CHARS
        from inference.plate_ocr_batch import decode
        idx = [0, 3, 3, 0, 3, 11, 11, 0]                 # "2", "2" again after a blank, then "A"
        logits = np.full((len(idx), len(CHARS) + 1), -9.0, np.float32)
        logits[np.arange(len(idx)), idx] = 9.0
        text, conf = decode(logits)
        self.assertEqual(text, "22A")
        self.assertGreater(conf, .99)

    def test_line_width_is_padded_to_a_multiple_of_64_with_the_last_column(self):
        from inference.plate_ocr_batch import prepare
        gray = np.tile(np.linspace(0, 255, 150, dtype=np.uint8), (40, 1))   # ratio 3.75 -> width 4 x 64
        x = prepare(gray)
        self.assertEqual(x.shape, (1, 64, 256))
        resized_w = int(np.ceil(64 * int(64 * 150 / 40) / 64))
        self.assertTrue(np.all(x[0, :, resized_w:] == x[0, :, resized_w - 1:resized_w]))
        self.assertLessEqual(float(x.max()), 1.0)
        self.assertGreaterEqual(float(x.min()), -1.0)

    def test_one_call_per_width_and_texts_back_in_order(self):
        from inference.plate_ocr import CHARS
        from inference.plate_ocr_batch import BatchPlateReader
        calls = []

        def infer(x):   # every line reads "7" with full confidence
            calls.append(x.shape)
            out = np.full((len(x), x.shape[-1] // 4 - 1, len(CHARS) + 1), -9.0, np.float32)
            out[:, 0, CHARS.index("7") + 1] = 9.0
            out[:, 1:, 0] = 9.0
            return out
        r = BatchPlateReader(infer)
        lines = [np.zeros((20, 60), np.uint8), np.zeros((20, 200), np.uint8), np.zeros((20, 58), np.uint8)]
        self.assertEqual(r.recognize(lines), ["7", "7", "7"])
        self.assertEqual(sorted(s[-1] for s in calls), [192, 640])   # widths 192 (two lines) and 640: two calls
        self.assertEqual(sorted(s[0] for s in calls), [1, 2])


if __name__ == "__main__":
    unittest.main()
