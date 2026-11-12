"""Vietnamese plate reading on the Cloud: the team's EasyOCR recognizer (VGG + BiLSTM + CTC, no CRAFT detector).

Ported from LinhDo2208/Vietnamese_lisence_plate_EasyOCR plate_reader.py (same team): 2-line plates (w/h < 2.5) are
cut in half and each line read alone; every line is read `passes` times with growing margin trims and the result
matching the VN format wins. Install the weights once with that repo's setup_model.py --home <model_home>.
"""
from __future__ import annotations

import os
import re
from collections import Counter

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TOP_LINE = re.compile(r"^\d{2}[A-Z][A-Z0-9]?$")  # 59G1
BOTTOM_LINE = re.compile(r"^\d{4,5}$")  # 63188
VN_PLATE = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{4,5}$")  # one line, or both lines joined
MARGINS = ((0.0, 0.0), (0.04, 0.04), (0.08, 0.06), (0.12, 0.08))  # (x, y) trim per side


def valid_plate(text: str) -> bool:
    return bool(VN_PLATE.match(text))


class PlateReader:
    def __init__(self, model_home: str, *, passes: int = 3, gpu: bool = True, name: str = "plate") -> None:
        import easyocr
        import torch
        import yaml
        with open(os.path.join(model_home, "user_network", f"{name}.yaml"), encoding="utf-8") as fh:
            self.chars = str(yaml.safe_load(fh)["character_list"])
        # quantize=False: EasyOCR's CPU int8 quantization drops LSTM.flatten_parameters and the custom net crashes.
        self.reader = easyocr.Reader(["en"], gpu=gpu and torch.cuda.is_available(), recog_network=name,
                                     detector=False, quantize=False, verbose=False,
                                     model_storage_directory=os.path.join(model_home, "model"),
                                     user_network_directory=os.path.join(model_home, "user_network"))
        self.margins = MARGINS[:max(1, min(passes, len(MARGINS)))]
        self.calls = 0

    def _once(self, gray, mx, my) -> str:
        h, w = gray.shape[:2]
        crop = gray[int(h * my): max(int(h * (1 - my)), int(h * my) + 4),
                    int(w * mx): max(int(w * (1 - mx)), int(w * mx) + 4)]
        ch, cw = crop.shape[:2]
        self.calls += 1
        out = self.reader.recognize(crop, horizontal_list=[[0, cw, 0, ch]], free_list=[], allowlist=self.chars)
        return "".join(c for c in "".join(t for _, t, _ in out).upper() if c in self.chars)

    def _line(self, gray, pattern) -> str:
        reads = [self._once(gray, mx, my) for mx, my in self.margins]
        pool = [r for r in reads if pattern.match(r)] or [r for r in reads if r]
        return Counter(pool).most_common(1)[0][0] if pool else ""

    def read(self, crop_bgr) -> tuple[str, int]:
        """-> (plate text without separators, recognize() calls spent)."""
        import cv2
        before = self.calls
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        h, w = gray.shape[:2]
        if w / h < 2.5:
            text = self._line(gray[:h // 2], TOP_LINE) + self._line(gray[h // 2:], BOTTOM_LINE)
        else:
            text = self._line(gray, VN_PLATE)
        return text, self.calls - before
