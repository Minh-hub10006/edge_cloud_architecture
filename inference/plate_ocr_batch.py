"""UC1 OCR in batches (13/11): the same reading as inference.plate_ocr.PlateReader (EasyOCR 1.7.2 recognize(), greedy
CTC, contrast retry under 0.1 confidence, 2 lines x margin passes, VN-format vote), but every line image of a crop is
prepared here and sent to the recognizer at once, grouped by padded width: Triton (model plate_ocr) then batches lines
of many crops and cameras together. The BiLSTM reads each sample alone, so batching does not change a result.

Line preparation copies EasyOCR step by step: utils.get_image_list / compute_ratio_and_resize, recognition.AlignCollate
(keep_ratio_with_pad) / NormalizePAD / adjust_contrast_grey / recognizer_predict / get_text, CTCLabelConverter.
scripts/plate_ocr_parity.py checks the texts against EasyOCR itself.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from inference.plate_ocr import BOTTOM_LINE, CHARS, MARGINS, TOP_LINE, VN_PLATE

IMG_H = 64
CONTRAST_THS, ADJUST_CONTRAST = 0.1, 0.5


def _ratio(w: int, h: int) -> float:   # utils.calculate_ratio
    r = w / h
    return 1 / r if r < 1.0 else r


def _adjust_contrast(img, target: float):   # recognition.adjust_contrast_grey
    high, low = np.percentile(img, 90), np.percentile(img, 10)
    if (high - low) / np.maximum(10, high + low) < target:
        img = (img.astype(int) - low + 25) * (200. / np.maximum(10, high - low))
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


def prepare(gray, adjust: float = 0.0):
    """One line image (uint8 grey) -> 1x64xW float32, W a multiple of 64, exactly as recognize() feeds the model;
    None when EasyOCR would skip it (zero width)."""
    import cv2
    from PIL import Image
    h, w = gray.shape[:2]
    ratio = _ratio(w, h)
    if int(IMG_H * ratio) == 0:
        return None
    # compute_ratio_and_resize: interpolation=Image.Resampling.LANCZOS (= 1) given to cv2, i.e. cv2.INTER_LINEAR
    img = (cv2.resize(gray, (IMG_H, int(IMG_H * ratio)), interpolation=1) if w / h < 1.0
           else cv2.resize(gray, (int(IMG_H * ratio), IMG_H), interpolation=1))
    img_w = math.ceil(max(ratio, 1)) * IMG_H                     # get_image_list: max_width
    pil = Image.fromarray(img, "L")
    if adjust > 0:
        pil = Image.fromarray(_adjust_contrast(np.array(pil.convert("L")), adjust), "L")
    pw, ph = pil.size
    resized_w = min(math.ceil(IMG_H * pw / float(ph)), img_w)
    x = np.asarray(pil.resize((resized_w, IMG_H), Image.BICUBIC), np.float32) / 255.
    x = (x - .5) / .5
    out = np.empty((1, IMG_H, img_w), np.float32)
    out[0, :, :resized_w] = x
    out[0, :, resized_w:] = x[:, -1:]                            # NormalizePAD: the last column repeated
    return out


def decode(logits, ignore_idx=()) -> tuple[str, float]:
    """TxC logits of one line -> (text, confidence) as recognizer_predict + decode_greedy."""
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = e / e.sum(axis=1, keepdims=True)
    if len(ignore_idx):
        prob[:, list(ignore_idx)] = 0.
        prob = prob / prob.sum(axis=1, keepdims=True)
    idx = prob.argmax(axis=1)
    keep = np.insert(idx[1:] != idx[:-1], 0, True) & (idx != 0)
    text = "".join(CHARS[i - 1] for i in idx[keep])
    best = prob.max(axis=1)[idx != 0]
    if not len(best):
        best = np.array([0.])
    return text, float(best.prod() ** (2.0 / np.sqrt(len(best))))


class BatchPlateReader:
    """infer(N x 1 x 64 x W float32) -> N x T x 37 logits: Triton (TritonSession plate_ocr), ONNX Runtime or the torch
    model. Thread-safe as long as infer is."""

    def __init__(self, infer, passes: int = 3) -> None:
        self.infer = infer
        self.margins = MARGINS[:max(1, min(passes, len(MARGINS)))]

    def _recognize(self, crops, adjust: float = 0.0) -> list[tuple[str, float]]:
        out = [("", 0.0)] * len(crops)
        groups = defaultdict(list)
        for i, c in enumerate(crops):
            x = prepare(c, adjust)
            if x is not None:
                groups[x.shape[-1]].append((i, x))
        for items in groups.values():   # one call per padded width: equal shapes, batched by Triton across crops
            logits = self.infer(np.stack([x for _, x in items]))
            for (i, _), lg in zip(items, logits):
                out[i] = decode(lg)
        return out

    def recognize(self, crops) -> list[str]:
        """get_text for each line image: second pass with more contrast when the first is under 0.1 confident."""
        first = self._recognize(crops)
        low = [i for i, (_, conf) in enumerate(first) if conf < CONTRAST_THS]
        if low:
            second = self._recognize([crops[i] for i in low], ADJUST_CONTRAST)
            for i, again in zip(low, second):
                if not first[i][1] > again[1]:
                    first[i] = again
        return ["".join(c for c in t.upper() if c in CHARS) for t, _ in first]

    def read(self, crop_bgr) -> tuple[str, int]:
        """-> (plate text, line images read): PlateReader.read with every pass of every line in one go."""
        import cv2
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
        h, w = gray.shape[:2]
        lines = ([(gray[:h // 2], TOP_LINE), (gray[h // 2:], BOTTOM_LINE)] if w / h < 2.5 else [(gray, VN_PLATE)])
        crops = []
        for g, _ in lines:
            gh, gw = g.shape[:2]
            for mx, my in self.margins:   # PlateReader._once
                crops.append(g[int(gh * my): max(int(gh * (1 - my)), int(gh * my) + 4),
                               int(gw * mx): max(int(gw * (1 - mx)), int(gw * mx) + 4)])
        reads = self.recognize(crops)
        text, n = "", len(self.margins)
        for k, (_, pattern) in enumerate(lines):   # PlateReader._line: the most common read in the VN format
            part = reads[k * n:(k + 1) * n]
            pool = [r for r in part if pattern.match(r)] or [r for r in part if r]
            text += Counter(pool).most_common(1)[0][0] if pool else ""
        return text, len(crops)
