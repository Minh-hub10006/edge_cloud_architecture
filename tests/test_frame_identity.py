from __future__ import annotations
import base64
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from media.frame_identity import append_identity, read_identity, stream_tag, BAND_HEIGHT

KEY = ("edge-246", "traffic-01", "timing", 991)


class FrameIdentityTests(unittest.TestCase):
    def test_crop_identity_epoch_and_corruption(self):
        pixels = np.full((72, 128, 3), 111, np.uint8)
        encoded = append_identity(pixels, key=KEY, seq=123456, fps=29)
        value = read_identity(encoded, key=KEY)
        self.assertEqual((value.seq, value.fps, value.padding), (123456, 29, False))
        np.testing.assert_array_equal(encoded[:-BAND_HEIGHT], pixels)
        with self.assertRaises(ValueError):
            read_identity(encoded, key=KEY[:-1] + (992,))
        encoded[-64:-56, :8] = 255 - encoded[-64:-56, :8]
        with self.assertRaises(ValueError):
            read_identity(encoded, key=KEY)

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg unavailable")
    def test_identity_survives_real_h264_with_late_join_and_browser_decoder(self):
        import cv2
        # Nonzero start proves readers don't reconstruct sequence from local PTS.
        frames = []
        for seq in range(701, 713):
            pixels = np.zeros((720, 1280, 3), np.uint8)
            pixels[200:400, 100 + (seq-701)*10:300 + (seq-701)*10] = (80, 170, 230)
            frames.append(append_identity(pixels, key=KEY, seq=seq, fps=29, padding=seq==712))
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "identity.mp4")
            result = subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
                "-pixel_format", "bgr24", "-video_size", "1280x784", "-framerate", "29", "-i", "pipe:0",
                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-profile:v", "baseline", "-bf", "0", "-pix_fmt", "yuv420p", "-b:v", "1000k", path,
            ], input=b"".join(f.tobytes() for f in frames), capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
            cap = cv2.VideoCapture(path)
            identities = []
            try:
                while True:
                    ok, decoded = cap.read()
                    if not ok:
                        break
                    identities.append(read_identity(decoded, key=KEY))
                    last = decoded
            finally:
                cap.release()
            self.assertEqual([i.seq for i in identities], list(range(701, 713)))
            self.assertTrue(identities[-1].padding)
            if shutil.which("node"):
                rgba = cv2.cvtColor(last[-64:, :128], cv2.COLOR_BGR2RGBA)
                code = "const fs=require('fs'),s=require('./web/live_sync.js'); const b=Buffer.from(fs.readFileSync(0,'utf8'),'base64'); console.log(JSON.stringify(s.decodeIdentity(b," + str(stream_tag(KEY)) + ")));"
                browser = subprocess.run(["node", "-e", code], input=base64.b64encode(rgba.tobytes()), capture_output=True, timeout=10)
                self.assertEqual(browser.returncode, 0, browser.stderr)
                marker = json.loads(browser.stdout)
                self.assertEqual((marker["seq"], marker["fps"], marker["padding"]), (712, 29, True))
                self.assertEqual(marker["pts"], identities[-1].pts_ns)
