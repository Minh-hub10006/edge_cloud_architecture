import unittest
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile
from types import SimpleNamespace

import numpy as np

from yolo11s_contract import Detection as Box, SHAPES, agreement, decode, prepare


class Yolo11CandidateContractTests(unittest.TestCase):
    def test_nms_review_thresholds_do_not_change_default(self):
        from scripts.yolo11s_nms_review import compare_thresholds
        prepared = SimpleNamespace(width=320, height=320, gain=1, pad_x=0, pad_y=0)
        boxes = np.zeros(SHAPES["boxes"], np.float32)
        scores = np.zeros(SHAPES["scores"], np.float32)
        boxes[0, :3] = [[0, 0, 100, 100], [20, 0, 120, 100], [20, 0, 120, 100]]
        scores[0, 0, 2], scores[0, 1, 2], scores[0, 2, 3] = .9, .8, .7
        values = {"boxes": boxes, "scores": scores}
        rows = compare_thresholds(values, prepared, decode(values, prepared))
        self.assertEqual([r["count"] for r in rows], [3, 2, 2])
        self.assertEqual(len(decode(values, prepared)), 3)
        self.assertEqual(rows[1]["detections"][1]["class_id"], 3)

    def test_score_encoding_rejects_mixed_coordinate_range(self):
        from scripts.yolo11s_snpe_candidate import check_score_encoding
        check_score_encoding("scores encoding : bitwidth 8, min 0.0, max 0.8, scale 0.003137")
        with self.assertRaises(ValueError):
            check_score_encoding("scores encoding : bitwidth 8, min 0.0, max 332.262695, scale 1.3029909")
        with self.assertRaises(ValueError):
            check_score_encoding("encoding missing")

    def test_diagnostic_packet_requires_opt_in(self):
        from scripts.yolo11s_snpe_candidate import diagnostic_marker
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("edge_packet/diagnostic.json", json.dumps(
                {"diagnostic_only": True, "production_ready": False}))
        with zipfile.ZipFile(stream) as archive:
            with self.assertRaises(ValueError):
                diagnostic_marker(archive, False)
            self.assertTrue(diagnostic_marker(archive, True)["diagnostic_only"])

    def test_diagnostic_prepares_existing_dlc_without_quantization(self):
        from scripts import yolo11s_snpe_candidate as runner
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "existing.dlc"
            candidate.write_bytes(b"existing candidate")
            refs = []
            for index in (120, 600, 1200):
                folder = root / f"frame_{index:06d}"
                folder.mkdir()
                for filename in ("source.png", "input_nhwc.raw", "reference.json"):
                    (folder / filename).write_bytes(b"test")
                refs.append((index, folder, {}, []))
            work = root / "prepare"
            work.mkdir()
            commands = []
            def fake_run(command, log, env, timeout):
                commands.append([str(v) for v in command])
                Path(log).write_text("test")
                if "--output_dlc" in command:
                    Path(command[command.index("--output_dlc") + 1]).write_bytes(b"prepared")
                return "--htp_socs --set_output_tensors"
            def fake_inspect(*args):
                (work / "qcs6490_info.log").write_text("test")
                return "graph", "HTP_V68_QCS6490_2MB boxes scores"
            with patch.object(runner, "run", side_effect=fake_run), \
                    patch.object(runner, "inspect_dlc", side_effect=fake_inspect), \
                    patch("sys.stdout", new=io.StringIO()):
                runner.prepare_packet(root, root, {}, candidate, "graph", work, refs,
                                      diagnostic={"status": "W8A8_CPU_FAIL"})
            self.assertEqual(candidate.read_bytes(), b"existing candidate")
            self.assertTrue(all(Path(cmd[0]).name == "snpe-dlc-graph-prepare" for cmd in commands))
            with zipfile.ZipFile(work / "edge_packet.zip") as archive:
                self.assertEqual(len(archive.namelist()), 13)
                marker = runner.diagnostic_marker(archive, True)
                self.assertFalse(marker["production_ready"])
                self.assertEqual(marker["cpu_report"]["status"], "W8A8_CPU_FAIL")

    def test_sdk_graph_info_formats(self):
        from scripts.yolo11s_snpe_candidate import graph_name
        self.assertEqual(graph_name("Info of graph: yolo11s320_float\n"), "yolo11s320_float")
        self.assertEqual(graph_name("| Graph Name: yolo11s320_float |"), "yolo11s320_float")
        with self.assertRaises(ValueError):
            graph_name("Cache Info:")
        with self.assertRaises(ValueError):
            graph_name("Info of graph: first\nInfo of graph: second\n")

    def test_reference_reader_accepts_platform_metadata(self):
        from scripts.yolo11s_snpe_candidate import reference_boxes
        box = Box(.1, .2, .6, .8, .9, 2, "car")
        record = {"detections": [{**asdict(box), "track_id": None,
                                  "calibrated_confidence": None}]}
        self.assertEqual(reference_boxes(record), [box])
        record["detections"][0]["unexpected"] = 1
        with self.assertRaises(ValueError):
            reference_boxes(record)

    def test_letterbox_is_centered_rgb_float32_nchw(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[:, :, 2] = 255
        value = prepare(image)
        self.assertEqual(value.tensor.shape, (1, 3, 320, 320))
        self.assertEqual(value.tensor.dtype, np.float32)
        self.assertGreater(float(value.tensor[0, 0, 160, 100]), .99)
        self.assertEqual(float(value.tensor[0, 2, 160, 100]), 0.0)
        self.assertAlmostEqual(float(value.tensor[0, 0, 0, 0]), 114 / 255, places=7)
        self.assertEqual((value.gain, value.pad_x, value.pad_y), (1.6, 0, 80))

    def test_decode_letterbox_and_class_aware_nms(self):
        prepared = SimpleNamespace(width=200, height=100, gain=1.6, pad_x=0, pad_y=80)
        boxes = np.zeros(SHAPES["boxes"], np.float32)
        scores = np.zeros(SHAPES["scores"], np.float32)
        boxes[0, 0] = [32, 112, 192, 272]
        boxes[0, 1] = [33, 113, 191, 271]
        boxes[0, 2] = [33, 113, 191, 271]
        scores[0, 0, 2] = .9
        scores[0, 1, 2] = .8
        scores[0, 2, 3] = .7
        detections = decode({"boxes": boxes, "scores": scores}, prepared)
        self.assertEqual(len(detections), 2)
        self.assertEqual([x.class_id for x in detections], [2, 3])
        self.assertAlmostEqual(detections[0].left, .1)
        self.assertAlmostEqual(detections[0].top, .2)

    def test_bad_tensor_contract_rejected(self):
        prepared = SimpleNamespace(width=200, height=100, gain=1.6, pad_x=0, pad_y=80)
        with self.assertRaises(ValueError):
            decode({"boxes": np.zeros((1, 1, 4), np.float32),
                    "scores": np.zeros(SHAPES["scores"], np.float32)}, prepared)

    def test_float_reference_gate_is_strict_and_identifies_no_map(self):
        box = Box(.1, .2, .6, .8, .9, 2, "car")
        same = Box(.1, .2, .6, .8, .9, 2, "car")
        good = agreement([box], [same])
        self.assertEqual(good["gate"], "PASS")
        self.assertEqual(good["quality_status"], "NOT_EVALUATED")
        shifted = Box(.4, .2, .8, .8, .9, 2, "car")
        self.assertEqual(agreement([box], [shifted])["gate"], "FAIL")


if __name__ == "__main__":
    unittest.main()
