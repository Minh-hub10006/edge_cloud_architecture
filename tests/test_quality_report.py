"""scripts/quality_report.py: completeness gate and the simulator training table."""
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from test_detrac import XML, norm  # tests/ is on sys.path under discovery


def _load():
    spec = importlib.util.spec_from_file_location(
        "quality_report", Path(__file__).resolve().parents[1] / "scripts" / "quality_report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qr = _load()


class QualityReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.ann = self.tmp / "ann" / "DETRAC-Train-Annotations-XML"
        self.ann.mkdir(parents=True)
        (self.ann / "MVI_00001.xml").write_text(XML, encoding="utf-8")
        self.res = self.tmp / "res"
        self.res.mkdir()
        (self.res / "index.jsonl").write_text(json.dumps({"sequence": "MVI_00001", "frames": 2}) + "\n"
                                              + json.dumps({"sequences": 1}) + "\n")
        (self.res / "scene.jsonl").write_text(json.dumps({"sequence": "MVI_00001", "window": 0,
                                                          "brightness": 90.0, "motion": 4.0}) + "\n")

    def dump(self, folder, frames=2):
        path = self.res / folder
        path.mkdir(parents=True, exist_ok=True)
        boxes = [[2, .9, *norm(100, 100, 200, 150)]]
        (path / "MVI_00001.json").write_text(json.dumps({"frames": [{"frame": i, "boxes": boxes} for i in range(frames)]}))

    def test_incomplete_dump_fails_the_gate(self):
        self.dump("quality/yolov8n", frames=1)
        self.assertEqual(qr.verify(self.res, {"MVI_00001": 2}), {"yolov8n_fp32": ["MVI_00001 1/2"]})
        with self.assertRaises(SystemExit):
            qr.main([str(self.tmp / "ann"), str(self.res)])

    def test_training_table_joins_edge_fp32_and_cloud_live(self):
        for folder in ("quality/yolov8n", "quality/yolo11s", "quality_live/rtdetrv2", "quality_live/lwdetr",
                       "quality_crf18/lwdetr"):
            self.dump(folder)
        qr.main([str(self.tmp / "ann"), str(self.res)])
        rows = [json.loads(x) for x in (self.res / "table_train.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(r["model"] for r in rows), ["lwdetr", "rtdetrv2", "yolo11s", "yolov8n"])
        report = (self.res / "QUALITY_REPORT.md").read_text(encoding="utf-8")
        self.assertIn("Choosing A_min", report)
        self.assertIn("lwdetr live - crf18", report)


if __name__ == "__main__":
    unittest.main()
