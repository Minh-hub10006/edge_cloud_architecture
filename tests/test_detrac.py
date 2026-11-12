"""UA-DETRAC tooling: annotation parsing, index/split, vehicle-only scoring with ignored regions."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

def _load_script(name):
    """Import scripts/<name>.py by path; adding scripts/ to sys.path would shadow same-named tests."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


detrac = _load_script("detrac")

XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_00001">
  <sequence_attribute camera_state="unstable" sence_weather="night"/>
  <ignored_region><box left="900" top="0" width="60" height="60"/></ignored_region>
  <frame density="2" num="1"><target_list>
    <target id="1"><box left="100" top="100" width="100" height="50"/><attribute vehicle_type="car"/></target>
    <target id="2"><box left="400" top="300" width="80" height="40"/><attribute vehicle_type="bus"/></target>
  </target_list></frame>
  <frame density="1" num="2"><target_list>
    <target id="1"><box left="910" top="10" width="20" height="20"/><attribute vehicle_type="car"/></target>
  </target_list></frame>
</sequence>"""


def norm(l, t, r, b):
    return [l / 960, t / 540, r / 960, b / 540]


class DetracTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        (self.tmp / "DETRAC-Train-Annotations-XML").mkdir()
        (self.tmp / "DETRAC-Train-Annotations-XML" / "MVI_00001_v3.xml").write_text(XML, encoding="utf-8")

    def test_parse_annotation(self):
        ann = detrac.parse_annotation(self.tmp / "DETRAC-Train-Annotations-XML" / "MVI_00001_v3.xml")
        self.assertEqual((ann["name"], ann["weather"], ann["camera"]), ("MVI_00001", "night", "unstable"))
        self.assertEqual(ann["ignored"], [(900., 0., 960., 60.)])
        self.assertEqual(ann["frames"][1], [(100., 100., 200., 150.), (400., 300., 480., 340.)])

    def test_index_matches_versioned_xml_and_split(self):
        for split, name in (("Insight-MVT_Annotation_Train", "MVI_00001"), ("Insight-MVT_Annotation_Test", "MVI_00002")):
            seq = self.tmp / split / name
            seq.mkdir(parents=True)
            for i in (1, 2, 3):
                (seq / f"img{i:05d}.jpg").write_bytes(b"x")
        rows = {r["sequence"]: r for r in detrac.index(self.tmp)}
        self.assertEqual((rows["MVI_00001"]["split"], rows["MVI_00001"]["frames"], rows["MVI_00001"]["weather"]),
                         ("train", 3, "night"))
        self.assertEqual((rows["MVI_00002"]["split"], rows["MVI_00002"]["annotation"]), ("test", None))

    def test_index_split_from_annotation_folder_when_images_are_flat(self):
        (self.tmp / "DETRAC-Test-Annotations-XML").mkdir()
        (self.tmp / "DETRAC-Test-Annotations-XML" / "MVI_00002.xml").write_text(
            XML.replace("MVI_00001", "MVI_00002").replace("night", "sunny"), encoding="utf-8")
        for name in ("MVI_00001", "MVI_00002", "MVI_00003"):
            seq = self.tmp / "DETRAC-Images" / name
            seq.mkdir(parents=True)
            (seq / "img00001.jpg").write_bytes(b"x")
        rows = {r["sequence"]: r for r in detrac.index(self.tmp)}
        self.assertEqual((rows["MVI_00001"]["split"], rows["MVI_00001"]["weather"]), ("train", "night"))
        self.assertEqual((rows["MVI_00002"]["split"], rows["MVI_00002"]["weather"]), ("test", "sunny"))
        self.assertEqual((rows["MVI_00003"]["split"], rows["MVI_00003"]["annotation"]), (None, None))

    def test_match_frame_classes_ignored_and_greedy(self):
        gt = [(100, 100, 200, 150), (900, 0, 950, 50)]  # second lies in the ignored region
        preds = [(.9, (102, 101, 199, 151)), (.8, (101, 100, 200, 150)), (.7, (905, 5, 945, 45))]
        self.assertEqual(detrac.match_frame(gt, preds, [(890, 0, 960, 60)]), (1, 1, 0))
        self.assertEqual(detrac.match_frame(gt, [], []), (0, 0, 2))

    def test_score_windows_vehicle_only_and_empty_window(self):
        dump = {"artifact": "m", "frames": [
            {"frame": 0, "boxes": [[2, .9, *norm(100, 100, 200, 150)],   # car TP
                                   [0, .9, *norm(400, 300, 480, 340)],   # person: ignored, bus GT missed
                                   [7, .6, *norm(600, 400, 700, 450)]]},  # truck FP
            {"frame": 1, "boxes": [[2, .9, *norm(912, 12, 928, 28)]]},   # GT + prediction both ignored
            {"frame": 25, "boxes": [[2, .9, *norm(10, 10, 50, 50)]]}]}     # frame 26 is not in the XML
        xml = self.tmp / "DETRAC-Train-Annotations-XML" / "MVI_00001_v3.xml"
        (first,) = detrac.score(xml, dump)  # unlabeled frame 26: no second row, no invented false positive
        self.assertEqual((first["tp"], first["fp"], first["fn"], first["frames"], first["gt"]), (1, 1, 1, 2, 2))
        self.assertAlmostEqual(first["f1"], 2 / 4)
        self.assertEqual(first["weather"], "night")
        # One-frame windows: frame 2 keeps only ignored boxes -> nothing to find, nothing invented.
        rows = detrac.score(xml, dump, window_s=1 / 25)
        self.assertEqual([(r["window"], r["start_frame"], r["f1"], r["gt"]) for r in rows][1], (1, 2, 1.0, 0))
        self.assertEqual(len(rows), 2)

    def test_score_at_lower_analytics_fps_holds_the_latest_analyzed_boxes(self):
        self.assertEqual(sum(detrac._on_grid(i, 15, 25) for i in range(25)), 15)
        self.assertEqual(sum(detrac._on_grid(j, 5, 15) for j in range(15)), 5)
        box = (100, 100, 200, 150)
        ann = {"name": "S", "weather": "sunny", "ignored": [], "frames": {i + 1: [box] for i in range(25)}}
        hit = [[2, .9, *norm(*box)]]
        # Only frame 0 detects the car: analyzing every shown frame finds it once; holding at 5 FPS keeps
        # it until the next analyzed frame (an empty one), so 3 of the 15 shown frames match.
        dump = {"frames": [{"frame": i, "boxes": hit if i == 0 else []} for i in range(25)]}
        (all15,) = detrac.score(ann, dump, analyze_fps=15)
        (at5,) = detrac.score(ann, dump, analyze_fps=5)
        self.assertEqual((all15["frames"], all15["tp"]), (15, 1))
        self.assertEqual((at5["frames"], at5["tp"]), (15, 3))
        # A car detected on every frame: lower analytics FPS loses nothing for a still scene.
        still = {"frames": [{"frame": i, "boxes": hit} for i in range(25)]}
        self.assertEqual(detrac.score(ann, still, analyze_fps=5)[0]["f1"], 1.0)

    def test_score_cli_writes_jsonl(self):
        dump = self.tmp / "dump.json"
        dump.write_text(json.dumps({"artifact": "lw", "frames": [{"frame": 0, "boxes": [[2, .9, *norm(100, 100, 200, 150)]]}]}))
        out = self.tmp / "q.jsonl"
        detrac.main(["score", str(self.tmp / "DETRAC-Train-Annotations-XML" / "MVI_00001_v3.xml"), str(dump),
                     "--out", str(out)])
        (row,) = [json.loads(x) for x in out.read_text().splitlines()]
        self.assertEqual((row["model_artifact"], row["tp"], row["fn"]), ("lw", 1, 1))

    def test_table_scores_dump_folder_with_split_model_and_common_threshold(self):
        dumps = self.tmp / "dumps"
        dumps.mkdir()
        frame0 = [[2, .9, *norm(100, 100, 200, 150)], [2, .3, *norm(600, 400, 700, 450)]]  # TP + weak FP
        (dumps / "MVI_00001.json").write_text(json.dumps({"frames": [{"frame": 0, "boxes": frame0}]}))
        (dumps / "MVI_09999.json").write_text(json.dumps({"frames": []}))  # no annotation -> skipped
        (row,) = detrac.table(self.tmp, dumps, "yolov8n")
        self.assertEqual((row["model"], row["split"], row["sequence"], row["fp"]), ("yolov8n", "train", "MVI_00001", 1))
        (strict,) = detrac.table(self.tmp, dumps, "yolov8n", min_conf=.5)
        self.assertEqual((strict["tp"], strict["fp"]), (1, 0))
        out = self.tmp / "t.jsonl"
        detrac.main(["table", str(self.tmp), str(dumps), "--model", "m", "--out", str(out)])
        self.assertEqual(len(out.read_text().splitlines()), 1)

    def test_average_precision_perfect_then_confident_false_positive(self):
        dumps = self.tmp / "dumps"
        dumps.mkdir()
        exact = [[2, .9, *norm(100, 100, 200, 150)], [5, .8, *norm(400, 300, 480, 340)]]
        frames = [{"frame": 0, "boxes": exact}, {"frame": 1, "boxes": [[2, .9, *norm(910, 10, 930, 30)]]}]
        (dumps / "MVI_00001.json").write_text(json.dumps({"frames": frames}))
        ap = detrac.average_precision(self.tmp, dumps, split="train")
        # 2 scored GT (the ignored-region car and the ignored prediction drop out); both found exactly
        self.assertEqual((ap["gt"], ap["frames"]), (2, 2))
        self.assertAlmostEqual(ap["AP50"], 1.0)
        self.assertAlmostEqual(ap["AP50:95"], 1.0)
        frames[0]["boxes"] = [[2, .95, *norm(600, 400, 700, 450)], *exact]  # most confident box is wrong
        (dumps / "MVI_00001.json").write_text(json.dumps({"frames": frames}))
        worse = detrac.average_precision(self.tmp, dumps, split="train")
        self.assertLess(worse["AP50"], 1.0)
        self.assertGreater(worse["AP50"], .5)
        self.assertEqual(detrac.average_precision(self.tmp, dumps, split="test")["sequences"], 0)

    def test_offline_box_change_equals_live_formula_when_ids_agree(self):
        from edge_agent.phase7 import _box_change as live
        prev = [(.1, .1, .2, .2), (.5, .5, .6, .6), (.8, .1, .9, .2)]
        cur = [(.11, .1, .21, .2), (.5, .5, .6, .6), (.3, .3, .35, .35)]  # moved, same, new (third left)
        expected = live({1: prev[0], 2: prev[1], 3: prev[2]}, {1: cur[0], 2: cur[1], 4: cur[2]})
        self.assertAlmostEqual(detrac._box_change(prev, cur), expected)
        self.assertEqual(detrac._box_change([], []), 0.0)
        self.assertEqual(detrac._box_change([], [(.1, .1, .2, .2)]), 1.0)

    def test_observed_features_use_live_cadence_and_vehicle_classes(self):
        car = [2, .9, .1, .1, .2, .2]
        tiny_moto = [3, .9, .5, .5, .505, .505]
        person = [0, .9, .3, .3, .4, .4]
        frames = [{"frame": i, "boxes": [car, tiny_moto, person] if i % 2 else [car]} for i in range(25)]
        (row,) = detrac.observed_features(frames, size=25).values()
        kept = [i for i in range(25) if i == 0 or int(i * 15 / 25) != int((i - 1) * 15 / 25)]
        self.assertEqual(len(kept), 15)
        counts = [2 if i % 2 else 1 for i in kept]
        self.assertAlmostEqual(row["vehicle_count"], sum(counts) / 15)
        self.assertAlmostEqual(row["small_ratio"], sum(.5 if c == 2 else 0 for c in counts) / 15)
        self.assertGreater(row["box_change"], 0)

    def test_score_rows_carry_observed_features(self):
        dump = {"frames": [{"frame": i, "boxes": [[2, .9, *norm(100, 100, 200, 150)]]} for i in range(3)]}
        (row,) = detrac.score(self.tmp / "DETRAC-Train-Annotations-XML" / "MVI_00001_v3.xml", dump)
        self.assertEqual((row["vehicle_count"], row["small_ratio"], row["box_change"]), (1, 0, 0))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_scene_features_per_second_from_video(self):
        import cv2
        import numpy as np
        path = self.tmp / "clip.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 180))
        for i in range(50):  # second 0 dark and static, second 1 alternating bright frames
            value = 0 if i < 25 else (200 if i % 2 else 0)
            writer.write(np.full((180, 320, 3), value, np.uint8))
        writer.release()
        first, second = detrac.scene(path)
        self.assertEqual((first["window"], second["window"]), (0, 1))
        self.assertAlmostEqual(first["brightness"], 0, delta=2)
        self.assertGreater(second["brightness"], 50)
        self.assertGreater(second["motion"], 50)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_video_from_images_keeps_frame_count(self):
        import cv2
        import numpy as np
        seq = self.tmp / "Insight-MVT_Annotation_Train" / "MVI_00001"
        seq.mkdir(parents=True)
        for i in range(1, 6):
            cv2.imwrite(str(seq / f"img{i:05d}.jpg"), np.full((540, 960, 3), i * 40, np.uint8))
        (made,) = detrac.video(self.tmp, self.tmp / "videos")
        capture = cv2.VideoCapture(str(made))
        count = 0
        while capture.read()[0]:
            count += 1
        capture.release()
        self.assertEqual(count, 5)
        self.assertEqual(detrac.video(self.tmp, self.tmp / "videos"), [made])  # idempotent


class PlaylistTests(unittest.TestCase):
    MANIFEST = {"fps": 25, "total_frames": 75, "sequences": [
        {"sequence": "A", "start_frame": 0, "frames": 50}, {"sequence": "B", "start_frame": 50, "frames": 25}]}

    def test_locate_wraps_looping_runs_and_label_windows_uses_median_frame(self):
        self.assertEqual(detrac.locate(0, self.MANIFEST), ("A", 0, 0))
        self.assertEqual(detrac.locate(30, self.MANIFEST), ("A", 30, 1))
        self.assertEqual(detrac.locate(74, self.MANIFEST), ("B", 24, 0))
        self.assertEqual(detrac.locate(75 + 52, self.MANIFEST), ("B", 2, 0))  # second loop
        lines = [json.dumps({"window_id": 0, "frames": [{"frame_id": i} for i in (26, 30, 34)]}),
                 json.dumps({"window_id": 1, "frames": []})]
        first, empty = detrac.label_windows(lines, self.MANIFEST)
        self.assertEqual(first["scene"], {"sequence": "A", "second": 1, "frames": 3})
        self.assertNotIn("scene", empty)
        # Staged run starting at frame 50 (B); frame 30 of the run is clip frame 80 -> wraps to A.
        staged = [json.dumps({"window_id": 0, "frames": [{"frame_id": 30}]})]
        (late,) = detrac.label_windows(staged, self.MANIFEST, start_frame=50)
        self.assertEqual(late["scene"], {"sequence": "A", "second": 0, "frames": 1})
        (early,) = detrac.label_windows([json.dumps({"frames": [{"frame_id": 10}]})], self.MANIFEST, start_frame=50)
        self.assertEqual(early["scene"]["sequence"], "B")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_playlist_concatenates_split_sequences_without_reencoding_frames(self):
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "DETRAC-Train-Annotations-XML").mkdir()
            for name, weather, count in (("MVI_00001", "night", 4), ("MVI_00002", "sunny", 6)):
                (root / "DETRAC-Train-Annotations-XML" / f"{name}.xml").write_text(
                    XML.replace("MVI_00001", name).replace("night", weather), encoding="utf-8")
                seq = root / "DETRAC-Images" / name
                seq.mkdir(parents=True)
                for i in range(1, count + 1):
                    cv2.imwrite(str(seq / f"img{i:05d}.jpg"), np.full((540, 960, 3), i * 30, np.uint8))
            detrac.video(root, root / "videos")
            chosen = detrac.playlist(root, root / "videos", root / "clip.mp4", minutes=1)
            manifest = json.loads((root / "clip.json").read_text())
            capture = cv2.VideoCapture(str(root / "clip.mp4"))
            frames = 0
            while capture.read()[0]:
                frames += 1
            capture.release()
        self.assertEqual({c["sequence"] for c in chosen}, {"MVI_00001", "MVI_00002"})
        self.assertEqual((manifest["total_frames"], frames), (10, 10))
        self.assertEqual([c["start_frame"] for c in manifest["sequences"]], [0, manifest["sequences"][0]["frames"]])


class LiveTranscodeTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg unavailable")
    def test_keeps_every_frame_at_live_size_with_identity_band(self):
        import cv2
        import numpy as np
        from media.frame_identity import BAND_HEIGHT
        lt = _load_script("live_transcode")
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "in" / "MVI_1.mp4", Path(tmp) / "out"
            src.parent.mkdir()
            writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 180))
            for i in range(30):
                writer.write(np.full((180, 320, 3), i * 8, np.uint8))
            writer.release()
            lt.main([str(src.parent), str(out)])
            capture = cv2.VideoCapture(str(out / "MVI_1.mp4"))
            shapes = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                shapes.append(frame.shape)
            capture.release()
            self.assertEqual(len(shapes), 30)
            self.assertEqual(shapes[0], (720 + BAND_HEIGHT, 1280, 3))
            self.assertEqual(list((out).glob("*.part")) + list(out.glob("*.tmp.mp4")), [])
            lt.main([str(src.parent), str(out)])  # resumable: existing output is kept


class EdgeParityCropTests(unittest.TestCase):
    def test_dump_crops_identity_band_before_inference(self):
        from unittest import mock
        import cv2
        import numpy as np
        parity = _load_script("edge_parity")
        seen = []

        class Detector:
            def infer(self, frame):
                seen.append(frame.shape)
                return [], 0.0
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "v.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (128, 100))
            for _ in range(2):
                writer.write(np.zeros((100, 128, 3), np.uint8))
            writer.release()
            args = SimpleNamespace(adapter="a", artifact="m", video=str(video), out=str(Path(tmp) / "o.json"),
                                   frames=10, stride=1, imgsz=320, layout="nchw", confidence=.25, nms_iou=.5,
                                   allow_diagnostic=False, crop_bottom=64)
            with mock.patch("benchmark.model.load_adapter", return_value=(Detector(), {"device": "x"})):
                parity.dump(args)
        self.assertEqual(seen, [(36, 128, 3), (36, 128, 3)])


class QualityDumpScriptTests(unittest.TestCase):
    def run_script(self, *args, **env):
        import os, subprocess
        from test_control_v1_four_modes import _bash  # tests/ is on sys.path under discovery
        bash = _bash()
        if bash is None:
            self.skipTest("bash unavailable")
        root = Path(__file__).resolve().parents[1]
        return subprocess.run([bash, "scripts/quality_dump.sh", *args], cwd=root, capture_output=True, text=True,
                              timeout=60, env={**os.environ, "EDGE_NODE": "edge-246", **env})

    def test_usage_missing_folder_and_bad_shard_fail_early(self):
        usage = self.run_script("yolov8n")
        self.assertEqual(usage.returncode, 2)
        self.assertIn("Usage", usage.stderr)
        self.assertEqual(self.run_script("resnet", "a", "b").returncode, 2)
        missing = self.run_script("yolov8n", "/no/such/dir", "out")
        self.assertIn("[Q0] FAIL: missing video folder", missing.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            bad = self.run_script("yolov8n", tmp, "out", QUALITY_SHARD="2/2")
        self.assertIn("[Q0] FAIL: QUALITY_SHARD", bad.stderr)


if __name__ == "__main__":
    unittest.main()
