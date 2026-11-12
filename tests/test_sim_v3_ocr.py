"""drl.sim_v3 plate-OCR load on the shared Cloud GPU (4090 co-run calibration, 29/10)."""
import dataclasses
import unittest

from drl.sim_v3 import FPS_LEVELS, TABLE_NAMES, SimV3, SimV3Config, TableV3


def table(new_per_second):
    rows = [{"sequence": "S0", "window": w, "split": "test", "model": m, "fps": f, "f1": .8}
            for w in range(30) for m in TABLE_NAMES for f in FPS_LEVELS]
    counts = [{"dataset": "detrac", "split": "test", "sequence": "S0", "second": w, "new_vehicles": new_per_second}
              for w in range(30)]
    return TableV3(rows, "test", count_rows=counts)


def cloud_latency(cfg, new):
    sim = SimV3(table(new), cfg, 0)
    sim.reset(6)
    secs = sim.step([("dfine_x_cloud", 15)] * 6)
    return secs[-1][0]["latency"], secs[-1][0]


class OcrLoadTests(unittest.TestCase):
    def test_no_ocr_leaves_the_calibrated_model_unchanged(self):
        base = SimV3Config(vn_share=0)
        a, rec = cloud_latency(base, 10)
        b, _ = cloud_latency(dataclasses.replace(base, ocr_ms_per_crop=0.0, ocr_crops_per_vehicle=3), 10)
        self.assertEqual(a, b)
        self.assertEqual((rec["downstream"], rec["ds_slow"], rec["ds_served"]), (0.0, 1.0, 1.0))

    def test_ocr_slows_detection_and_detection_slows_ocr(self):
        cfg = SimV3Config(vn_share=0, ocr_ms_per_crop=3.85, ocr_crops_per_vehicle=2)
        light, rec_light = cloud_latency(cfg, 1)
        heavy, rec_heavy = cloud_latency(cfg, 8)  # 6 cameras x 8 new vehicles x 2 crops = 96 crops/s
        self.assertGreater(heavy, light)
        self.assertAlmostEqual(rec_heavy["downstream"], 96 * 3.85 / 1000)
        self.assertGreater(rec_heavy["ds_slow"], rec_light["ds_slow"])
        self.assertGreaterEqual(rec_light["ds_slow"], 1.0)

    def test_ocr_falls_behind_when_detection_fills_the_gpu(self):
        cfg = SimV3Config(vn_share=0, ocr_ms_per_crop=3.85, ocr_crops_per_vehicle=3)
        _, rec = cloud_latency(cfg, 10)  # 180 crops/s x 3.85 ms > the GPU left by 6 D-FINE-X streams
        self.assertLess(rec["ds_served"], 1.0)
        _, rec = cloud_latency(dataclasses.replace(cfg, ocr_workers=1), 3)  # 54 crops/s: one worker, GPU room left
        self.assertEqual(rec["ds_served"], 1.0)


class LiveOcrModelTests(unittest.TestCase):
    """Fit to the testbed of 30/10 (sweep4/calib_extract.py): reads/s and workers -> busy workers, read time,
    D-FINE stretch 1 / (1 - .2 b)."""
    cfg = SimV3Config(vn_share=0, ocr_read_ms=50.0, ocr_crops_per_vehicle=.6)

    def test_busy_workers_and_read_time_match_the_lots(self):
        from drl.sim_v3 import ocr_busy
        for reads, workers, busy, read_ms in ((13, 4, .65, 52), (41, 4, 2.5, 62), (41, 8, 3.9, 95)):
            b, ms, served = ocr_busy(reads, workers, self.cfg)
            self.assertAlmostEqual(b, busy, delta=.25 * busy + .1)
            self.assertAlmostEqual(ms, read_ms, delta=.1 * read_ms)
            self.assertEqual(served, 1.0)
        b, ms, served = ocr_busy(80, 4, self.cfg)  # beyond what 4 workers can do: all busy, backlog grows
        self.assertEqual(b, 4.0)
        self.assertLess(served, 1.0)

    def test_detection_slows_with_busy_workers_and_off_is_unchanged(self):
        base = SimV3Config(vn_share=0)
        off, _ = cloud_latency(base, 10)
        self.assertEqual(off, cloud_latency(dataclasses.replace(base, ocr_crops_per_vehicle=3), 10)[0])
        few, rec_few = cloud_latency(dataclasses.replace(self.cfg, ocr_workers=4), 3)
        many, rec_many = cloud_latency(dataclasses.replace(self.cfg, ocr_workers=8), 10)  # 6 cams x 10 x .6 = 36 reads/s
        self.assertGreater(many, few)
        self.assertTrue(0 < rec_few["downstream"] < rec_many["downstream"] <= .2 * 8)  # GPU held = .2 x busy workers
        self.assertGreater(rec_many["ds_slow"], rec_few["ds_slow"])


class OcrBacklogTests(unittest.TestCase):
    def test_queue_grows_with_too_few_workers_and_drains_with_more(self):
        cfg = SimV3Config(vn_share=0, ocr_read_ms=50.0, ocr_crops_per_vehicle=1.0, ocr_backlog=True, ocr_workers=2)
        sim = SimV3(table(10), cfg, 0)  # 6 cameras x 10 new vehicles x 1 read = 60 reads/s > 2 workers (~28/s)
        sim.reset(6)
        secs = sim.step([("yolo11s_edge", 15)] * 6)
        grown = secs[-1][0]["ocr_queue"]
        self.assertGreater(grown, secs[0][0]["ocr_queue"])
        sim.ocr_workers = 8
        sim.ocr_crops = 0.0  # no new plates: the queue only drains
        secs = sim.step([("yolo11s_edge", 15)] * 6)
        self.assertLess(secs[-1][0]["ocr_queue"], grown)

    def test_backlog_off_is_unchanged(self):
        base = SimV3Config(vn_share=0, ocr_read_ms=50.0, ocr_crops_per_vehicle=.6)
        a, _ = cloud_latency(base, 3)
        b, rec = cloud_latency(dataclasses.replace(base, ocr_backlog=True), 3)
        self.assertEqual(a, b)
        self.assertEqual(rec["ocr_queue"], 0.0)


class FixedCameraTests(unittest.TestCase):
    def test_a_fixed_camera_cycles_its_own_location_only(self):
        rows = [{"sequence": s, "window": w, "split": "test", "model": m, "fps": f, "f1": .8}
                for s in ("A1", "A2", "B1", "C1", "D1") for w in range(8) for m in TABLE_NAMES for f in FPS_LEVELS]
        table = TableV3(rows, "test")
        table.locations = {"detrac": {("detrac", "A"): [("detrac", "A1"), ("detrac", "A2")],
                                      ("detrac", "B"): [("detrac", "B1")], ("detrac", "C"): [("detrac", "C1")],
                                      ("detrac", "D"): [("detrac", "D1")]}}
        for mode, at_most in (("location", 2), ("random", 5)):
            sim = SimV3(table, SimV3Config(vn_share=0, camera_mode=mode, scenarios=("surge", "diurnal"),
                                           surge_every_s=1.0), 1)
            sim.reset(6)
            seen = [set() for _ in range(6)]
            for _ in range(12):  # 60 s over 8 s sequences: many sequence ends
                sim.step([("yolo11s_edge", 15)] * 6)
                for i, s in enumerate(sim.streams):
                    seen[i].add(s.key[1])
            if mode == "location":
                self.assertTrue(all(len(v) <= at_most and len({k[0] for k in v}) == 1 for v in seen), seen)
            else:
                self.assertTrue(any(len({k[0] for k in v}) > 1 for v in seen), seen)


class Round27StateTests(unittest.TestCase):
    def test_probe_history_late_so_far_and_observation(self):
        rows = [{"sequence": "S0", "window": w, "split": "test", "model": m, "fps": f, "f1": .8}
                for w in range(30) for m in TABLE_NAMES for f in FPS_LEVELS]
        probes = [{"split": "test", "sequence": "S0", "second": w, "probe_missed": [w % 7, 3]} for w in range(30)]
        cfg = SimV3Config(vn_share=0, probe_period_s=5.0, probe_selective=True)
        sim = SimV3(TableV3(rows, "test", probe_rows=probes), cfg, 0)
        sim.reset(3)
        sim.probe_plan = {0: 2}  # only stream 0 probes, over 2 seconds
        sim.step([("yolo11s_edge", 15)] * 3)
        self.assertIsNotNone(sim.streams[0].probe_ema)
        self.assertIsNone(sim.streams[1].probe_ema)
        self.assertEqual(sim.streams[0].probe_n, 2)
        self.assertEqual([s.obs_t for s in sim.streams], [5, 5, 5])
        self.assertTrue(all(0 <= s.late_sum <= 5 for s in sim.streams))
        self.assertEqual(len(sim.reads_window), 5)
        base = sim.observe("probe_ocr2").shape[1]
        self.assertEqual(sim.observe("probe_ocr3_fps3c70p").shape[1], base + 6)


class DynamicsKeepDatasetTests(unittest.TestCase):
    def test_vn_share_holds_under_surge_and_diurnal(self):
        rows = [{"sequence": s, "window": w, "split": "test", "model": m, "fps": f, "f1": .8 if ds == "detrac" else .5,
                 "dataset": ds} for ds, s in (("detrac", "D1"), ("detrac", "D2"), ("vn_hard", "V1"), ("vn_hard", "V2"))
                for w in range(60) for m in TABLE_NAMES for f in FPS_LEVELS]
        table = TableV3(rows, "test")
        for keep, expect_all_vn in ((True, True), (False, False)):
            cfg = SimV3Config(vn_share=1.0, scenarios=("surge", "diurnal"), surge_every_s=1.0,
                              dynamics_keep_dataset=keep)
            sim = SimV3(table, cfg, 3)
            sim.reset(12)
            seen = set()
            for _ in range(6):
                sim.step([("yolo11s_edge", 15)] * 12)
                seen |= {s.key[0] for s in sim.streams}
            self.assertEqual(seen == {"vn_hard"}, expect_all_vn, (keep, seen))


if __name__ == "__main__":
    unittest.main()
