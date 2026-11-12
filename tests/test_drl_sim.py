"""drl.sim (multi-Edge simulator) and drl.mappo (trainer/evaluator) on a synthetic quality table."""
import dataclasses
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from drl.sim import CLOUD_MODES, MultiEdgeSim, QualityTable, SimConfig, jain, water_fill
from drl import mappo
from edge_agent.drl_policy import ACTION_MODES, OBS_V2, MlpActor

MODELS = ("yolov8n", "yolo11s", "rtdetrv2", "lwdetr")


def synthetic_rows(sequences=4, seconds=40, split="train"):
    """Night sequences: YOLO weak, DETR strong. Day sequences: every model good."""
    rows, scene = [], []
    for s in range(sequences):
        night = s % 2 == 0
        for w in range(seconds):
            for model, (day_f1, night_f1) in zip(MODELS, ((.9, .45), (.92, .6), (.93, .85), (.95, .9))):
                rows.append({"sequence": f"S{s}", "window": w, "split": split, "model": model,
                             "f1": night_f1 if night else day_f1, "vehicle_count": 12.0, "box_change": .2,
                             "small_ratio": .3})
            scene.append({"sequence": f"S{s}", "window": w, "brightness": 40.0 if night else 150.0, "motion": 8.0})
    return rows, scene


def table(**kw):
    rows, scene = synthetic_rows(**kw)
    return QualityTable(rows, scene)


class HelperTests(unittest.TestCase):
    def test_water_fill_is_max_min_fair(self):
        self.assertEqual(water_fill([.1, .2]), [.1, .2])
        alloc = water_fill([.1, .5, .6])
        self.assertAlmostEqual(alloc[0], .1)
        self.assertAlmostEqual(alloc[1], .45)
        self.assertAlmostEqual(alloc[2], .45)
        self.assertAlmostEqual(sum(water_fill([.9, .9, .9])), 1.0)

    def test_jain(self):
        self.assertAlmostEqual(jain([1, 1, 1]), 1.0)
        self.assertAlmostEqual(jain([1, 0]), .5)

    def test_quality_table_keeps_complete_seconds_of_the_split(self):
        rows, scene = synthetic_rows(sequences=1, seconds=3)
        rows = [r for r in rows if not (r["window"] == 1 and r["model"] == "lwdetr")]  # second 1 incomplete
        rows += [{**r, "split": "test", "sequence": "T"} for r in rows]
        t = QualityTable(rows, scene, split="train")
        self.assertEqual(t.sequences, {"S0": [0, 2]})
        self.assertEqual(t.cells[("S0", 0)][1], (40.0, 8.0))

    def test_subset_keeps_only_named_sequences(self):
        t = table(sequences=3, seconds=2)
        part = t.subset(["S1"])
        self.assertEqual(part.sequences, {"S1": [0, 1]})
        self.assertEqual({k[0] for k in part.cells}, {"S1"})
        self.assertEqual(len(t.sequences), 3)  # original untouched


class SimTests(unittest.TestCase):
    def test_observations_are_the_live_layout_and_bounded(self):
        sim = MultiEdgeSim(table(), seed=3)
        obs = sim.reset()
        self.assertEqual(obs.shape, (len(sim.edges), len(OBS_V2)))
        self.assertTrue(np.isfinite(obs).all() and (obs >= 0).all() and (obs <= 1).all())
        self.assertEqual(len(sim.global_state()), 4)

    def test_same_seed_same_episode(self):
        def run(seed):
            sim = MultiEdgeSim(table(), seed=seed)
            sim.reset()
            out = []
            for k in range(5):
                _obs, r, _done, _info = sim.step([k % 4] * len(sim.edges))
                out.append(r)
            return out
        self.assertEqual(run(7), run(7))

    def test_episode_ends_after_configured_decisions(self):
        sim = MultiEdgeSim(table(), SimConfig(episode_decisions=3), seed=1)
        sim.reset()
        dones = [sim.step([0] * len(sim.edges))[2] for _ in range(3)]
        self.assertEqual(dones, [False, False, True])

    def test_stale_cloud_masks_cloud_modes_and_falls_back(self):
        sim = MultiEdgeSim(table(), SimConfig(outage_p=0), seed=2)
        sim.reset()
        sim.outage_left = 50
        for mask in sim.masks():
            self.assertEqual([m in CLOUD_MODES for m in ACTION_MODES], [not ok for ok in mask])
        _obs, _r, _d, info = sim.step([ACTION_MODES.index("lwdetr_large_cloud")] * len(sim.edges))
        self.assertTrue(all(e["mode"] == "yolo_edge" for e in info["edges"]))

    def test_cloud_contention_lowers_timely_ratio(self):
        cfg = SimConfig(edges=(4, 4), cloud_background=(.15, .15), outage_p=0, random_profile_p=0, jitter=0)
        def timely(mode):
            sim = MultiEdgeSim(table(), dataclasses.replace(cfg), seed=4)
            sim.reset()
            parts = []
            for _ in range(4):
                parts += sim.step([ACTION_MODES.index(mode)] * 4)[3]["edges"]
            return np.mean([p["timely"] for p in parts])
        self.assertLess(timely("lwdetr_large_cloud"), timely("yolo_edge") - .2)

    def test_accuracy_deficit_is_penalized_by_lagrange(self):
        cfg = SimConfig(edges=(1, 1), outage_p=0, random_profile_p=0, a_min=.99, lagrange=10)
        sim = MultiEdgeSim(table(), cfg, seed=5)
        sim.reset()
        _o, reward, _d, info = sim.step([0])
        e = info["edges"][0]
        expected = (e["timely"] - 10 * (.99 - e["f1"]) - .2 * min(1, max(0, e["latency"] - 500) / 1000)
                    - e["loss"] - .05 * e["switch"])
        self.assertAlmostEqual(info["rewards"][0], expected)
        self.assertAlmostEqual(reward, expected)  # one Edge: Jain = 1

    def test_ta_reward_is_the_timely_accuracy_metric(self):
        cfg = SimConfig(edges=(1, 1), outage_p=0, random_profile_p=0, reward="ta", ta_deadline_ms=200)
        sim = MultiEdgeSim(table(), cfg, seed=5)
        sim.reset()
        _o, reward, _d, info = sim.step([ACTION_MODES.index("yolo11s_edge")])
        e = info["edges"][0]
        self.assertAlmostEqual(info["rewards"][0], e["ta"][200] - .05 * e["switch"])
        self.assertTrue(0 <= e["ta"][100] <= e["ta"][200] <= e["ta"][500] <= 1)

    def test_async_decisions_and_coordination_features(self):
        cfg = SimConfig(edges=(6, 6), outage_p=0, async_share=1 / 3, coord_features=True, priority_feature=True)
        sim = MultiEdgeSim(table(), cfg, seed=4)
        obs = sim.reset()
        self.assertEqual(obs.shape[1], len(OBS_V2) + 3)
        self.assertEqual(len(sim.turn), 2)
        before = [e.mode for e in sim.edges]
        movers = set(sim.turn)
        for i, mask in enumerate(sim.masks()):
            self.assertEqual(sum(mask), 4 if i in movers else 1)  # others may only keep their model
        sim.step([ACTION_MODES.index("rtdetr_cloud")] * 6)
        after = [e.mode for e in sim.edges]
        self.assertEqual([i for i in range(6) if after[i] != before[i]], sorted(movers))
        share, busy = sim.coordination()
        self.assertAlmostEqual(share, 2 / 6)
        self.assertTrue(0 <= busy <= 1)
        self.assertAlmostEqual(sim.observe()[0, -2], share)

    def test_fps_actions_use_per_fps_f1_and_scale_the_load(self):
        rows, scene = synthetic_rows(sequences=2, seconds=20)
        rows += [{**r, "fps": f, "f1": r["f1"] * (f / 15)} for r in rows for f in (5, 15)]
        cfg = SimConfig(edges=(1, 1), outage_p=0, random_profile_p=0, fps_levels=(5, 15))
        sim = MultiEdgeSim(QualityTable(rows, scene), cfg, seed=3)
        sim.reset()
        self.assertEqual(len(sim.actions), 8)
        self.assertEqual(sim.observe().shape[1], len(OBS_V2) + 1)  # + current FPS
        f1 = {}
        for fps in (15, 5):
            _o, _r, _d, info = sim.step([sim.action_index("yolo11s_edge", fps)])
            e = info["edges"][0]
            self.assertEqual(e["fps"], fps)
            f1[fps] = e["f1"]
            self.assertAlmostEqual(sim.edges[0].last["lam"], fps)
        self.assertLess(f1[5], f1[15])

    def test_within_episode_dynamics_only_when_enabled(self):
        def trace(**dyn):
            sim = MultiEdgeSim(table(), SimConfig(edges=(3, 3), outage_p=0, **dyn), seed=9)
            sim.reset()
            seen = set()
            for _ in range(60):
                sim._tick()
                seen.add((round(sim.background, 6), round(sim.network_ms, 6), tuple(e.enc_mult for e in sim.edges)))
            return seen
        self.assertEqual(len(trace()), 1)
        self.assertGreater(len(trace(dyn_bg_p=.2, dyn_enc_p=.2, dyn_net_p=.2)), 5)

    def test_saturated_gpu_latency_follows_the_closed_loop(self):
        # 12 LW-DETR streams, 2 tickets each: one ticket waits for all 24 -> >= 24 x 1000 / 95 ms (live p50 266).
        sim = MultiEdgeSim(table(), SimConfig(edges=(12, 12), outage_p=0, random_profile_p=0, jitter=0,
                                              cloud_background=(1, 1), network_ms=(3, 3), wan_p=0), seed=2)
        sim.reset()
        sim.step([ACTION_MODES.index("lwdetr_large_cloud")] * 12)
        self.assertGreaterEqual(min(e.last["latency"] for e in sim.edges), 24 * 1000 / 95)

    def test_edge_cpu_contention_slows_local_models_only_when_enabled(self):
        def trace(**dyn):
            sim = MultiEdgeSim(table(), SimConfig(edges=(3, 3), outage_p=0, **dyn), seed=9)
            sim.reset()
            for _ in range(60):
                sim._tick()
            return [e.cap_mult for e in sim.edges]
        self.assertEqual(trace(), [1.0] * 3)
        mults = trace(dyn_edge_p=.2, edge_swing=(.5, .9))
        self.assertTrue(all(.5 <= m <= .9 for m in mults))

    def test_int8_delta_applies_to_npu_edge_modes_only(self):
        def f1(seed, action, delta):
            cfg = SimConfig(edges=(1, 1), random_profile_p=0, outage_p=0,
                            int8_delta={"yolo_edge": delta, "rtdetr_cloud": delta})
            sim = MultiEdgeSim(table(), cfg, seed=seed)
            sim.reset()
            return sim.edges[0].p.npu, sim.step([action])[3]["edges"][0]["f1"]
        seen = set()
        for seed in range(30):
            npu, plain = f1(seed, 0, 0.0)
            _, shifted = f1(seed, 0, -.1)
            _, cloud_plain = f1(seed, 2, 0.0)
            _, cloud_shifted = f1(seed, 2, -.1)
            self.assertAlmostEqual(plain - shifted, .1 if npu else 0.0)
            self.assertAlmostEqual(cloud_plain, cloud_shifted)  # Cloud models never run INT8
            seen.add(npu)
        self.assertEqual(seen, {True, False})


class CalibrationTests(unittest.TestCase):
    # Measured timely ratio per Edge x mode, control-v1 explore on v45 (results/tableB_v45b/TABLE_B.md).
    MEASURED = {("edge-246", "yolo_edge"): .994, ("edge-246", "rtdetr_cloud"): .840,
                ("edge-246", "lwdetr_large_cloud"): .735, ("edge-247", "yolo11s_edge"): 1.0,
                ("edge-247", "rtdetr_cloud"): .983, ("edge-252", "yolo_edge"): .823,
                ("edge-252", "rtdetr_cloud"): .320, ("edge-252", "lwdetr_large_cloud"): .296}

    # Two Cloud tickets in flight (v47/v48, results/tableB_v47_clean); 252 omitted (its encoder share varied).
    MEASURED_PAIRED = {("edge-246", "rtdetr_cloud"): .986, ("edge-246", "lwdetr_large_cloud"): .982,
                       ("edge-247", "rtdetr_cloud"): .989, ("edge-247", "lwdetr_large_cloud"): .984}

    def test_real_edges_reproduce_measured_timely_ratio(self):
        for inflight, measured_by in ((1, self.MEASURED), (2, self.MEASURED_PAIRED)):
            self.check(SimConfig(outage_p=0, cloud_inflight=inflight), measured_by)

    def check(self, cfg, measured_by):
        import random
        from drl.sim import REAL_EDGES, _Edge
        profiles = {p.name: p for p in REAL_EDGES}
        for (edge, mode), measured in measured_by.items():
            sim = MultiEdgeSim(table(), cfg, seed=0)
            sim.reset()
            sim.background, sim.network_ms = 1.0, 3.0
            sim.edges = [_Edge(profiles[edge], random.Random(0), sim.table, cfg)]
            sim.edges[0].mode = mode
            ticks = []
            for _ in range(10):
                sim._tick()
                ticks.append(sim.edges[0].last["timely"] / cfg.target_fps)
            self.assertAlmostEqual(np.mean(ticks), measured, delta=.06, msg=f"{edge} {mode}")


class EvaluateTests(unittest.TestCase):
    def test_baselines_run_and_report_metrics(self):
        result = mappo.evaluate(table(), policy="always_yolo_edge", cfg=SimConfig(episode_decisions=4), episodes=2)
        self.assertEqual(result["cloud_share"], 0.0)
        self.assertGreater(result["timely"], .5)
        rules = mappo.evaluate(table(), policy="rules", cfg=SimConfig(episode_decisions=4), episodes=2)
        self.assertIn("below_a_min", rules)

    def test_new_baselines_on_the_real_edge_scenario(self):
        cfg = SimConfig(episode_decisions=4, real_edges_only=True, outage_p=0, a_min=.8)
        results = {p: mappo.evaluate(table(), policy=p, cfg=cfg, episodes=2)
                   for p in ("always_yolo_edge", "rules_acc", "best_fixed", "oracle")}
        self.assertEqual(set(results["oracle"]["per_edge"]), {"edge-246", "edge-247", "edge-252"})
        # Night sequences: YOLO falls below A_min; the oracle (knows F1) must do no worse on the shortfall.
        self.assertLessEqual(results["oracle"]["f1_deficit"], results["always_yolo_edge"]["f1_deficit"])
        # rules_acc keeps the weak-encoder Edge (252) off the Cloud.
        self.assertEqual(results["rules_acc"]["per_edge"]["edge-252"]["cloud_share"], 0.0)
        for r in results.values():
            self.assertTrue(0 <= r["jain"] <= 1 and r["latency_p95_ms"] >= 0)


    def test_rules_fps_steps_down_the_ladder_when_late_and_up_when_on_time(self):
        sim = MultiEdgeSim(table(), SimConfig(edges=(1, 1), outage_p=0, fps_levels=(5, 10, 15)), seed=1)
        sim.reset()
        e, mask = sim.edges[0], sim.masks()[0]
        e.mode, e.fps = "lwdetr_large_cloud", 15
        self.assertEqual(sim.actions[mappo._rules_fps(e, .5, mask, sim)], ("lwdetr_large_cloud", 10))
        self.assertEqual(sim.actions[mappo._rules_fps(e, .93, mask, sim)], ("lwdetr_large_cloud", 15))
        e.mode, e.fps = "yolo11s_edge", 15
        self.assertEqual(sim.actions[mappo._rules_fps(e, 1.0, mask, sim)], ("lwdetr_large_cloud", 5))
        no_cloud = [not (m in CLOUD_MODES) for m, _f in sim.actions]
        self.assertEqual(sim.actions[mappo._rules_fps(e, 1.0, no_cloud, sim)], ("yolo11s_edge", 15))
        r = mappo.evaluate(table(), policy="rules_fps", cfg=SimConfig(episode_decisions=4, fps_levels=(5, 10, 15)),
                           episodes=1)
        self.assertIn("ta_200", r)


class RegimeAndSequentialTests(unittest.TestCase):
    def weather_table(self):
        rows, scene = synthetic_rows(sequences=4, seconds=30)
        for r in rows:
            r["weather"] = "night" if int(r["sequence"][1:]) % 2 == 0 else "sunny"
        return QualityTable(rows, scene)

    def test_cameras_keep_their_scene_regime_and_wan_sites_add_delay(self):
        t = self.weather_table()
        self.assertEqual(t.regimes(), {"night": ["S0", "S2"], "sunny": ["S1", "S3"]})
        same = dict(edges=(6, 6), outage_p=0, random_profile_p=0, jitter=0, encode_jitter=0, network_ms=(3, 3), wan_p=0)
        sim = MultiEdgeSim(t, SimConfig(**same, scene_regime=True, regime_switch_p=0, edge_wan_p=1.0,
                                        wan_ms=(50, 50)), seed=4)
        sim.reset()
        start = [e.regime for e in sim.edges]
        for _ in range(120):  # sequences end and jump: always within the camera's regime
            sim._tick()
            for e in sim.edges:
                self.assertEqual(t.weather[e.sequence], e.regime)
        self.assertEqual(start, [e.regime for e in sim.edges])
        self.assertTrue(all(e.net_ms == 50 for e in sim.edges))
        sim.step([sim.action_index("rtdetr_cloud")] * 6)
        lan = MultiEdgeSim(t, SimConfig(**same), seed=4)
        lan.reset()
        lan.step([lan.action_index("rtdetr_cloud")] * 6)
        for w, l in zip(sim.edges, lan.edges):
            self.assertEqual(w.p.name, l.p.name)
            self.assertGreater(w.last["latency"], l.last["latency"] + 30)  # +47 ms WAN, less GPU queueing

    def test_sequential_choices_see_the_gpu_already_committed(self):
        from drl import seq
        sim = MultiEdgeSim(self.weather_table(), seq._cfg(SimConfig(edges=(4, 4), outage_p=0, fps_levels=(5, 10, 15))),
                           seed=2)
        seen = []
        def choose(obs, mask):
            seen.append(obs[-7])  # committed GPU share so far (/2)
            return sim.action_index("lwdetr_large_cloud", 15)
        rounds, parts = seq.run_episode(sim, choose)
        first = [s[0][-7] for s in rounds[0][0]]
        self.assertEqual(first[0], 0.0)
        self.assertAlmostEqual(first[1], 15 / 95 / 2, places=5)
        self.assertTrue(all(b > a for a, b in zip(first, first[1:])))
        self.assertTrue(all(p["mode"] == "lwdetr_large_cloud" for p in parts))

    def test_imitation_labels_are_the_central_split(self):
        import random
        from drl import central, seq
        sim = MultiEdgeSim(self.weather_table(), seq._cfg(SimConfig(edges=(5, 5), outage_p=0, fps_levels=(5, 10, 15))),
                           seed=3)
        acts = central.central_actions((5, 10, 15))
        k3 = acts.index(("lwdetr_large_cloud", 15, 3))
        data = seq.imitation_data(sim, lambda o, m: k3, 1, random.Random(0))
        first = data[:5]
        cloud = [sim.actions[d[2]] == ("lwdetr_large_cloud", 15) for d in first]
        self.assertEqual(sum(cloud), 3)
        self.assertTrue(all(sim.actions[d[2]] in (("lwdetr_large_cloud", 15), ("yolo11s_edge", 15)) for d in data))
        self.assertEqual(len(first[0][0]), len(seq.camera_obs(sim, 0, 0.0, 0, 0)))


class BenefitTests(unittest.TestCase):
    def test_cloud_benefit_follows_the_link_and_the_edge_load(self):
        from drl import central
        sim = MultiEdgeSim(table(), SimConfig(edges=(2, 2), outage_p=0, random_profile_p=0, jitter=0, encode_jitter=0,
                                              network_ms=(3, 3), wan_p=0), seed=5)
        sim.reset()
        e = sim.edges[0]
        base = central.cloud_benefit(sim, e)
        e.net_ms = 150  # WAN/4G site: Cloud results come late
        self.assertLess(central.cloud_benefit(sim, e), base - .2)
        e.net_ms, e.cap_mult = None, .05  # Edge CPU swamped: YOLO11s locally misses the deadline
        self.assertGreater(central.cloud_benefit(sim, e), base)
        acts = central.central_actions((5, 10, 15))
        sim.edges[1].cap_mult = 1.0
        idx = central.apply(sim, ("lwdetr_large_cloud", 15, 1), rank="benefit")
        self.assertEqual(sim.actions[idx[0]], ("lwdetr_large_cloud", 15))  # the overloaded Edge gets the slot
        self.assertEqual(len(acts), 13 * 2 * 3)


class CentralTests(unittest.TestCase):
    def test_central_action_puts_the_k_strongest_encoders_on_the_cloud(self):
        from drl import central
        cfg = SimConfig(edges=(5, 5), outage_p=0, fps_levels=(5, 10, 15))
        sim = MultiEdgeSim(table(), central._central_cfg(cfg), seed=3)
        sim.reset()
        idx = central.apply(sim, ("rtdetr_cloud", 10, 2))
        on_cloud = [i for i, a in enumerate(idx) if sim.actions[a] == ("rtdetr_cloud", 10)]
        self.assertEqual(len(on_cloud), 2)
        weakest = min(sim.edges[i].enc_fps() for i in on_cloud)
        self.assertTrue(all(sim.edges[i].enc_fps() <= weakest for i in range(5) if i not in on_cloud))
        self.assertTrue(all(sim.actions[a] == central.LOCAL for i, a in enumerate(idx) if i not in on_cloud))
        acts = central.central_actions((5, 10, 15))
        mask = central.central_mask(acts, 5, stale=True)
        self.assertTrue(all(m == (k == 0) for m, (_mo, _f, k) in zip(mask, acts)))
        r = central.evaluate_central(table(), lambda o, m: int(np.flatnonzero(m)[-1]),
                                     SimConfig(edges=(3, 3), episode_decisions=3, fps_levels=(5, 10, 15)), episodes=1)
        self.assertGreater(r["cloud_share"], 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class TrainSmokeTests(unittest.TestCase):
    def test_train_exports_an_actor_the_edge_accepts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "actor.json"
            mappo.train(table(), out, iters=2, episodes=2, epochs=1, cfg=SimConfig(episode_decisions=3),
                        log=lambda _: None, eval_table=None)
            actor = MlpActor.load(out)
        self.assertEqual(actor.actions, ACTION_MODES)
        self.assertIn(actor.act(np.zeros(len(OBS_V2)), [True] * 4), range(4))

    def test_priority_actor_is_simulator_only(self):
        cfg = SimConfig(episode_decisions=3, priority_feature=True)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "p.json"
            mappo.train(table(), out, iters=1, episodes=2, epochs=1, cfg=cfg, log=lambda _: None)
            with self.assertRaises(ValueError):
                MlpActor.load(out)  # the Edge runtime refuses the extra input
            actor = mappo.load_actor(out)
        sim = MultiEdgeSim(table(), cfg, seed=1)
        obs = sim.reset()
        self.assertEqual(obs.shape[1], len(OBS_V2) + 1)
        self.assertIn(actor.act(obs[0], [True] * 4), range(4))

    def test_lagrange_rises_while_accuracy_floor_is_missed_and_falls_when_met(self):
        for a_min, rises in ((1.0, True), (0.0, False)):
            cfg = SimConfig(episode_decisions=3, a_min=a_min, lagrange=1.0)
            with tempfile.TemporaryDirectory() as tmp:
                mappo.train(table(), Path(tmp) / "a.json", iters=2, episodes=2, epochs=1, cfg=cfg, log=lambda _: None)
            self.assertEqual(cfg.lagrange > 1.0, rises)


if __name__ == "__main__":
    unittest.main()


class SimProbeTests(unittest.TestCase):
    """docs/PLAN_PROBE_VI.md P5: probe fields replay the offline probe table; off -> absent; outage -> dropped."""

    def table(self):
        rows = [{"sequence": "S", "window": s, "model": m, "f1": .8, "split": "train", "weather": "sunny"}
                for s in range(40) for m in ("yolov8n", "yolo11s", "rtdetrv2", "lwdetr")]
        probes = [{"sequence": "S", "second": s, "probe_missed": [3, 5]} for s in range(40)]
        signals = [{"sequence": "S", "second": s, "conf_mean": .7, "uncertain_ratio": .1} for s in range(40)]
        scene = [{"sequence": "S", "window": s, "brightness": 100., "motion": 3., "contrast": 40., "laplacian_small": 900.}
                 for s in range(40)]  # live scale: edge_agent.resources_v2.scene_features
        from drl.sim import QualityTable
        return QualityTable(rows, scene, "train", probes, signals)

    def run_sim(self, **cfg):
        from drl.sim import MultiEdgeSim, SimConfig
        sim = MultiEdgeSim(self.table(), SimConfig(edges=(1, 1), outage_p=0.0, **cfg), seed=0)
        sim.reset()
        sim.edges[0].mode = "yolo11s_edge"
        raws = []
        for _ in range(20):
            sim._tick()
            raws.append(sim.edges[0].history[-1]["raw"])
        return sim, raws

    def test_probe_cadence_values_and_age(self):
        _sim, raws = self.run_sim(probe_period_s=4, probe_frames=2)   # one probe every 2 s
        probed = [r["probe_missed"] for r in raws if r["probe_missed"] is not None]
        self.assertEqual(len(probed), 10)
        self.assertTrue(set(probed) <= {3.0, 5.0})
        self.assertTrue(all(r["probe_age_s"] in (0.0, 1.0) for r in raws[2:]))
        self.assertEqual((raws[-1]["contrast"], raws[-1]["laplacian_small"], raws[-1]["uncertain_ratio"]),
                         (40., 900., .1))  # variant B features

    def test_probe_off_and_outage(self):
        _sim, raws = self.run_sim()
        self.assertTrue(all(r["probe_missed"] is None and r["probe_age_s"] is None for r in raws))
        sim, _ = self.run_sim(probe_period_s=2)
        sim.outage_left = 50  # Cloud unreachable: no probe lands, age keeps growing
        ages = []
        for _ in range(5):
            sim._tick()
            ages.append(sim.edges[0].history[-1]["raw"]["probe_age_s"])
        self.assertEqual(ages, sorted(ages))
        self.assertTrue(all(sim.edges[0].history[-k]["raw"]["probe_missed"] is None for k in range(1, 6)))
