"""Control v1 with the four benchmarked models: controller policies, DRL actor, real executor."""
import asyncio
import io
import json
from pathlib import Path
import tempfile
from time import monotonic_ns, time_ns
from types import SimpleNamespace
import unittest

import grpc
import numpy as np

from contracts import Box, FrameIdentity, MediaFrameRef, ModelMode
from contracts.control_client_v1 import GrpcControlClient
from contracts.proto import edge_video_platform_pb2_grpc as pb_grpc
from cloud_core import CloudFrameCache, DecodedMediaFrame, ResultHub
from cloud_core.grpc_service import CloudDataPlaneServicer
from cloud_core.control_v1 import CloudTelemetry, ControlService, install_grpc, sample_forever
from edge_agent.drl_policy import ACTION_MODES, FORMAT, HISTORY, OBS_V2, MlpActor, observation_v2, write_actor
from edge_agent.state_v1 import Action, RuleController, TimeWindowMonitor

Y8, Y11 = ModelMode.YOLO_EDGE, ModelMode.YOLO11S_EDGE
RT, LW = ModelMode.RTDETR_CLOUD, ModelMode.LWDETR_CLOUD
FOUR = (Y8, Y11, RT, LW)
KEY = ('edge-a', 'camera', 'control', 1)


class Clock:
    def __init__(self): self.ns = 0
    def __call__(self): return self.ns
    def advance(self, ms): self.ns += round(ms * 1e6)


def feedback(sequence=1, pressure=.2, inflight=.3):
    return dict(control_v1=1, cloud_instance_id='cloud', sequence=sequence, window_ms=1000.,
                measurement_to_send_ms=5., cloud_inflight_norm=inflight,
                cloud_arrival_pressure_raw=pressure, cloud_arrival_pressure_norm=min(pressure, 1),
                unknown_capacity_tickets=0, arrival_rate_rps=5.)


def snapshot(clock, controller, *, fresh=True, pressure=.2, inflight=.3, backlog=0.):
    """A finished window; fresh=False leaves Cloud feedback missing (stale)."""
    monitor = TimeWindowMonitor(KEY, clock=clock, capability=CAP)
    if fresh:
        monitor.feedback.accept(feedback(pressure=pressure, inflight=inflight))
    monitor.queue(round(backlog * 8), 8)
    clock.advance(1000)
    return monitor.finish(controller.action)


CAP = {"yolov8n_fps": 101.6, "yolo11s_fps": 86.3, "encode_fps": 66.6}


def actor_spec(modes, logits):
    """Linear actor whose output equals fixed logits regardless of the observation."""
    return {"format": FORMAT, "observation": list(OBS_V2), "actions": [m.value for m in modes],
            "layers": [{"w": np.zeros((len(OBS_V2), len(modes))).tolist(), "b": list(logits), "act": "linear"}]}


class ControllerModesTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()

    def make(self, policy, initial=RT, modes=FOUR, levels=(15,), **kw):
        return RuleController(initial, levels[0], allowed_modes=modes, fps_levels=levels, stream_key=KEY,
                              policy=policy, cooldown_s=1, clock=self.clock, **kw)

    def test_accepts_four_models_and_orders_actions(self):
        c = self.make('observe', levels=(10, 15))
        self.assertEqual((c.edge_modes, c.cloud_modes), ((Y8, Y11), (RT, LW)))
        self.assertEqual(c.actions[:3], ((Y8, 10.), (Y8, 15.), (Y11, 10.)))
        self.assertEqual(len(c.actions), 8)

    def test_rejects_invalid_configuration(self):
        with self.assertRaises(ValueError):
            self.make('rules', modes=(Y8, ModelMode.RTMDET_EDGE), initial=Y8)
        with self.assertRaises(ValueError):
            self.make('rules', modes=(RT, RT))
        with self.assertRaises(ValueError):
            self.make('greedy')
        with self.assertRaises(ValueError):
            self.make('drl')  # no actor
        wrong = MlpActor(actor_spec((Y8, RT), (0, 0)))
        with self.assertRaises(ValueError):
            self.make('drl', actor=wrong)
        with self.assertRaisesRegex(ValueError, 'one FPS level'):
            self.make('drl', levels=(10, 15), actor=MlpActor(actor_spec(FOUR, (0, 0, 0, 0))))

    def test_stale_cloud_falls_back_to_first_edge_mode_even_in_cooldown(self):
        c = self.make('explore', initial=LW, modes=(Y11, Y8, RT, LW), seed=1)
        action, reason = c.decide(snapshot(self.clock, c, fresh=False), now=c.action.since_ns + 1)
        self.assertEqual((action.mode, reason), (Y11, 'cloud_feedback_stale'))

    def test_rules_generalize_over_placements(self):
        c = self.make('rules', initial=LW)
        action, reason = c.decide(snapshot(self.clock, c, pressure=.95))
        self.assertEqual((action.mode, reason), (Y8, 'cloud_busy'))
        c = self.make('rules', initial=Y11)
        action, reason = c.decide(snapshot(self.clock, c, backlog=.75))
        self.assertEqual((action.mode, reason), (RT, 'local_backlog'))

    def test_cycle_visits_all_four_and_holds_on_stale_cloud(self):
        c = self.make('cycle', initial=Y8)
        seen = [c.action.mode]
        for _ in range(4):
            seen.append(c.decide(snapshot(self.clock, c))[0].mode)
        self.assertEqual(seen, [Y8, Y11, RT, LW, Y8])
        c = self.make('cycle', initial=Y11)
        action, reason = c.decide(snapshot(self.clock, c, fresh=False))
        self.assertEqual((action.mode, reason), (Y11, 'cloud_feedback_stale'))

    def test_explore_is_seeded_covering_and_never_repeats(self):
        def run(seed):
            c = self.make('explore', initial=Y8, seed=seed)
            out = []
            for _ in range(40):
                before = c.action
                action, reason = c.decide(snapshot(self.clock, c))
                self.assertEqual(reason, 'explore')
                self.assertNotEqual(action.mode, before.mode)
                self.assertNotEqual(action.decision_id, before.decision_id)
                out.append(action.mode)
            return out
        first = run(7)
        self.assertEqual(first, run(7))
        self.assertEqual(set(first), set(FOUR))

    def test_explore_respects_cooldown_and_masks_cloud_when_stale(self):
        c = self.make('explore', initial=Y8, seed=3)
        self.assertEqual(c.decide(snapshot(self.clock, c), now=c.action.since_ns + 1)[1], 'cooldown')
        for _ in range(20):
            action, _ = c.decide(snapshot(self.clock, c, fresh=False))
            self.assertIn(action.mode, (Y8, Y11))
        only_edge = self.make('explore', initial=Y8, modes=(Y8, RT), seed=3)
        self.assertEqual(only_edge.decide(snapshot(self.clock, only_edge, fresh=False))[1], 'no_alternative')

    def test_drl_follows_actor_mask_and_holds(self):
        prefer_lw = MlpActor(actor_spec(FOUR, (0, 1, 2, 3)))
        c = self.make('drl', initial=Y8, actor=prefer_lw)
        action, reason = c.decide(snapshot(self.clock, c))
        self.assertEqual((action.mode, reason), (LW, 'drl'))
        c.action = Action(Y8, 15., 'edge', self.clock())
        self.clock.advance(2000)
        action, reason = c.decide(snapshot(self.clock, c, fresh=False))  # Cloud masked -> best Edge
        self.assertEqual((action.mode, reason), (Y11, 'drl'))
        decision = action.decision_id
        prefer_current = MlpActor(actor_spec(FOUR, (0, 9, 0, 0)))
        c.actor = prefer_current
        self.clock.advance(2000)
        action, reason = c.decide(snapshot(self.clock, c))
        self.assertEqual((action.decision_id, reason), (decision, 'drl'))  # hold keeps identity


def window(raw=None, *, mode="yolo_edge", target=15., media=15., capability=CAP):
    base = {name: None for name in ("timely_fps", "active_path_latency_ms", "transport_overhead_ms",
            "edge_backlog_norm", "cloud_arrival_pressure_norm", "box_change_mean", "cpu_max_core_pct",
            "temp_c", "encoder_write_ms", "brightness", "motion", "vehicle_count")}
    return {"raw": {**base, **(raw or {})}, "context": {"model_mode": mode, "target_fps": target},
            "media_fps": media, "capability": capability}


class ObservationV2Tests(unittest.TestCase):
    def test_layout_is_22_values_in_documented_order(self):
        self.assertEqual(len(OBS_V2), 22)
        self.assertEqual(OBS_V2[:3], ("cap_yolov8n", "cap_yolo11s", "cap_encode"))
        self.assertEqual(OBS_V2[-4:], tuple(f"mode_{m}" for m in ACTION_MODES))

    def test_normalization_bits_and_onehot(self):
        raw = {"timely_fps": 12., "active_path_latency_ms": 250., "transport_overhead_ms": 30.,
               "edge_backlog_norm": .25, "cloud_arrival_pressure_norm": .9, "box_change_mean": .2,
               "cpu_max_core_pct": 80., "temp_c": 70., "encoder_write_ms": 40., "brightness": 51.,
               "motion": 12.75, "vehicle_count": 10.}
        o = dict(zip(OBS_V2, observation_v2([window(raw)], "rtdetr_cloud")))
        expected = {"cap_yolov8n": 101.6 / 300, "cap_encode": 66.6 / 300, "cpu_max_core": .8, "temp": .7,
                    "temp_ok": 1., "edge_backlog": .25, "encoder_load": .6, "transport": .3, "transport_ok": 1.,
                    "cloud_pressure": .9, "timely_ratio": .8, "path_latency": .25, "brightness": .2,
                    "motion": .2, "vehicle_count": .2, "box_change": .2, "box_change_ok": 1.,
                    "mode_rtdetr_cloud": 1., "mode_yolo_edge": 0.}
        for name, value in expected.items():
            self.assertAlmostEqual(o[name], value, msg=name)

    def test_missing_optional_values_clip_and_worst_latency(self):
        o = dict(zip(OBS_V2, observation_v2([window({"encoder_write_ms": 500., "vehicle_count": 400.})],
                                            "yolo_edge")))
        self.assertEqual((o["temp"], o["temp_ok"], o["transport_ok"], o["box_change_ok"]), (0., 0., 0., 0.))
        self.assertEqual((o["encoder_load"], o["vehicle_count"], o["timely_ratio"]), (1., 1., 0.))
        self.assertEqual(o["path_latency"], 1.)  # nothing completed is not an unknown-good latency

    def test_history_mean_ignores_missing_and_keeps_last_five(self):
        windows = ([window({"temp_c": 90.}), window({"temp_c": None}), window({"temp_c": 50.})]
                   + [window({"temp_c": 70.}) for _ in range(3)])
        o = dict(zip(OBS_V2, observation_v2(windows, "yolo_edge")))
        self.assertEqual(HISTORY, 5)
        self.assertAlmostEqual(o["temp"], (50 + 70 * 3) / 4 / 100)  # oldest dropped, None skipped

    def test_rejects_missing_capability_unknown_mode_or_empty(self):
        with self.assertRaisesRegex(ValueError, 'capability'):
            observation_v2([window(capability={})], "yolo_edge")
        with self.assertRaises(ValueError):
            observation_v2([window()], "dfine_cloud")
        with self.assertRaises(ValueError):
            observation_v2([], "yolo_edge")

    def test_live_window_feeds_observation(self):
        clock = Clock()
        monitor = TimeWindowMonitor(KEY, clock=clock, fps_cap=15, capability=CAP)
        for name, value in (("brightness", 100.), ("encoder_write_ms", 20.), ("cpu_max_core_pct", 50.),
                            ("vehicle_count", 5), ("small_ratio", .4), ("temp_c", 61.)):
            monitor.add(name, value)
        clock.advance(1000)
        snap = monitor.finish(Action(Y8, 15., 'd', 0))
        self.assertEqual((snap["media_fps"], snap["capability"], snap["raw"]["small_ratio"]), (15, CAP, .4))
        o = dict(zip(OBS_V2, observation_v2([snap], "yolo_edge")))
        self.assertAlmostEqual(o["encoder_load"], .3)
        self.assertAlmostEqual(o["cpu_max_core"], .5)
        with self.assertRaises(ValueError):
            monitor.add("small_ratio", 1.5)


class ActorTests(unittest.TestCase):
    def test_mlp_forward_tiebreak_and_mask(self):
        width = len(OBS_V2)
        w1 = np.zeros((width, 2)); w1[0, 0] = 1.
        spec = {"format": FORMAT, "observation": list(OBS_V2),
                "actions": ["yolo_edge", "rtdetr_cloud", "lwdetr_large_cloud"],
                "layers": [{"w": w1.tolist(), "b": [0, 0], "act": "relu"},
                           {"w": [[0, 1, 1], [0, 0, 0]], "b": [0, 0, 0], "act": "linear"}]}
        actor = MlpActor(spec)
        obs = np.zeros(width); obs[0] = 2.
        self.assertEqual(list(actor.logits(obs)), [0., 2., 2.])
        self.assertEqual(actor.act(obs, [True, True, True]), 1)  # tie -> lowest index
        self.assertEqual(actor.act(obs, [True, False, True]), 2)
        with self.assertRaises(ValueError):
            actor.act(obs, [False, False, False])
        with self.assertRaises(ValueError):
            actor.act(obs, [True, True])

    def test_invalid_actor_files_rejected(self):
        good = actor_spec((Y8, RT), (0, 0))
        MlpActor(good)
        bad = [dict(good, format="edge-drl-actor-v1"), dict(good, observation=list(OBS_V2)[:-1]),
               dict(good, actions=["yolo_edge", "yolo_edge"]), dict(good, actions=["dfine_cloud", "yolo_edge"]),
               dict(good, layers=[]), dict(good, layers=[dict(good["layers"][0], b=[0])]),
               dict(good, layers=[dict(good["layers"][0], act="sigmoid")]),
               dict(good, layers=[dict(good["layers"][0], b=[float("nan"), 0])])]
        for spec in bad:
            with self.assertRaises((ValueError, KeyError)):
                MlpActor(spec)

    def test_write_actor_round_trip(self):
        actions = ("yolo_edge", "rtdetr_cloud")
        width = len(OBS_V2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "actor.json"
            write_actor(path, actions, [(np.zeros((width, 2)), [0., 1.], "linear")])
            actor = MlpActor.load(path)
            self.assertEqual(actor.actions, actions)
            self.assertEqual(actor.act(np.zeros(width), [True, True]), 1)
            with self.assertRaises(ValueError):
                write_actor(path, actions, [(np.zeros((width, 3)), [0., 0., 0.], "linear")])


class Tracker:
    def update(self, result):
        from dataclasses import replace
        return replace(result, boxes=tuple(replace(box, track_id=1) for box in result.boxes))
    def reset(self, key): pass


class Detector:
    runtime = 'fake-test'
    def __init__(self, model_id): self.model_id = model_id
    def infer(self, pixels):
        return (Box(.1, .1, .5, .6, .9, 2, 'car'),), 1.


def identity(i):
    return FrameIdentity(*KEY[:3], KEY[3], i, time_ns(), i * 33_333_333)


class FourModeExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        cache = CloudFrameCache()
        class Tap:
            async def publish_reference(self, ref):
                await cache.announce(ref)
                await cache.ingest(DecodedMediaFrame(stream_key=ref.frame.stream_key,
                    encoded_seq=ref.encoded_seq, media_pts_ns=ref.media_pts_ns,
                    pixels=np.zeros((8, 8, 3), dtype=np.uint8), received_utc_ns=time_ns()))
        self.core = CloudDataPlaneServicer(hub=ResultHub(tracker=Tracker()), frame_cache=cache, frame_taps=Tap(),
            cloud_adapters={RT: Detector('rt-artifact'), LW: Detector('lw-artifact')}, max_cloud_inflight=2)
        self.telemetry = CloudTelemetry(self.core, {'rt-artifact': 170, 'lw-artifact': 85}, window_ms=100)
        self.core.control_ticket_observer = self.telemetry
        self.service = ControlService(self.core, self.telemetry)
        self.server = grpc.aio.server()
        pb_grpc.add_EdgeCloudDataPlaneServicer_to_server(self.core, self.server)
        install_grpc(self.server, self.service)
        port = self.server.add_insecure_port('127.0.0.1:0')
        await self.server.start()
        self.client = GrpcControlClient(f'127.0.0.1:{port}', timeout_s=2)
        await self.client.start()
        self.log = io.StringIO()
        self.sampler = asyncio.create_task(sample_forever(self.telemetry, log=self.log))

    async def asyncTearDown(self):
        self.sampler.cancel()
        await asyncio.gather(self.sampler, return_exceptions=True)
        await self.client.close()
        await self.server.stop(0)
        await self.service.close()

    def publisher(self):
        class Publisher:
            profile = SimpleNamespace(fps=30)
            frames_submitted = padding_frames_submitted = 0
            async def __aenter__(self): return self
            async def __aexit__(self, *args): pass
            async def publish(self, captured):
                ref = MediaFrameRef(captured.identity, self.frames_submitted,
                                    self.frames_submitted * 33_333_333, time_ns())
                self.frames_submitted += 1
                await self.on_frame_submitted(ref)
            async def publish_padding(self, captured): self.padding_frames_submitted += 1
        return Publisher()

    def source(self, n):
        from edge_agent.capture import CapturedFrame
        class Source:
            async def __aiter__(self):
                for i in range(n):
                    yield CapturedFrame(identity=identity(i), pixels=np.zeros((16, 16, 3), dtype=np.uint8),
                                        capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(1 / 30)
        return Source()

    async def test_cycle_executes_all_four_models_with_matching_ids(self):
        from edge_agent.control_v1 import run_pipeline
        controller = RuleController(Y8, 30, allowed_modes=FOUR, fps_levels=(30,), stream_key=KEY,
                                    policy='cycle', cooldown_s=.15)
        monitor = TimeWindowMonitor(KEY, window_ms=150)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback())
        windows = []
        summary = await run_pipeline(self.source(60), self.publisher(), self.client,
            {Y8: Detector('yolov8n-artifact'), Y11: Detector('yolo11s-artifact')}, controller, monitor,
            cloud_model_ids={RT: 'rt-artifact', LW: 'lw-artifact'}, priming_frames=0, padding_frames=0,
            blur_width=16, emit=windows.append, resource_interval_s=.05,
            resource_sampler=SimpleNamespace(sample=lambda: {"cpu_max_core_pct": 70., "temp_c": None, "ram_pct": 40.}))
        # v2 state inputs measured live: resources, encoder, pixels (motion needs two samples), boxes.
        for name in ("cpu_max_core_pct", "ram_pct", "encoder_write_ms", "brightness", "motion", "vehicle_count"):
            self.assertTrue(any(w['raw'][name] is not None for w in windows), name)
        self.assertTrue(all(w['raw']['temp_c'] is None for w in windows))  # unreadable stays missing
        self.assertEqual(max(w['raw']['vehicle_count'] or 0 for w in windows), 1)  # one car per fake result
        accepted = [e for w in windows for e in w['frames'] if e['status'] == 'accepted']
        self.assertEqual({e['model_mode'] for e in accepted}, {m.value for m in FOUR})
        ids = {'yolo_edge': 'yolov8n-artifact', 'yolo11s_edge': 'yolo11s-artifact',
               'rtdetr_cloud': 'rt-artifact', 'lwdetr_large_cloud': 'lw-artifact'}
        self.assertTrue(all(e['model_id'] == ids[e['model_mode']] for e in accepted))
        self.assertTrue(all(e['placement'] == ('edge' if e['model_mode'].endswith('_edge') else 'cloud')
                            for e in accepted))
        self.assertEqual(set(summary['frames_by_mode']), set(ids))
        self.assertEqual(summary['stats'].get('processing_errors', 0), 0)
        self.assertGreaterEqual(summary['stats']['action_changes'], 3)
        # Global Cloud state for the centralized critic: one JSON line per closed window.
        lines = [json.loads(line) for line in self.log.getvalue().splitlines()]
        self.assertTrue(lines)
        self.assertEqual([x['sequence'] for x in lines], sorted({x['sequence'] for x in lines}))
        self.assertTrue(any(x['arrival_rate_rps'] > 0 for x in lines))
        self.assertTrue(all('logged_utc_ns' in x for x in lines))

    async def test_jpeg_tickets_carry_frames_and_send_no_media_refs(self):
        from edge_agent.control_v1 import run_pipeline
        class NoRefs:
            async def publish_reference(self, ref):
                raise AssertionError("JPEG mode must not send MediaFrameRefs to the Cloud")
        self.core.frame_taps = NoRefs()
        shapes = []
        class Recording(Detector):
            def infer(self, pixels):
                shapes.append(pixels.shape)
                return super().infer(pixels)
        self.core.cloud_adapters[RT] = Recording('rt-artifact')
        controller = RuleController(RT, 30, allowed_modes=(RT,), fps_levels=(30,), stream_key=KEY, policy='observe')
        monitor = TimeWindowMonitor(KEY, window_ms=150)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback())
        windows = []
        summary = await run_pipeline(self.source(20), self.publisher(), self.client, {}, controller, monitor,
            cloud_model_ids={RT: 'rt-artifact'}, priming_frames=0, padding_frames=0, blur_width=16,
            emit=windows.append, resource_interval_s=.05, cloud_jpeg_quality=85, cloud_jpeg_size=64,
            resource_sampler=SimpleNamespace(sample=lambda: {"cpu_max_core_pct": 70., "temp_c": None, "ram_pct": 40.}))
        accepted = [e for w in windows for e in w['frames'] if e['status'] == 'accepted']
        self.assertGreater(len(accepted), 5)
        self.assertEqual(set(shapes), {(64, 64, 3)})  # the Cloud model got the square JPEG, not an RTSP frame
        self.assertTrue(all(e['jpeg_bytes'] > 0 and e['jpeg_encode_ms'] >= 0 for e in accepted))
        self.assertEqual(summary['stats'].get('processing_errors', 0), 0)
        with self.assertRaisesRegex(ValueError, 'JPEG'):
            await run_pipeline(self.source(1), self.publisher(), self.client, {}, controller, monitor,
                               cloud_model_ids={RT: 'rt-artifact'}, cloud_jpeg_quality=0, emit=lambda v: None)

    async def test_missing_detector_or_cloud_id_fails_before_media(self):
        from edge_agent.control_v1 import run_pipeline
        controller = RuleController(Y8, 30, allowed_modes=(Y8, Y11, RT), fps_levels=(30,), stream_key=KEY)
        publisher = self.publisher()
        with self.assertRaisesRegex(ValueError, 'yolo11s_edge'):
            await run_pipeline(self.source(1), publisher, self.client, {Y8: Detector('y')}, controller,
                               TimeWindowMonitor(KEY), cloud_model_ids={RT: 'rt-artifact'}, emit=lambda v: None)
        with self.assertRaisesRegex(ValueError, 'rtdetr_cloud'):
            await run_pipeline(self.source(1), publisher, self.client,
                               {Y8: Detector('y'), Y11: Detector('y11')}, controller,
                               TimeWindowMonitor(KEY), emit=lambda v: None)
        self.assertEqual(publisher.frames_submitted, 0)


class CliTests(unittest.TestCase):
    def test_parser_accepts_four_modes_policies_and_ids(self):
        from edge_agent.control_v1 import parser
        args = parser().parse_args(['--video', 'v.mp4', '--mode', 'yolo11s_edge', '--policy', 'explore',
            '--seed', '5', '--allowed-modes', 'yolo_edge,yolo11s_edge,rtdetr_cloud,lwdetr_large_cloud',
            '--cloud-model-ids', '{"rtdetr_cloud": "rt"}', '--yolo11s-dlc', 'm.dlc', '--http-port', '8005',
            '--loop-source'])
        self.assertEqual((args.mode, args.policy, args.seed, args.http_port, args.loop_source),
                         ('yolo11s_edge', 'explore', 5, 8005, True))
        self.assertFalse(parser().parse_args(['--video', 'v']).loop_source)
        with self.assertRaises(SystemExit):
            parser().parse_args(['--video', 'v', '--mode', 'rtmdet_edge'])

    def test_run_rejects_drl_without_actor_and_bad_ids(self):
        from edge_agent.control_v1 import parser, run
        base = ['--video', 'v.mp4', '--mode', 'yolo_edge', '--allowed-modes', 'yolo_edge,rtdetr_cloud']
        with self.assertRaisesRegex(ValueError, '--actor'):
            asyncio.run(run(parser().parse_args(base + ['--policy', 'drl'])))
        with self.assertRaisesRegex(ValueError, 'JSON object'):
            asyncio.run(run(parser().parse_args(base + ['--cloud-model-ids', '[1]'])))


def _bash():
    import shutil, subprocess
    for candidate in (r"C:\Program Files\Git\bin\bash.exe", shutil.which("bash")):
        try:
            if candidate and subprocess.run([candidate, "-c", "echo ok"], capture_output=True,
                                            timeout=20).stdout.strip() == b"ok":
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            pass
    return None


class ControlScriptTests(unittest.TestCase):
    def run_script(self, *args):
        import subprocess
        bash = _bash()
        if bash is None:
            self.skipTest("bash unavailable")
        root = Path(__file__).resolve().parents[1]
        return subprocess.run([bash, "scripts/control.sh", *args], cwd=root, capture_output=True,
                              text=True, timeout=60, env={**__import__("os").environ, "EDGE_NODE": "edge-246"})

    def test_usage_and_unknown_policy_fail_before_any_work(self):
        usage = self.run_script()
        self.assertEqual(usage.returncode, 2)
        self.assertIn("Usage", usage.stderr)
        bad = self.run_script("edge", "greedy")
        self.assertEqual(bad.returncode, 1)
        self.assertIn("[C1] FAIL: unknown policy greedy", bad.stderr)


if __name__ == '__main__':
    unittest.main()


class ProbeTests(FourModeExecutorTests):
    """docs/PLAN_PROBE_VI.md P1-P3 over the real gRPC control path (reuses the parent's server fixture only)."""
    test_cycle_executes_all_four_models_with_matching_ids = None
    test_jpeg_tickets_carry_frames_and_send_no_media_refs = None
    test_missing_detector_or_cloud_id_fails_before_media = None

    async def run_probe(self, mode, *, period=.2, frames=40):
        from edge_agent.control_v1 import run_pipeline
        seen = []
        class Reference(Detector):
            def infer(self, pixels):
                seen.append(pixels.shape)
                # one extra vehicle the Edge fake never reports -> probe_missed = 1
                return (Box(.1, .1, .5, .6, .9, 2, 'car'), Box(.6, .6, .9, .9, .8, 2, 'car')), 1.
        self.core.cloud_adapters[ModelMode.DFINEX_CLOUD] = Reference('dx-artifact')
        controller = RuleController(mode, 30, allowed_modes=(mode,), fps_levels=(30,), stream_key=KEY, policy='observe')
        monitor = TimeWindowMonitor(KEY, window_ms=150)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback())
        windows = []
        summary = await run_pipeline(self.source(frames), self.publisher(), self.client,
            {Y8: Detector('yolov8n-artifact'), Y11: Detector('yolo11s-artifact')}, controller, monitor,
            cloud_model_ids={RT: 'rt-artifact', ModelMode.DFINEX_CLOUD: 'dx-artifact'}, priming_frames=0,
            padding_frames=0, blur_width=16, emit=windows.append, resource_interval_s=.05, cloud_jpeg_quality=85,
            cloud_jpeg_size=64, probe_period_s=period, probe_frames=2,
            resource_sampler=SimpleNamespace(sample=lambda: {"cpu_max_core_pct": 70., "temp_c": None, "ram_pct": 40.}))
        return summary, windows, seen

    async def test_edge_stream_probes_cloud_reference_without_touching_results(self):
        summary, windows, seen = await self.run_probe(Y11)
        stats = summary['stats']
        self.assertGreaterEqual(stats.get('probe_ok', 0), 3)
        self.assertEqual(stats.get('processing_errors', 0), 0)
        self.assertEqual(set(summary['frames_by_mode']), {'yolo11s_edge'})  # probes are not frames of the stream
        self.assertEqual(len(seen), stats['probe_ok'])  # every successful probe ran the reference once
        missed = [w['raw']['probe_missed'] for w in windows if w['raw']['probe_missed'] is not None]
        self.assertTrue(missed and all(v == 1 for v in missed))
        self.assertTrue(any(w['raw']['probe_rtt_ms'] for w in windows))
        events = [e for w in windows for e in w['frames']]
        self.assertTrue(all(e['model_mode'] == 'yolo11s_edge' for e in events))
        self.assertFalse(any(e['decision_id'].startswith('probe/') for e in events))

    async def test_probe_period_bounds_the_probe_count(self):
        summary, _w, _s = await self.run_probe(Y11, period=10)
        self.assertEqual(summary['stats'].get('probe_sent', 0), 1)  # 2 per 10 s = one per 5 s; run lasts ~1.5 s
        summary, _w, seen = await self.run_probe(Y11, period=0)
        self.assertEqual(summary['stats'].get('probe_sent', 0), 0)

    async def test_stream_on_reference_model_probes_only_the_edge(self):
        summary, windows, seen = await self.run_probe(ModelMode.DFINEX_CLOUD)
        stats = summary['stats']
        self.assertGreater(stats.get('probe_sent', 0), 0)
        # The Cloud result of the frame is reused: the reference ran only for the stream's own frames.
        self.assertEqual(len(seen), summary['frames_by_mode']['dfine_x_cloud'])
        self.assertEqual(stats.get('probe_sent', 0), stats.get('probe_ok', 0) + stats.get('probe_dropped', 0))

    def test_probe_missed_counts_reference_vehicles_the_edge_misses(self):
        from edge_agent.control_v1 import probe_missed
        car = lambda l, t, c=.9: Box(l, t, l + .1, t + .1, c, 2, 'car')
        self.assertEqual(probe_missed([car(.1, .1)], [car(.1, .1), car(.5, .5)]), 1)
        self.assertEqual(probe_missed([car(.1, .1, .2)], [car(.1, .1)]), 1)       # Edge box below 0.30
        self.assertEqual(probe_missed([], [car(.1, .1, .5)]), 0)                 # reference box below 0.65
        self.assertEqual(probe_missed([car(.1, .1), car(.12, .12)], [car(.1, .1)]), 0)
        person = Box(.1, .1, .2, .2, .9, 0, 'person')
        self.assertEqual(probe_missed([], [person]), 0)


class CloudSchedulerTests(FourModeExecutorTests):
    """docs/PLAN_CONTROL_V3_VI.md C1-C5 over the real gRPC control path: one Cloud scheduler, per-stream commands."""
    test_cycle_executes_all_four_models_with_matching_ids = None
    test_jpeg_tickets_carry_frames_and_send_no_media_refs = None
    test_missing_detector_or_cloud_id_fails_before_media = None

    async def start_scheduler(self, policy, period=.2):
        from cloud_core.scheduler_v3 import Scheduler, StreamRegistry
        self.service.registry = StreamRegistry()
        self.service.scheduler = Scheduler(self.service.registry, self.telemetry, policy, period_s=period, log=io.StringIO())
        task = asyncio.create_task(self.service.scheduler.run())
        self.addAsyncCleanup(lambda: asyncio.gather(task, return_exceptions=True))
        self.addCleanup(task.cancel)
        return self.service.scheduler

    async def run_stream(self, key, frames=60, fps_levels=(30,)):
        from edge_agent.control_v1 import run_pipeline
        controller = RuleController(Y8, 30, allowed_modes=FOUR, fps_levels=fps_levels, stream_key=key, policy='cloud')
        monitor = TimeWindowMonitor(key, window_ms=150)
        await asyncio.sleep(.12)
        monitor.feedback.accept(await self.client.get_feedback(stream_key=list(key)))
        windows = []
        from edge_agent.capture import CapturedFrame
        class Source:
            async def __aiter__(self):
                for i in range(frames):
                    yield CapturedFrame(identity=FrameIdentity(*key[:3], key[3], i, time_ns(), i * 33_333_333),
                                        pixels=np.zeros((16, 16, 3), dtype=np.uint8), capture_mono_ns=monotonic_ns())
                    await asyncio.sleep(1 / 30)
        summary = await run_pipeline(Source(), self.publisher(), self.client,
            {Y8: Detector('yolov8n-artifact'), Y11: Detector('yolo11s-artifact')}, controller, monitor,
            cloud_model_ids={RT: 'rt-artifact', LW: 'lw-artifact'}, priming_frames=0, padding_frames=0,
            blur_width=16, emit=windows.append, resource_interval_s=.05,
            resource_sampler=SimpleNamespace(sample=lambda: {"cpu_max_core_pct": 70., "temp_c": None, "ram_pct": 40.}))
        return summary, windows

    async def test_each_stream_runs_its_own_command(self):
        from cloud_core.policies_v3 import FixedPolicy
        scheduler = await self.start_scheduler(FixedPolicy({"default": ["yolo11s_edge", 30],
                                                            "streams": {"edge-b:camera:control": ["rtdetr_cloud", 30]}}))
        a, b = ('edge-a', 'camera', 'control', 1), ('edge-b', 'camera', 'control', 1)
        (sa, wa), (sb, wb) = await asyncio.gather(self.run_stream(a), self.run_stream(b))
        self.assertGreaterEqual(sa['stats'].get('state_uploaded', 0), 3)      # C1: windows reach the Cloud
        self.assertIn(b, scheduler.commands)                                 # C2: a command per live stream
        # C3/C4: after the first command each stream runs its own model (frames before it ran the initial YOLOv8n).
        self.assertGreater(sa['frames_by_mode'].get('yolo11s_edge', 0), 30)
        self.assertGreater(sb['frames_by_mode'].get('rtdetr_cloud', 0), 30)
        self.assertNotIn('rtdetr_cloud', sa['frames_by_mode'])
        self.assertTrue(any(w['context']['decision_id'].startswith('cloud/') for w in wb))
        self.assertEqual(sb['stats'].get('processing_errors', 0), 0)

    async def test_command_change_and_expiry_fallback(self):
        from cloud_core.policies_v3 import FixedPolicy
        scheduler = await self.start_scheduler(FixedPolicy({"default": ["rtdetr_cloud", 30]}))
        key = ('edge-c', 'camera', 'control', 1)
        async def switch_then_stop():
            await asyncio.sleep(.9)
            scheduler.policy = FixedPolicy({"default": ["lwdetr_large_cloud", 30]})   # new decision mid-run
            await asyncio.sleep(.9)
            scheduler.policy = FixedPolicy({"default": ["bogus_mode", 30]})          # invalid -> rejected
        changer = asyncio.create_task(switch_then_stop())
        summary, windows = await self.run_stream(key, frames=110)
        await changer
        frames = summary['frames_by_mode']
        self.assertGreater(frames.get('rtdetr_cloud', 0), 5)
        self.assertGreater(frames.get('lwdetr_large_cloud', 0), 5)
        modes = [w['context']['model_mode'] for w in windows]
        self.assertEqual(modes[-1], 'yolo_edge')  # invalid commands: a Cloud stream falls back to the Edge model


class SchedulerUnitTests(unittest.TestCase):
    def test_rules_policy_ranks_strong_encoders_and_backs_off_under_pressure(self):
        from cloud_core.policies_v3 import RulesPolicy
        streams = {(f'e{i}', 'c', 's', 0): {"capability": {"encode_fps": 10 * i}} for i in range(1, 8)}
        actions = RulesPolicy(85).decide(streams, {"cloud_arrival_pressure_norm": .2})   # k = floor(.8*85/15) = 4
        cloud = sorted(k[0] for k, (m, _f) in actions.items() if m == "dfine_x_cloud")
        self.assertEqual(cloud, ['e4', 'e5', 'e6', 'e7'])
        actions = RulesPolicy(85).decide(streams, {"cloud_arrival_pressure_norm": .95})
        self.assertEqual(sum(m == "dfine_x_cloud" for m, _f in actions.values()), 3)

    def test_registry_expires_silent_streams_and_ignores_older_windows(self):
        from cloud_core.scheduler_v3 import StreamRegistry
        now = [0]
        reg = StreamRegistry(expire_s=1, clock=lambda: now[0])
        state = lambda w: {"control_v1": 1, "stream_key": ["n", "c", "s", 0], "window_id": w,
                           "context": {"model_mode": "yolo_edge"}, "raw": {"timely_fps": 10.0}}
        reg.update(state(5)); reg.update(state(3))
        self.assertEqual(reg.live()[("n", "c", "s", 0)]["window_id"], 5)
        now[0] = 2_000_000_000
        self.assertEqual(reg.live(), {})
        with self.assertRaises(ValueError):
            reg.update({**state(1), "raw": {"timely_fps": float("nan")}})


class StateV3Tests(unittest.TestCase):
    def test_same_raw_same_vector_in_sim_and_live(self):
        from drl import state_v3
        raw = {"cpu_max_core_pct": 50, "timely_fps": 15, "probe_missed_last": 4, "probe_missed": 2,
               "probe_age_s": 3, "brightness": 128, "laplacian_small": 2000, "conf_mean": .7}
        for variant in state_v3.VARIANTS:
            v = state_v3.vector(raw, "dfine_x_cloud", 10, variant)
            self.assertEqual(len(v), len(state_v3.names(variant)))
            self.assertTrue(all(0 <= x <= 1 for x in v))
        probe = dict(zip(state_v3.names("probe"), state_v3.vector(raw, "yolo_edge", 15, "probe")))
        self.assertEqual((probe["probe_missed_last"], probe["probe_available"], probe["mode_yolo_edge"]), (.4, 1.0, 1.0))
        never = dict(zip(state_v3.names("probe"), state_v3.vector({}, "yolo_edge", 15, "probe")))
        self.assertEqual((never["probe_age_s"], never["probe_available"]), (1.0, 0.0))
