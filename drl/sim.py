"""Multi-Edge simulator for MAPPO training (docs/DRL_DESIGN_VI.md).

One tick = one 1 s control window per Edge; one decision every HISTORY ticks (5 s), exactly as live.
Each tick produces a TimeWindowMonitor-shaped snapshot, and observations come from the live
edge_agent.drl_policy.observation_v2, so simulator and Edge share one state definition.

Sources (see SimConfig):
  * quality: per-second F1 + model-observed features from `detrac.py table` (Cloud modes: live-transcoded
    video; Edge modes: FP32 dump plus a measured INT8 delta on NPU Edges); pixel features from `detrac.py scene`.
  * capacity: benchmark capabilities (scripts/hosts EDGE_CAPABILITY) and Cloud model capacities (control.sh).
  * Cloud sharing: max-min water-filling of GPU time (contention benchmark, 2/3 Edges).
  * v1 calibration (2026-10-13, control-v1 explore on v45, results/tableB_v45b, 3 Edges x ~19 min): Edge-mode
    rate limited by x264 + YOLO sharing the CPU, Cloud modes by one request in flight per Edge (rate =
    1000 / round trip), round trip interpolated over 1000/encode_fps, encoder load, CPU and latency bases.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random

import numpy as np

from edge_agent.drl_policy import ACTION_MODES, HISTORY, observation_v2

TABLE_MODEL = {"yolo_edge": "yolov8n", "yolo11s_edge": "yolo11s",
               "rtdetr_cloud": "rtdetrv2", "lwdetr_large_cloud": "lwdetr"}
CLOUD_MODES = frozenset({"rtdetr_cloud", "lwdetr_large_cloud"})
EDGE_CAP_KEY = {"yolo_edge": "yolov8n_fps", "yolo11s_edge": "yolo11s_fps"}


# Image / YOLO-confidence features of state variant B (scripts/hardness_signal.py names).
# shortcut: conf features are the YOLO11s table's whatever model runs (live: the running model); per-mode tables later.
SIGNAL_FEATURES = ("contrast", "laplacian_small", "conf_mean", "uncertain_ratio")


@dataclass(frozen=True)
class EdgeProfile:
    name: str
    yolov8n_fps: float
    yolo11s_fps: float
    encode_fps: float
    npu: bool                # INT8 NPU: apply the measured INT8-vs-FP32 F1 delta to Edge modes
    temp_idle_c: float | None
    temp_full_c: float | None  # steady temperature at 100% hottest-core load (None: no sensor)

    @property
    def capability(self):
        return {"yolov8n_fps": self.yolov8n_fps, "yolo11s_fps": self.yolo11s_fps, "encode_fps": self.encode_fps}


# Benchmarks 2026-10-11 (pipeline, inflight 2); thermal ends fitted to the explore runs (246 ~48-52 C at
# ~30% hottest core, 252 ~70-76 C at ~56%).
REAL_EDGES = (
    EdgeProfile("edge-246", 101.6, 86.3, 66.6, True, 40.0, 70.0),
    EdgeProfile("edge-247", 201.4, 165.2, 268.8, False, None, None),
    EdgeProfile("edge-252", 45.6, 39.0, 20.9, True, 50.0, 87.0),
)

# ---- v1 calibration, measured per Edge (results/tableB_v45b, TABLE_B.md) ----
# Edge-mode timely ratio: 246/247 1.00, 252 .82 -> min(1, encode_fps / 25.5): x264 and YOLO pre/post share the CPU.
EDGE_FACTOR_ENC_FPS = 25.5
# Cloud round trip p50 minus mean GPU time (ms) at x = 1000/encode_fps, per tickets in flight. Weak encoders
# publish late, so the Cloud waits longer for the frame. Interpolated, last slope extrapolated.
#   1 in flight (v45, tableB_v45b): 247 67, 246 83, 252 177 ms.
#   2 in flight (v47/v48, tableB_v47_clean): 247 88, 246 101, 252 129 ms (two tickets queue for the GPU and
#   the per-stream order gate; 252 was much less loaded that evening).
JOIN_POINTS = {1: ((3.72, 58.0), (15.0, 75.0), (47.8, 168.0)),
               2: ((3.72, 79.0), (15.0, 92.0), (47.8, 120.0))}
# Cloud modes also lose frames the Edge never published: timely x edge_factor ** .5 (252: .31 measured).
CLOUD_MEDIA_EXP = .5
# Encoder load (x264 ms per frame x 15 FPS / 1000): 247 .04, 246 .14, 252 ~1.1 measured.
ENC_LOAD = (9.3, 4.5)  # load = 9.3 / encode_fps * max(1, 25.5 / encode_fps) ** 4.5
# Timely accuracy needs capture-to-result latency: path latency + the wait before a worker takes the frame
# (measured p50 +10..+38 ms, larger on weak encoders) and its spread (p95/p50 ~1.1-1.4 -> lognormal sigma .15).
PRE_WORKER_MS = (8.0, .6)  # 8 + .6 x 1000 / encode_fps
LATENCY_SIGMA = .15
TA_DEADLINES_MS = (100, 200, 300, 500)
SWITCH_IN_CLOUD_DIP = .9  # first second after switching to a Cloud model (measured .79-.85 vs .85-.91 later)


def extra_feature_names(cfg):
    """Simulator-only observation columns appended after OBS_V2 (see SimConfig)."""
    return ((("priority",) if cfg.priority_feature else ()) + (("fps",) if len(cfg.fps_levels) > 1 else ())
            + (("cloud_share", "gpu_busy") if cfg.coord_features else ()))


def interp_join_ms(x, inflight=2):
    pts = JOIN_POINTS[min(inflight, 2)]
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return y1 + (y1 - y0) * (x - x1) / (x1 - x0)


@dataclass
class SimConfig:
    target_fps: float = 15.0
    # Analytics FPS the policy may pick with each model (simulator experiment; live control v1 keeps 15). With
    # several levels the action is (model, FPS) and F1 comes from the per-FPS table (scripts/fps_table.py).
    fps_levels: tuple = (15,)
    deadline_ms: float = 500.0
    queue_frames: float = 2.0             # live analytics queue keeps only the latest frames (put_latest)
    # GPU frames/s with 2 slots, fitted 2026-10-14 on 20 live experiments (4-12 streams, all-LW / all-RT / mixed /
    # explore at 5-15 FPS; scripts/sim_replay_ta.py): mean |sim - live| TA@200 0.008, worst 0.033.
    cloud_capacity: dict = field(default_factory=lambda: {"rtdetr_cloud": 172.5, "lwdetr_large_cloud": 95.0})
    a_min: float = .7                     # accuracy floor; chosen from the quality table
    lagrange: float = 1.0                 # lambda; the trainer adapts it
    w_latency: float = .2
    w_loss: float = 1.0
    w_switch: float = .05
    w_fair: float = .1
    episode_decisions: int = 60           # 5 min
    edges: tuple = (1, 4)                 # Edges per episode (inclusive range)
    jitter: float = .2                    # +-20% capability randomization
    encode_jitter: float = .5             # encoder share varies more (252: load 1.1 vs 0.29 on the same day)
    random_profile_p: float = .25         # chance an Edge is a synthetic device, not one of the real three
    cloud_background: tuple = (.6, 1.)    # share of Cloud GPU left for us (other tenants)
    network_ms: tuple = (3., 10.)         # LAN (testbed measured ~3 ms) ...
    wan_ms: tuple = (10., 60.)            # ... or, with probability wan_p, a WAN-like path
    wan_p: float = .3
    outage_p: float = .002                # per second; Cloud feedback stale -> Cloud modes masked
    outage_s: tuple = (3, 10)
    real_edges_only: bool = False         # evaluation scenario: exactly the three real Edges, no jitter
    cloud_inflight: int = 2               # Cloud tickets in flight per Edge (control.sh CONTROL_CLOUD_INFLIGHT)
    # reward "ta" (chosen 2026-10-13): per Edge TA@ta_deadline_ms - w_switch x switch, i.e. the evaluation metric
    # itself; "constrained": timely - lambda x F1 shortfall - latency/loss/switch penalties (earlier design).
    reward: str = "constrained"
    # Within-episode dynamics (per 1 s tick), ranges from measurements: other tenants take a share of the GPU
    # (249 is shared), an Edge's encoder share swings (252: load 1.1 vs .29 on one day), WAN-like spells.
    dyn_bg_p: float = 0.0                 # chance per tick to redraw the GPU share left to us from cloud_background
    dyn_enc_p: float = 0.0                # chance per tick and Edge to redraw its encoder multiplier from enc_swing
    enc_swing: tuple = (.5, 1.5)
    dyn_net_p: float = 0.0                # chance per tick to redraw the network delay (WAN with wan_p)
    # Other work on the Edge CPU (other tenants / processes, e.g. 247 at 90% RAM): local model speed multiplier.
    dyn_edge_p: float = 0.0               # chance per tick and Edge to redraw its local-model multiplier
    edge_swing: tuple = (.5, 1.)
    # Scene regimes (realistic cameras): each camera keeps one weather/lighting regime (UA-DETRAC weather tag:
    # sunny/cloudy/rainy/night) and plays sequences of that regime; the regime changes rarely (per tick chance).
    # Off: every camera jumps to a random sequence of any weather (the playlist-like testbed).
    scene_regime: bool = False
    regime_switch_p: float = 1 / 1800
    # Per-Edge sites: with this chance an Edge sits behind a WAN/4G link (its own delay from wan_ms, jitter
    # redrawn with dyn_net_p) instead of the shared LAN.
    edge_wan_p: float = 0.0
    # Simulator-only experiment: a fixed per-Edge random priority in [0, 1] appended to the observation, so
    # identical Edges sharing one actor can split roles (who goes to the Cloud). Not in the Edge runtime.
    priority_feature: bool = False
    # Simulator-only coordination experiment ("Edges see each other" through the Cloud's feedback): the share of
    # Edges currently on the Cloud and the true GPU busy share including other tenants (nvidia-smi-like).
    coord_features: bool = False
    # Asynchronous decisions: per decision only this share of Edges may change model (their 5 s timers are not
    # aligned in reality); the others keep their model, so later movers see what earlier movers did.
    async_share: float = 1.0
    ta_deadline_ms: float = 200.0
    # Mean paired per-second F1, NPU INT8 (Edge246) minus FP32 (Edge247), 10 test videos / 578 s (2026-10-13).
    int8_delta: dict = field(default_factory=lambda: {"yolo_edge": -.0313, "yolo11s_edge": -.0287})
    # Probe (docs/PLAN_PROBE_VI.md): probe_frames frames per probe_period_s (0 = off); a probe is dropped when the
    # Cloud is unreachable, with probability probe_busy_drop x GPU overload above 80%, and (stream on the Cloud)
    # when the Edge still has queued frames (the Edge-side YOLO run waits for a free accelerator).
    probe_period_s: float = 0.0
    probe_frames: int = 2
    probe_busy_drop: float = 1.0
    # v0 assumptions (fit from control-v1 explore state.jsonl): thermal time constant, throttling.
    temp_tau_s: float = 60.
    throttle_c: float = 90.
    throttle_factor: float = .6


class QualityTable:
    """(sequence, second) -> per-mode F1 and model-observed features, plus pixel features."""

    def __init__(self, rows, scene_rows=(), split="train", probe_rows=(), signal_rows=()):
        per, self.f1_fps, self.weather = {}, {}, {}
        # docs/PLAN_PROBE_VI.md P5: per (sequence, second) the probe values (YOLO11s INT8 misses vs D-FINE-X on two
        # frames) and the image / YOLO features of state variant B (scripts/hardness_signal.py).
        self.extras = {}
        for r in probe_rows:
            self.extras.setdefault((r["sequence"], r["second"]), {})["probe"] = r["probe_missed"]
        for r in signal_rows:  # only the YOLO confidence features; pixel ones come from scene_rows (live scale)
            self.extras.setdefault((r["sequence"], r["second"]), {}).update(
                {k: r[k] for k in ("conf_mean", "uncertain_ratio") if r.get(k) is not None})
        for row in rows:
            if split and row.get("split") != split:
                continue
            mode = {v: k for k, v in TABLE_MODEL.items()}.get(row["model"])
            if not mode:
                continue
            key = (row["sequence"], row["window"])
            if row.get("weather"):
                self.weather[row["sequence"]] = row["weather"]
            if "fps" in row:  # analytics-FPS table: F1 with boxes held between analyzed frames
                self.f1_fps.setdefault(key, {})[(mode, row["fps"])] = row["f1"]
            if "fps" not in row or (row["fps"] == 15 and mode not in per.get(key, {})):
                per.setdefault(key, {})[mode] = (
                    row["f1"], row.get("vehicle_count"), row.get("box_change"), row.get("small_ratio"))
        scene = {(r["sequence"], r["window"]): (r.get("brightness"), r.get("motion")) for r in scene_rows}
        # Same edge_agent.resources_v2.scene_features as live: contrast / laplacian for state variant B.
        for r in scene_rows:
            self.extras.setdefault((r["sequence"], r["window"]), {}).update(
                {k: r[k] for k in ("contrast", "laplacian_small") if r.get(k) is not None})
        self.cells = {k: (v, scene.get(k, (None, None))) for k, v in per.items() if len(v) == len(ACTION_MODES)}
        self.sequences = {}
        for sequence, second in sorted(self.cells):
            self.sequences.setdefault(sequence, []).append(second)
        if not self.sequences:
            raise ValueError("quality table has no second scored by all four models")

    @classmethod
    def load(cls, table_paths, scene_path=None, split="train", probe_path=None, signal_path=None):
        read = lambda path: [] if path is None else [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]  # noqa: E731
        rows = [json.loads(line) for p in table_paths for line in Path(p).read_text(encoding="utf-8").splitlines()]
        return cls(rows, read(scene_path), split, read(probe_path), read(signal_path))

    def subset(self, sequences):
        """Same table restricted to `sequences` (e.g. a validation hold-out of the train split)."""
        part = object.__new__(QualityTable)
        part.cells = {k: v for k, v in self.cells.items() if k[0] in sequences}
        part.sequences = {s: self.sequences[s] for s in sequences}
        part.f1_fps = {k: v for k, v in self.f1_fps.items() if k[0] in sequences}
        part.weather = {k: v for k, v in self.weather.items() if k in sequences}
        part.extras = {k: v for k, v in self.extras.items() if k[0] in sequences}
        return part

    def regimes(self):
        """weather -> its sequences (sequences without a tag form the "any" regime)."""
        out = {}
        for seq in sorted(self.sequences):
            out.setdefault(self.weather.get(seq, "any"), []).append(seq)
        return out

    def f1(self, key, mode, fps):
        """F1 of `mode` analyzing `fps` frames/s at (sequence, second); 15 falls back to the base table."""
        value = self.f1_fps.get(key, {}).get((mode, fps))
        if value is None:
            if fps != 15:
                raise KeyError(f"no F1 for {mode} at {fps} FPS: load the per-FPS table (scripts/fps_table.py)")
            value = self.cells[key][0][mode][0]
        return value

    def mean_f1(self):
        return {m: float(np.mean([v[0][m][0] for v in self.cells.values()])) for m in ACTION_MODES}


def water_fill(needs, capacity=1.0):
    """Max-min fair split of `capacity` among `needs` -> allocations (each <= its need)."""
    alloc = [0.0] * len(needs)
    left, active = capacity, sorted(range(len(needs)), key=lambda i: needs[i])
    while active:
        share = left / len(active)
        i = active[0]
        if needs[i] <= share:
            alloc[i] = needs[i]
            left -= needs[i]
            active.pop(0)
        else:
            for j in active:
                alloc[j] = share
            break
    return alloc


MAX_LATENCY_MS = 10_000.0  # a frame later than this is lost to the viewer anyway


def _md1_ms(arrival, rate):
    """Mean M/D/1 queueing wait; utilization capped at 0.95 (beyond that the backlog term takes over)."""
    rho = min(arrival / rate, .95) if rate > 0 else .95
    return 1000 * rho / (2 * max(rate, 1e-6) * (1 - rho))


def on_time_share(latency_ms, deadline_ms, sigma=LATENCY_SIGMA):
    """Share of frames whose lognormal capture-to-result latency (median latency_ms) is <= deadline_ms."""
    if latency_ms <= 0:
        return 1.0
    return .5 * (1 + math.erf(math.log(deadline_ms / latency_ms) / (sigma * math.sqrt(2))))


def jain(values):
    values = [v for v in values]
    if not values or not any(values):
        return 1.0
    return sum(values) ** 2 / (len(values) * sum(v * v for v in values))


class _Edge:
    def __init__(self, profile, rng, table, cfg):
        self.p, self.rng, self.table, self.cfg = profile, rng, table, cfg
        self.mode, self.backlog, self.history = "yolo_edge", 0.0, deque(maxlen=HISTORY)
        self.since_switch = 99
        self.enc_mult = 1.0
        self.cap_mult = 1.0
        self.net_ms = None  # None: the shared network delay; else this Edge's own (WAN/4G site)
        self.regime = rng.choice(sorted(table.regimes())) if cfg.scene_regime else None
        self.fps = cfg.target_fps
        self.priority = rng.random()
        self.temp = profile.temp_idle_c
        self.probe_clock, self.probe_last, self.probe_age = 0.0, None, None
        self._jump()

    def _jump(self):
        pool = self.table.regimes()[self.regime] if self.regime else sorted(self.table.sequences)
        self.sequence = self.rng.choice(pool)
        seconds = self.table.sequences[self.sequence]
        self.cursor = self.rng.randrange(len(seconds))

    def cell(self):
        seconds = self.table.sequences[self.sequence]
        if self.cursor >= len(seconds):
            self._jump()
            seconds = self.table.sequences[self.sequence]
        return self.table.cells[(self.sequence, seconds[self.cursor])]

    def throttled(self):
        return self.temp is not None and self.temp >= self.cfg.throttle_c

    def slow(self):
        return self.cfg.throttle_factor if self.throttled() else 1.0

    def enc_fps(self):
        """Encoder capacity now: nominal profile x the current swing (dynamics) x thermal throttling."""
        return self.p.encode_fps * self.enc_mult * self.slow()

    def edge_factor(self):
        """Share of the 15 FPS the Edge publishes and analyzes while x264 and YOLO share its CPU."""
        return min(1.0, self.enc_fps() / EDGE_FACTOR_ENC_FPS)

    def edge_rate(self):
        return min(getattr(self.p, EDGE_CAP_KEY[self.mode]) * self.slow() * self.cap_mult, self.fps * self.edge_factor())

    def switch_regime(self):
        self.regime = self.rng.choice(sorted(self.table.regimes()))
        self._jump()

    def join_ms(self):
        """Cloud round trip without GPU time and queueing: waiting for this Edge's published frame."""
        return interp_join_ms(1000 / self.enc_fps(), self.cfg.cloud_inflight)

    def encoder_load(self):
        enc = self.enc_fps()
        return ENC_LOAD[0] / enc * max(1.0, EDGE_FACTOR_ENC_FPS / enc) ** ENC_LOAD[1]


class MultiEdgeSim:
    def __init__(self, table, cfg=None, seed=0):
        self.table, self.cfg, self.rng = table, cfg or SimConfig(), random.Random(seed)

    # ---- episode setup (domain randomization) ----
    def _profile(self, k):
        cfg, rng = self.cfg, self.rng
        if rng.random() < cfg.random_profile_p:
            v8 = rng.uniform(20, 250)
            enc = rng.uniform(15, 280)
            npu = rng.random() < .5
            idle = rng.uniform(40, 60) if npu else None
            return EdgeProfile(f"synthetic-{k}", v8, v8 * rng.uniform(.75, .9), enc, npu,
                               idle, None if idle is None else idle + rng.uniform(20, 50))
        base = rng.choice(REAL_EDGES)
        j = lambda x: x * rng.uniform(1 - cfg.jitter, 1 + cfg.jitter)  # noqa: E731
        return EdgeProfile(base.name, j(base.yolov8n_fps), j(base.yolo11s_fps),
                           base.encode_fps * rng.uniform(1 - cfg.jitter, 1 + cfg.encode_jitter),
                           base.npu, base.temp_idle_c, base.temp_full_c)

    def reset(self):
        cfg, rng = self.cfg, self.rng
        profiles = list(REAL_EDGES) if cfg.real_edges_only else [self._profile(k) for k in range(rng.randint(*cfg.edges))]
        self.edges = [_Edge(p, random.Random(rng.random()), self.table, cfg) for p in profiles]
        self.background = rng.uniform(*cfg.cloud_background)
        self.network_ms = rng.uniform(*(cfg.wan_ms if rng.random() < cfg.wan_p else cfg.network_ms))
        for e in self.edges:
            if cfg.edge_wan_p and rng.random() < cfg.edge_wan_p:
                e.net_ms = rng.uniform(*cfg.wan_ms)
        self.outage_left = 0
        self.decision = 0
        self._draw_turn()
        for _ in range(HISTORY):  # warm-up windows so the first observation has history
            self._tick()
        return self.observe()

    # ---- observation ----
    def stale(self):
        return self.outage_left > 0

    @property
    def actions(self):
        """(mode, analytics FPS) per action index; one FPS level keeps the ACTION_MODES order."""
        return [(m, f) for m in ACTION_MODES for f in self.cfg.fps_levels]

    def action_index(self, mode, fps=None):
        return self.actions.index((mode, self.cfg.target_fps if fps is None else fps))

    def masks(self):
        out = []
        for i, e in enumerate(self.edges):
            if i in self.turn:
                out.append([not (m in CLOUD_MODES and self.stale()) for m, _f in self.actions])
            else:  # not this Edge's turn: it keeps its model and FPS
                out.append([(m, f) == (e.mode, e.fps) for m, f in self.actions])
        return out

    def _draw_turn(self):
        n = len(self.edges)
        k = max(1, round(self.cfg.async_share * n))
        self.turn = set(range(n)) if k >= n else set(self.rng.sample(range(n), k))

    def coordination(self):
        """(share of Edges on the Cloud, GPU busy share incl. other tenants) as the Cloud would broadcast them."""
        on_cloud = sum(e.mode in CLOUD_MODES for e in self.edges) / max(1, len(self.edges))
        busy = min(1.0, (1 - self.background) + self.background * min(1.0, self.cloud_util))
        return on_cloud, busy

    def observe(self):
        obs = np.stack([observation_v2(e.history, e.mode) for e in self.edges])
        extra = []
        if self.cfg.priority_feature:
            extra.append([e.priority for e in self.edges])
        if len(self.cfg.fps_levels) > 1:
            extra.append([e.fps / self.cfg.target_fps for e in self.edges])
        if self.cfg.coord_features:
            share, busy = self.coordination()
            extra += [[share] * len(self.edges), [busy] * len(self.edges)]
        return np.concatenate([obs, np.array(extra).T], axis=1) if extra else obs

    def global_state(self):
        """Critic-only extras: Edge count, Cloud headroom, Cloud load, outage."""
        return np.array([len(self.edges) / 6, self.background, min(1.0, self.cloud_util), float(self.stale())])

    # ---- dynamics ----
    def _tick(self):
        cfg, lam = self.cfg, self.cfg.target_fps
        rng = self.rng
        if cfg.dyn_bg_p and rng.random() < cfg.dyn_bg_p:
            self.background = rng.uniform(*cfg.cloud_background)
        if cfg.dyn_net_p and rng.random() < cfg.dyn_net_p:
            self.network_ms = rng.uniform(*(cfg.wan_ms if rng.random() < cfg.wan_p else cfg.network_ms))
        if cfg.dyn_enc_p:
            for e in self.edges:
                if rng.random() < cfg.dyn_enc_p:
                    e.enc_mult = rng.uniform(*cfg.enc_swing)
        if cfg.dyn_edge_p:
            for e in self.edges:
                if rng.random() < cfg.dyn_edge_p:
                    e.cap_mult = rng.uniform(*cfg.edge_swing)
        for e in self.edges:
            if e.net_ms is not None and cfg.dyn_net_p and rng.random() < cfg.dyn_net_p:
                e.net_ms = rng.uniform(*cfg.wan_ms)
            if e.regime and rng.random() < cfg.regime_switch_p:
                e.switch_regime()
        if self.outage_left > 0:
            self.outage_left -= 1
        elif self.rng.random() < cfg.outage_p:
            self.outage_left = self.rng.randint(*cfg.outage_s)
        cloud = [i for i, e in enumerate(self.edges) if e.mode in CLOUD_MODES and not self.stale()]
        caps = {m: c * self.background for m, c in cfg.cloud_capacity.items()}
        # LAN transport (~3 ms) is already inside the join term; WAN sites add their own delay
        extra = {i: max(0.0, (self.network_ms if e.net_ms is None else e.net_ms) - 3.0) for i, e in enumerate(self.edges)}
        # One request in flight per Edge: rate = 1000 / round trip, and the round trip grows with the GPU queue.
        wait, rtt, demand, need, util = {m: 0.0 for m in caps}, {}, {}, [], 0.0
        for _ in range(3):  # fixed point: queue wait -> request rate -> GPU utilization -> queue wait
            rtt = {i: self.edges[i].join_ms() + 1000 / caps[self.edges[i].mode] + wait[self.edges[i].mode] + extra[i]
                   for i in cloud}
            # Closed loop when the GPU saturates: every Edge keeps cloud_inflight tickets queued, so one ticket waits
            # for the GPU to serve all of them (measured 2026-10-14: 8/10/12 LW-DETR streams, p50 172/219/266 ms).
            cycle = 1000 * cfg.cloud_inflight * sum(1 / caps[self.edges[i].mode] for i in cloud)
            rtt = {i: max(v, cycle) for i, v in rtt.items()}
            demand = {i: min(self.edges[i].fps, cfg.cloud_inflight * 1000 / rtt[i]) * self.edges[i].edge_factor() ** CLOUD_MEDIA_EXP
                      for i in cloud}
            need = [demand[i] / caps[self.edges[i].mode] for i in cloud]
            util = sum(need)
            wait = {m: _md1_ms(util * c, c) for m, c in caps.items()}
        share = dict(zip(cloud, water_fill(need)))
        self.cloud_util = util
        # What the Cloud itself reports: arrivals x nominal GPU seconds (it cannot see other tenants).
        self.cloud_pressure = sum(demand[i] / cfg.cloud_capacity[self.edges[i].mode] for i in cloud)
        for i, e in enumerate(self.edges):
            if i in share:
                stats = self._edge_second(e, True, share[i] * caps[e.mode], rtt[i], e.fps)
            else:
                stats = self._edge_second(e, e.mode in CLOUD_MODES, 0.0, None, e.fps)
            e.history.append(self._snapshot(e, stats, e.cell()))
            e.last = stats
            e.cursor += 1
            e.since_switch += 1

    def _edge_second(self, e, on_cloud, cloud_rate, rtt, lam):
        cfg = self.cfg
        if on_cloud:
            rate = 0.0 if rtt is None else cloud_rate  # rtt None: Cloud unreachable, nothing comes back
            if e.since_switch == 0:
                rate *= SWITCH_IN_CLOUD_DIP
        else:
            rate = e.edge_rate()
        available = e.backlog + lam
        done = min(rate, available)
        backlog = available - done
        dropped = max(0.0, backlog - cfg.queue_frames)
        e.backlog = backlog - dropped
        if on_cloud:
            latency = MAX_LATENCY_MS if rtt is None else rtt
        else:
            # Model time + ~6 ms pipeline, stretched when x264 and YOLO share a weak CPU (252: 36 ms measured).
            latency = (1000 / (getattr(e.p, EDGE_CAP_KEY[e.mode]) * e.slow() * e.cap_mult) + 6) / e.edge_factor()
        latency = min(MAX_LATENCY_MS, latency)  # latest-only queue: an analyzed frame is never a stale one
        on_time = 1.0 if latency <= cfg.deadline_ms else max(0.0, 1 - (latency - cfg.deadline_ms) / cfg.deadline_ms)
        timely = done * on_time
        completion = latency + PRE_WORKER_MS[0] + PRE_WORKER_MS[1] * 1000 / e.enc_fps()
        # Hottest core: measured ~15% (247), 30% (246), 56% (252), mode-independent -> 10 + 1000 / encode_fps.
        cpu = min(100.0, 10 + 1000 / e.enc_fps() + self.rng.gauss(0, 3))
        if e.temp is not None:
            steady = e.p.temp_idle_c + (e.p.temp_full_c - e.p.temp_idle_c) * cpu / 100
            e.temp += (steady - e.temp) * (1 - math.exp(-1 / cfg.temp_tau_s))
        return {"timely": timely, "done": done, "dropped": dropped, "latency": latency, "cpu": cpu,
                "on_cloud": on_cloud, "completion": completion, "lam": lam}

    def _snapshot(self, e, s, cell):
        per_mode, (brightness, motion) = cell
        _f1, vehicles, box_change, small = per_mode[e.mode]
        seconds = self.table.sequences[e.sequence]
        f1 = self.table.f1((e.sequence, seconds[min(e.cursor, len(seconds) - 1)]), e.mode, e.fps)
        if e.p.npu and e.mode not in CLOUD_MODES:
            f1 = min(1.0, max(0.0, f1 + self.cfg.int8_delta.get(e.mode, 0.0)))
        s["f1"] = f1
        raw = {"cpu_max_core_pct": s["cpu"], "temp_c": e.temp,
               "edge_backlog_norm": min(1.0, e.backlog / self.cfg.queue_frames),
               "encoder_write_ms": e.encoder_load() * 1000 / self.cfg.target_fps,  # media stays 15 FPS
               "transport_overhead_ms": (self.network_ms if e.net_ms is None else e.net_ms)
                                        if s["on_cloud"] and not self.stale() else None,
               "cloud_arrival_pressure_norm": min(1.0, self.cloud_pressure),
               "timely_fps": s["timely"], "active_path_latency_ms": s["latency"] if s["done"] > 0 else None,
               "brightness": brightness, "motion": motion, "vehicle_count": vehicles,
               "box_change_mean": box_change, "small_ratio": small,
               **self._hardness(e, self.table.extras.get((e.sequence, seconds[min(e.cursor, len(seconds) - 1)]), {}))}
        return {"raw": raw, "capability": e.p.capability, "media_fps": self.cfg.target_fps,
                "context": {"target_fps": e.fps, "model_mode": e.mode}}

    def _hardness(self, e, extras):
        """State variant B features (when loaded) and the probe fields of variant A for this second."""
        out = {name: extras.get(name) for name in SIGNAL_FEATURES}
        cfg = self.cfg
        missed = []
        if cfg.probe_period_s and extras.get("probe"):
            e.probe_clock += 1.0
            interval = cfg.probe_period_s / cfg.probe_frames
            while e.probe_clock >= interval:
                e.probe_clock -= interval
                overload = max(0.0, min(1.0, (self.cloud_util - .8) / .2))
                if self.stale() or self.rng.random() < cfg.probe_busy_drop * overload:
                    continue  # Cloud unreachable or no free GPU slot
                if e.mode in CLOUD_MODES and e.backlog > 0:
                    continue  # Edge accelerator busy: the Edge-side YOLO run is skipped
                missed.append(extras["probe"][len(missed) % len(extras["probe"])])
        if missed:
            e.probe_last, e.probe_age = missed[-1], 0.0
        elif e.probe_age is not None:
            e.probe_age += 1.0
        out.update(probe_missed=float(np.mean(missed)) if missed else None, probe_missed_last=e.probe_last,
                   probe_age_s=e.probe_age)
        return out

    # ---- oracle baseline (upper bound: knows the next 5 s of ground-truth F1) ----
    def oracle_actions(self, shortfall_weight=5.0):
        """Per Edge, the mode with the best predicted timely ratio minus a heavy F1-shortfall penalty."""
        cfg, lam = self.cfg, self.cfg.target_fps
        caps = {m: c * self.background for m, c in cfg.cloud_capacity.items()}
        wait = {m: _md1_ms(min(self.cloud_util, .95) * c, c) for m, c in caps.items()}
        actions = []
        for e in self.edges:
            seconds = self.table.sequences[e.sequence][e.cursor:e.cursor + HISTORY] or [None]
            best = None
            for a, mode in enumerate(ACTION_MODES):
                if mode in CLOUD_MODES:
                    if self.stale():
                        continue
                    rtt = e.join_ms() + 1000 / caps[mode] + wait[mode] + max(0.0, self.network_ms - 3.0)
                    timely = min(lam, cfg.cloud_inflight * 1000 / rtt) * e.edge_factor() ** CLOUD_MEDIA_EXP / lam
                else:
                    saved = e.mode
                    e.mode = mode
                    timely = e.edge_rate() / lam
                    e.mode = saved
                f1s = [self.table.cells[(e.sequence, s)][0][mode][0] for s in seconds if s is not None]
                f1 = float(np.mean(f1s)) if f1s else 0.0
                if e.p.npu and mode not in CLOUD_MODES:
                    f1 += cfg.int8_delta.get(mode, 0.0)
                score = timely - shortfall_weight * max(0.0, cfg.a_min - f1) + .01 * f1
                if best is None or score > best[0]:
                    best = (score, a)
            actions.append(self.action_index(ACTION_MODES[best[1]]))
        return actions

    # ---- decision step ----
    def step(self, actions):
        """actions: one ACTION_MODES index per Edge -> (obs, team_reward, done, info)."""
        cfg = self.cfg
        switched = []
        for i, (e, a) in enumerate(zip(self.edges, actions)):
            mode, fps = self.actions[a] if i in self.turn else (e.mode, e.fps)
            if mode in CLOUD_MODES and self.stale():
                mode = "yolo_edge"  # the live controller's stale fallback
            switched.append((mode, fps) != (e.mode, e.fps))
            if mode != e.mode:
                e.since_switch = 0
            e.mode, e.fps = mode, fps
        windows = [[] for _ in self.edges]
        for _ in range(HISTORY):
            self._tick()
            for k, e in enumerate(self.edges):
                windows[k].append(e.last)
        rewards, parts = [], []
        for k, e in enumerate(self.edges):
            w = windows[k]
            timely = float(np.mean([x["timely"] / x["lam"] for x in w]))
            f1 = float(np.mean([x["f1"] for x in w]))
            latency = float(np.mean([x["latency"] for x in w]))
            loss = float(np.mean([x["dropped"] / x["lam"] for x in w]))
            # Timely accuracy TA@D: mean over frames to analyze of F1 x 1[result within D ms of capture].
            ta = {d: float(np.mean([x["f1"] * x["done"] / x["lam"] * on_time_share(x["completion"], d)
                                    for x in w])) for d in sorted({*TA_DEADLINES_MS, cfg.ta_deadline_ms})}
            if cfg.reward == "ta":
                r = ta[cfg.ta_deadline_ms] - cfg.w_switch * switched[k]
            else:
                r = (timely - cfg.lagrange * max(0.0, cfg.a_min - f1)
                     - cfg.w_latency * min(1.0, max(0.0, latency - cfg.deadline_ms) / 1000)
                     - cfg.w_loss * loss - cfg.w_switch * switched[k])
            rewards.append(r)
            parts.append({"timely": timely, "f1": f1, "latency": latency, "loss": loss, "switch": switched[k],
                          "mode": e.mode, "fps": e.fps, "edge": e.p.name, "ta": ta,
                          "weather": self.table.weather.get(e.sequence)})
        team = float(np.mean(rewards)) - cfg.w_fair * (1 - jain([p["timely"] for p in parts]))
        self.decision += 1
        self._draw_turn()
        done = self.decision >= cfg.episode_decisions
        return self.observe(), team, done, {"edges": parts, "rewards": rewards}
