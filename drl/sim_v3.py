"""Simulator v3 (docs/PLAN_DRL_V3B_VI.md P2): N streams on Edges of the three measured profiles, one (model, FPS)
command per stream every 5 s from a Cloud-side policy, per-second accuracy from the merged F1 table.

Edge side (measured on the v50.5 testbed, 3 streams per Edge, results/test_v50): an Edge hosts at most
`streams_per_edge` streams (the measured load) and the run adds Edges drawn from the three profiles, so the Edge side
never extrapolates. Cloud side: one shared GPU, `cloud_inflight` tickets per stream, M/D/1 wait and the closed-loop
cycle of drl/sim.py (fitted on 8-12 live LW-DETR streams); capacities = 4090 batch-1 benchmark x `gpu_efficiency`.

Per stream and second: done = analysed frames served / asked, on_time = share within the deadline (lognormal),
TA = F1 x done x on_time, goodput AG = 15 x TA (useful frames/s shown with boxes).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import random

import numpy as np

from drl.sim import _md1_ms, on_time_share

MODES = ("yolo_edge", "yolo11s_edge", "rtdetr_cloud", "dfine_x_cloud")
CLOUD = frozenset({"rtdetr_cloud", "dfine_x_cloud"})
FPS_LEVELS = (5, 10, 15)
GPU_W = (34.0, 335.0)  # W = 34 + 335 x GPU utilization (fit, see metrics)
TABLE_NAMES = {"yolov8n_int8": "yolo_edge", "yolo11s_int8": "yolo11s_edge", "rtdetrv2": "rtdetr_cloud",
               "dfine_x": "dfine_x_cloud"}


def ocr_busy(reads, workers, cfg):
    """Live OCR model (SimV3Config.ocr_read_ms): reads/s on `workers` -> (busy workers, ms per read, share served)."""
    c = cfg.ocr_read_ms_per_busy[0] + cfg.ocr_read_ms_per_busy[1] * workers
    room = 1000 - reads * c
    busy = reads * cfg.ocr_read_ms / room if room > 0 else float("inf")
    if busy < workers:
        return busy, cfg.ocr_read_ms + c * busy, 1.0
    read_ms = cfg.ocr_read_ms + c * workers  # every worker busy: the backlog grows
    return float(workers), read_ms, min(1.0, workers * 1000 / (reads * read_ms))


@dataclass(frozen=True)
class EdgeV3:
    name: str
    local_ms: dict      # per-frame model time on the Edge accelerator (testbed medians; YOLOv8n ~ .85 x YOLO11s)
    path_ms: float      # capture -> result latency of an Edge model minus model time (testbed)
    join_ms: float      # Cloud path latency minus Cloud processing (testbed)


# LAN sweep 28-29/10 (v50.6/v50.9, N 4-14, 13 valid lots; ~/sweep_lan/phan_tich_hieu_chinh.py): accepted-frame
# medians. local = edge_inference_ms; path = Edge-mode latency - local; join = D-FINE-X latency - cloud_handling_ms.
# 247 (Windows) timers tick at ~16 ms, so its parts are coarse. Earlier profile (24/10, 3 streams per Edge):
# 246 15.3/13.0, 10.1, 16.7; 247 10.9/9.3, 5.7, 6.7; 252 28.3/24.1 (YOLOv8n then on CPU), 13.4, 33.1.
EDGES_V3 = (
    EdgeV3("edge-246", {"yolo11s_edge": 14.3, "yolo_edge": 8.9}, 20.0, 21.6),
    EdgeV3("edge-247", {"yolo11s_edge": 11.5, "yolo_edge": 9.3}, 4.5, 10.8),
    EdgeV3("edge-252", {"yolo11s_edge": 27.7, "yolo_edge": 20.6}, 32.8, 49.7),
)


@dataclass
class SimV3Config:
    deadline_ms: float = 200.0
    streams_per_edge: int = 3
    cloud_inflight: int = 2
    edge_inflight: int = 2                # NPU pipeline depth (dual DLC / 2 workers)
    # 4090 batch 1 (res_capacity.jsonl, 25/10): images/s of the model call alone.
    capacity: dict = field(default_factory=lambda: {"rtdetr_cloud": 252.0, "dfine_x_cloud": 100.0})
    # Live service rate / batch-1 benchmark rate. Fit on the LAN sweep 28-29/10 (~/sweep_lan/sim_vs_sweep.py): 1.2 puts
    # D-FINE-X latency within 5 ms on every Edge and GPU util .58 (measured ~.50; batching beats batch 1). Was .8
    # (assumed): sim D-FINE-X latency 117 ms vs 31 measured. shortcut: Cloud never above ~82 req/s in the sweep, so
    # the saturated regime is extrapolated; run one Cloud-only stress lot before trusting N > 14.
    gpu_efficiency: float = 1.2
    background: tuple = (1.0, 1.0)        # GPU share left to us (other tenants)
    # Plate OCR on the same Cloud GPU (plate_lisence: the Edge crops every vehicle's plate on its own path, the Cloud
    # OCRs the crops with EasyOCR, votes per track until a read succeeds): crops/s = ocr_crops_per_vehicle x new
    # vehicles/s over every camera (results/new_vehicles.jsonl, ByteTrack), independent of the scheduler. Measured on
    # the 4090 (scripts/corun_measure.py, 29/10, D-FINE-X ORT + EasyOCR as two processes, synthetic VN plates): an OCR
    # call takes 8.7 ms alone of which 3.85 ms keep the GPU busy (one worker <= ~115 crops/s, CPU-bound); time-slicing
    # stretches each side's service time with the other's GPU share (D-FINE x (1 + 1.70 x OCR share), OCR x (1 + .41
    # x detection share)) and each side queues behind itself (M/D/1): p50 within a few ms of the measurement wherever
    # both queues stay below 100%. shortcut: synthetic plates, one OCR worker; redo with real crops / the real service.
    ocr_ms_per_crop: float = 0.0          # GPU-busy ms per crop; 0 = no OCR on the Cloud (3.85 measured)
    ocr_crops_per_vehicle: float = 1.0
    ocr_call_ms: float = 8.7
    ocr_workers: int = 4  # OCR processes: enough that the CPU part never limits (one worker ~115 crops/s)
    ocr_crops_range: tuple | None = None  # training: crops per vehicle ~ U(lo, hi) per episode
    det_stretch_per_ocr_share: float = 1.70
    ocr_stretch_per_det_share: float = .41
    # Live plate OCR model (testbed 30/10, N 4-14, the team's recognizer as its own service, ~sweep4/calib_extract.py;
    # replaces the co-run model when ocr_read_ms > 0): ocr_crops_per_vehicle = reads sent per new vehicle (VN 0.6). A
    # read takes ocr_read_ms + c x b ms, b = OCR workers busy on average (Little: b = reads/s x read ms, solved), c = a +
    # s x workers (4 workers: 6, 8: 11.5); each busy worker holds ocr_gpu_hold of the GPU and stretches every Cloud
    # model's service by 1 / (1 - hold x b) (D-FINE-X 16.5 ms -> 33 at b 2.5 -> 65-77 at b 3.9, 13 lots within 15%).
    # shortcut: fitted at D-FINE ~60-80 req/s; detection's own load on OCR is in the fit, not modelled separately.
    ocr_read_ms: float = 0.0
    ocr_read_ms_per_busy: tuple = (.5, 1.375)
    ocr_gpu_hold: float = .2
    ocr_cap_exponent: float = 0.0  # Cloud model throughput / slow^exponent (fit in scripts/sim_v3_validate_ocr.py)
    ocr_workers_choices: tuple | None = None  # training: workers drawn per episode
    latency_sigma: float = .15
    decision_s: int = 5
    episode_decisions: int = 24           # 2 min, as a testbed run
    vn_share: float = .4                  # chance a stream plays a VN-Hard sequence (VN is ~15% of the seconds)
    # Probe (state variant A, docs/PLAN_PROBE_VI.md): probe_frames D-FINE-X frames per probe_period_s per stream not
    # already on D-FINE-X; dropped with probability (GPU utilization - .8) / .2 (no free slot). 0 = off.
    probe_period_s: float = 0.0
    probe_frames: int = 2
    # Selective probing (round 24, 30/10): the scheduler sets sim.probe_plan = {stream: seconds 0..decision_s} before
    # each step; a stream's probe is the mean over its last k seconds (probe_frames frames each), the GPU pays for
    # exactly those frames. False: every stream probes once per decision (probe_period_s), as before.
    probe_selective: bool = False
    # GPU price (Cloud-side, per decision): lambda <- max(0, lambda + price_step x (GPU util - price_target)), sent to
    # every stream in the "*_price" state variants; the training reward charges lambda x the stream's GPU share.
    price_target: float = .85
    price_step: float = .5
    # Mobile uplink (docs/PLAN_DRL_V3_VI.md section 13): each Edge site draws LAN / 4G / 5G; a 4G/5G site replays a
    # per-second (uplink Mbit/s, RTT ms) trace from `net_traces` (load_net_traces). Cloud streams of one site share
    # its uplink; a JPEG is jpeg_kb (640x640 q85 measured 76 KB mean, VN + UA-DETRAC frames, 26/10).
    net_mix: dict = field(default_factory=lambda: {"lan": 1.0})
    net_traces: dict = field(default_factory=dict)   # "4g"/"5g" -> list of (ul_mbps array, rtt_ms array)
    jpeg_kb: float = 76.0
    # Domain randomization (training only): per episode gpu_efficiency ~ U(range), per site uplink x U(range).
    gpu_efficiency_range: tuple | None = None
    uplink_scale_range: tuple | None = None
    # LAN dynamics scenarios (26/10, user: LAN + dedicated Cloud is the real deployment). Any of:
    #   "surge"    correlated hard scenes: a surge starts with prob 1/surge_every_s per s, lasts U(surge_s); 70% of
    #              streams jump to hard sequences (top 30% by mean F1 gap D-FINE-X - YOLO11s at 15 FPS)
    #   "diurnal"  compressed day: hard-scene share per phase (diurnal_shares) over diurnal_period_s, random offset
    #   "thermal"  Cloud GPU heats with load (tau thermal_tau_s, steady 40 + 50 x util C) and loses up to 30% capacity
    #              above 80 C  -- ASSUMED parameters, to be measured on 249
    #   "edgeslow" an Edge's accelerator runs 2x slower (thermal throttling, as 252 at 94 C) for U(edgeslow_s)
    # scenario_random: each episode enables each listed scenario with prob .5 (training); else all listed (evaluation).
    scenarios: tuple = ()
    scenario_random: bool = False
    # True: a hard/normal switch stays in the stream's dataset, so vn_share holds under surge/diurnal (round 19 fix,
    # 30/10); False: pools across datasets (rounds <= 19 as trained)
    dynamics_keep_dataset: bool = False
    # "random": a stream jumps to a random sequence of its dataset when one ends (rounds <= 25: a new road about every
    # 55 s on UA-DETRAC); "location" (30/10): a stream is a fixed camera, it plays the sequences of one location in turn
    # and the surge/diurnal scene swaps are off (a fixed camera cannot change road).
    camera_mode: str = "random"
    vn_share_range: tuple | None = None  # training: vn_share ~ U(range) per episode
    # OCR backlog (round 22 OCR knob, 30/10): reads the workers cannot take this second queue and are read later;
    # a policy may change sim.ocr_workers between decisions. False: per-second served share only (rounds <= 21).
    ocr_backlog: bool = False
    surge_every_s: float = 120.0
    surge_s: tuple = (30, 90)
    surge_share: float = .7
    diurnal_period_s: float = 300.0
    diurnal_shares: tuple = (.2, .7, .3, .7, .6)   # early morning, rush, midday, evening rush, night
    thermal_tau_s: float = 90.0
    edgeslow_every_s: float = 150.0
    edgeslow_s: tuple = (30, 90)


class TableV3:
    """(dataset, sequence) -> list of per-second {(mode, fps): F1}."""

    def __init__(self, rows, split, names=TABLE_NAMES, probe_rows=(), count_rows=(), probe_field="probe_missed"):
        per = {}
        for r in rows:
            if r.get("split") != split or r["model"] not in names:
                continue
            second = r.get("window", r.get("second"))
            key = (r.get("dataset", "detrac"), r["sequence"])
            cell = per.setdefault(key, {}).setdefault(second, {})
            cell[(names[r["model"]], r["fps"])] = r["f1"]
        for r in count_rows:
            cell = per.get((r["dataset"], r["sequence"]), {}).get(r["second"])
            if cell is not None and r["split"] == split:
                cell["new"] = r["new_vehicles"]
        # Probe replay (results/detrac/probe_table.jsonl, results/vn_hard/probe_table_vn.jsonl): YOLO11s misses vs
        # D-FINE-X on two frames of the second; a cell without one simply yields no probe.
        for r in probe_rows:
            cell = per.get((r.get("dataset", "detrac"), r["sequence"]), {}).get(r["second"])
            if cell is not None:
                cell["probe"] = r[probe_field]
        need_keys = {(m, f) for m in MODES for f in FPS_LEVELS}
        self.seq = {k: [v[s] for s in sorted(v) if need_keys <= v[s].keys()] for k, v in per.items()}
        self.seq = {k: v for k, v in self.seq.items() if v}
        self.by_dataset = {}
        for k in sorted(self.seq):
            self.by_dataset.setdefault(k[0], []).append(k)
        # hard pool for the dynamics scenarios: top 30% sequences by mean F1 gap D-FINE-X - YOLO11s (15 FPS)
        gap = {k: float(np.mean([c[("dfine_x_cloud", 15)] - c[("yolo11s_edge", 15)] for c in v])) for k, v in self.seq.items()}
        order = sorted(gap, key=gap.get, reverse=True)
        self.hard = order[:max(1, round(.3 * len(order)))]
        self.normal = order[len(self.hard):] or order
        # the same split inside each dataset (SimV3Config.dynamics_keep_dataset)
        self.hard_by, self.normal_by = {}, {}
        for ds, keys in self.by_dataset.items():
            o = [k for k in order if k in set(keys)]
            self.hard_by[ds] = o[:max(1, round(.3 * len(o)))]
            self.normal_by[ds] = o[len(self.hard_by[ds]):] or o
        # camera locations (SimV3Config.camera_mode "location"): UA-DETRAC sequences grouped by fixed-camera background
        # (scripts/detrac_locations.py); a sequence without a group (every VN video) is its own location
        loc_file = Path(__file__).resolve().parents[1] / "results/detrac/locations.json"
        names = json.loads(loc_file.read_text(encoding="utf-8")) if loc_file.exists() else {}
        self.locations = {}
        for k in sorted(self.seq):
            self.locations.setdefault(k[0], {}).setdefault((k[0], names.get(k[1], k[1])), []).append(k)
        if not self.seq:
            raise ValueError("no second has every (model, FPS)")

    @classmethod
    def load(cls, path, split, names=TABLE_NAMES, probe_paths=(), probe_field="probe_missed"):
        """probe_field: "probe_missed" (vehicles the Edge misses) or "probe_disagree" (20 x (1 - box-agreement F1),
        scripts/probe_agreement.py)."""
        read = lambda p: [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines()]  # noqa: E731
        counts = Path("results/new_vehicles.jsonl")  # scripts/new_vehicles.py
        return cls(read(path), split, names, [r for p in probe_paths for r in read(p)],
                   read(counts) if counts.exists() else (), probe_field)


class _Stream:
    def __init__(self, edge, table, cfg, rng):
        self.edge, self.table, self.rng = edge, table, rng
        self.keep_dataset = cfg.dynamics_keep_dataset
        self.fixed = cfg.camera_mode == "location"
        if cfg.camera_mode not in ("random", "location"):
            raise ValueError(f"camera_mode {cfg.camera_mode!r}")
        datasets = sorted(table.by_dataset)
        vn = "vn_hard" in table.by_dataset and rng.random() < cfg.vn_share
        self.dataset = "vn_hard" if vn else ("detrac" if "detrac" in datasets else datasets[0])
        self.mode, self.fps = "yolo11s_edge", 15
        self.switched, self.last = False, None
        self.probe_last, self.probe_age = None, None
        self.probe_ema, self.probe_var = None, 0.0  # weighted probe history (round 27); weight grows with the frames
        self.late_sum, self.obs_t = 0.0, 0  # late-or-dropped frame-seconds and seconds since the episode began
        self.want_hard = None  # None: no dynamics scenario picks the pool (dataset pools)
        if self.fixed:
            self.clips = rng.choice(sorted(table.locations[self.dataset].values()))
            self.clip = rng.randrange(len(self.clips))
            self.key = self.clips[self.clip]
            self.cursor = rng.randrange(len(table.seq[self.key]))
        else:
            self._jump()

    def _jump(self):
        if self.fixed:  # the next sequence of the same location, from its start
            self.clip = (self.clip + 1) % len(self.clips)
            self.key, self.cursor = self.clips[self.clip], 0
            return
        if self.want_hard is None:
            pool = self.table.by_dataset[self.dataset]
        elif self.keep_dataset:
            pool = (self.table.hard_by if self.want_hard else self.table.normal_by)[self.dataset]
        else:
            pool = self.table.hard if self.want_hard else self.table.normal
        self.key = self.rng.choice(pool)
        self.cursor = self.rng.randrange(len(self.table.seq[self.key]))

    def set_hard(self, hard):
        if self.fixed:
            return
        if hard != self.want_hard:
            self.want_hard = hard
            self._jump()

    def cell(self):
        seconds = self.table.seq[self.key]
        if self.cursor >= len(seconds):
            self._jump()
            seconds = self.table.seq[self.key]
        return seconds[self.cursor]

    def lookahead(self, n):
        """Next n seconds' F1 cells (oracle only)."""
        seconds = self.table.seq[self.key]
        return seconds[self.cursor:self.cursor + n] or [self.cell()]


class SimV3:
    def __init__(self, table, cfg=None, seed=0):
        self.table, self.cfg, self.rng = table, cfg or SimV3Config(), random.Random(seed)

    def reset(self, n_streams, edges=None):
        """edges: Edge profile per stream (testbed replay); default round-robin over the three profiles,
        streams_per_edge per Edge."""
        cfg, rng = self.cfg, self.rng
        if edges is None:
            edges = [EDGES_V3[(i // cfg.streams_per_edge) % len(EDGES_V3)] for i in range(n_streams)]
        scfg = replace(cfg, vn_share=rng.uniform(*cfg.vn_share_range)) if cfg.vn_share_range else cfg
        self.streams = [_Stream(e, self.table, scfg, random.Random(rng.random())) for e in edges]
        self.host = [i // cfg.streams_per_edge for i in range(n_streams)]  # Edge instance per stream
        self.background = self.bg_base = rng.uniform(*cfg.background)
        self.downstream, self.ds_served, self.ds_slow = 0.0, 1.0, 1.0
        self.ocr_crops = rng.uniform(*cfg.ocr_crops_range) if cfg.ocr_crops_range else cfg.ocr_crops_per_vehicle
        self.ocr_workers = int(rng.choice(cfg.ocr_workers_choices)) if cfg.ocr_workers_choices else cfg.ocr_workers
        self.ocr_hist = [0.0]
        self.ocr_queue = 0.0  # reads waiting for an OCR worker (ocr_backlog)
        self.reads_window = []  # OCR arrivals per second over the last 30 s (the OCR service's counter)
        self.gpu_util, self.price = 0.0, 0.0
        self.gpu_eff = rng.uniform(*cfg.gpu_efficiency_range) if cfg.gpu_efficiency_range else cfg.gpu_efficiency
        # one link per Edge instance: (kind, trace index, cursor)
        kinds, weights = zip(*sorted(cfg.net_mix.items()))
        self.links = {}
        for h in sorted(set(self.host)):
            kind = rng.choices(kinds, weights)[0]
            if kind == "lan" or not cfg.net_traces.get(kind):
                self.links[h] = ("lan", None, 0)
            else:
                k = rng.randrange(len(cfg.net_traces[kind]))
                self.links[h] = (kind, k, rng.randrange(len(cfg.net_traces[kind][k][0])))
        self.ul_scale = {h: rng.uniform(*cfg.uplink_scale_range) if cfg.uplink_scale_range else 1.0 for h in self.links}
        self.active = {x for x in cfg.scenarios if not cfg.scenario_random or rng.random() < .5}
        self.t, self.surge_left, self.gpu_temp = 0, 0, 40.0
        self.phase_offset = rng.uniform(0, cfg.diurnal_period_s)
        self.slow = {h: 0 for h in self.links}  # seconds of 2x slowdown left per Edge instance
        if "diurnal" in self.active:
            self._diurnal(force=True)
        return self.streams

    def _diurnal(self, force=False):
        cfg = self.cfg
        phases = len(cfg.diurnal_shares)
        pos = (self.t + self.phase_offset) % cfg.diurnal_period_s
        phase = int(pos / cfg.diurnal_period_s * phases)
        if force or phase != getattr(self, "phase", None):
            self.phase = phase
            for s in self.streams:
                s.set_hard(self.rng.random() < cfg.diurnal_shares[phase])

    def _dynamics(self):
        """Advance the LAN dynamics scenarios by one second (before the second is served)."""
        cfg, rng = self.cfg, self.rng
        self.t += 1
        if "diurnal" in self.active and self.surge_left == 0:
            self._diurnal()
        if "surge" in self.active:
            if self.surge_left > 0:
                self.surge_left -= 1
                if self.surge_left == 0:  # back to the background pattern
                    if "diurnal" in self.active:
                        self._diurnal(force=True)
                    else:
                        for s in self.streams:
                            s.set_hard(False)
            elif rng.random() < 1 / cfg.surge_every_s:
                self.surge_left = rng.randint(*cfg.surge_s)
                for s in self.streams:
                    s.set_hard(rng.random() < cfg.surge_share)
        if "edgeslow" in self.active:
            for h in self.slow:
                if self.slow[h] > 0:
                    self.slow[h] -= 1
                elif rng.random() < 1 / cfg.edgeslow_every_s:
                    self.slow[h] = rng.randint(*cfg.edgeslow_s)

    def thermal_factor(self):
        if "thermal" not in self.active:
            return 1.0
        return 1 - .3 * min(1.0, max(0.0, (self.gpu_temp - 80) / 8))

    def local_ms(self, i, mode):
        return self.streams[i].edge.local_ms[mode] * (2.0 if self.slow.get(self.host[i], 0) > 0 else 1.0)

    def link_now(self, host):
        """(uplink Mbit/s, extra RTT ms) of an Edge site this second; LAN: (None, 0) - already in join_ms."""
        kind, k, cursor = self.links[host]
        if kind == "lan":
            return None, 0.0
        ul, rtt = self.cfg.net_traces[kind][k]
        return float(ul[cursor % len(ul)]) * self.ul_scale[host], float(rtt[cursor % len(rtt)])

    def _second(self):
        cfg, out = self.cfg, []
        self._dynamics()
        crops = (self.ocr_crops * sum(s.cell().get("new", 0) for s in self.streams)
                 if cfg.ocr_ms_per_crop or cfg.ocr_read_ms else 0.0)
        if cfg.ocr_read_ms:
            busy, read_ms, served = ocr_busy(crops, self.ocr_workers, cfg)
            if cfg.ocr_backlog:  # a queue left from earlier seconds keeps every worker busy until it drains
                if self.ocr_queue > 0 or served < 1.0:
                    busy, read_ms = float(self.ocr_workers), ocr_busy(1e9, self.ocr_workers, cfg)[1]
                cap = self.ocr_workers * 1000 / read_ms
                done_reads = min(crops + self.ocr_queue, cap)
                self.ocr_queue = crops + self.ocr_queue - done_reads
                served = 1.0 if self.ocr_queue == 0 else done_reads / max(1e-9, crops + self.ocr_queue)
            ds = cfg.ocr_gpu_hold * busy
            slow = 1 / (1 - min(.9, ds))
        else:
            ds = cfg.ocr_ms_per_crop / 1000 * crops  # OCR's share of the GPU this second
            slow = 1 + cfg.det_stretch_per_ocr_share * ds  # detection kernels interleaved with OCR's
        if cfg.ocr_read_ms:
            # Live: OCR stretches each D-FINE-X call (cloud_inflight calls overlap, so a call lasts inflight x the
            # throughput interval) while the throughput drops only as slow^exponent (testbed: no queue at b 3.9).
            base = {m: c * self.gpu_eff * self.bg_base * self.thermal_factor() for m, c in cfg.capacity.items()}
            caps = {m: c / slow ** cfg.ocr_cap_exponent for m, c in base.items()}
            stretch = {m: (slow - 1) * cfg.cloud_inflight * 1000 / c for m, c in base.items()}
        else:
            caps = {m: c * self.gpu_eff * self.bg_base * self.thermal_factor() / slow for m, c in cfg.capacity.items()}
            stretch = dict.fromkeys(caps, 0.0)
        cloud = [i for i, s in enumerate(self.streams) if s.mode in CLOUD]
        # Cloud: fixed point queue wait -> per-stream request rate -> GPU utilization (drl/sim.py model).
        wait, rate, util = 0.0, {}, 0.0
        bits = cfg.jpeg_kb * 8 * 1024 / 1e6  # Mbit per frame
        links = {h: self.link_now(h) for h in self.links}
        net_ms = {}
        for _ in range(4):
            svc = {i: 1000 / caps[self.streams[i].mode] + stretch[self.streams[i].mode] for i in cloud}
            cycle = 1000 * cfg.cloud_inflight * sum(1 / caps[self.streams[i].mode] for i in cloud)
            # The closed-loop cycle (every stream keeps cloud_inflight tickets queued) only binds once the GPU is full;
            # below that a stream sends at its FPS (testbed, ~50% GPU: 22-49 ms path latency).
            rtt = {i: max(self.streams[i].edge.join_ms + svc[i] + wait + net_ms.get(i, 0.0), cycle if util >= .95 else 0.0)
                   for i in cloud}
            rate = {i: min(self.streams[i].fps, cfg.cloud_inflight * 1000 / rtt[i]) for i in cloud}
            # Mobile uplink: the site's Cloud streams (and probes) share it; over capacity, frames are dropped
            # pro rata and the rest wait in the uplink queue (M/D/1 at the frame transfer time).
            for h, (ul, extra) in links.items():
                mine = [i for i in cloud if self.host[i] == h]
                if ul is None or not mine:
                    continue
                probe = self._probe_rate() * sum(self.streams[i].mode != "dfine_x_cloud" for i, x in enumerate(self.host) if x == h)
                demand = sum(rate[i] for i in mine) + probe
                cap = max(ul, 1e-3) / bits  # frames/s the uplink carries
                scale = min(1.0, cap / max(demand, 1e-9))
                tx = 1000 / cap
                for i in mine:
                    rate[i] *= scale
                    net_ms[i] = extra + tx + _md1_ms(min(demand, cap * .95), cap)
            util = sum(rate[i] / caps[self.streams[i].mode] for i in cloud) + self._probe_load(caps)
            mean_svc = (sum(svc[i] * rate[i] for i in cloud) / max(1e-9, sum(rate.values()))) if cloud else 0.0
            wait = _md1_ms(util * 1000 / mean_svc, 1000 / mean_svc) if cloud else 0.0
        self.downstream = ds
        if crops and cfg.ocr_read_ms:
            self.ds_served, self.ds_slow = served, read_ms / cfg.ocr_read_ms
        elif crops:  # OCR workers: stretched by detection's GPU share (at unstretched speed), queue behind themselves;
            # the GPU caps them too: OCR keeps up only while its GPU work fits next to detection's
            socr = cfg.ocr_call_ms * (1 + cfg.ocr_stretch_per_det_share * util / slow)
            gpu_room = max(1e-6, 1 - min(1.0, util / slow)) * 1000 / cfg.ocr_ms_per_crop  # crops/s the GPU can still do
            self.ds_served = min(1.0, cfg.ocr_workers * 1000 / (crops * socr), gpu_room / crops)
            per = crops / cfg.ocr_workers
            self.ds_slow = (socr + _md1_ms(per, 1000 / socr)) / cfg.ocr_call_ms  # OCR response time vs alone
        else:
            self.ds_served, self.ds_slow = 1.0, 1.0
        self.background = self.bg_base * max(.05, 1 - ds)  # free share as a GPU monitor shows it (floor: no /0)
        self.gpu_util = util
        if "thermal" in self.active:
            steady = 40 + 50 * min(1.0, util)
            self.gpu_temp += (steady - self.gpu_temp) * (1 - math.exp(-1 / cfg.thermal_tau_s))
        # Edge accelerator: streams of one Edge instance share it (edge_inflight frames in parallel).
        load = {}
        for i, s in enumerate(self.streams):
            if s.mode not in CLOUD:
                load[self.host[i]] = load.get(self.host[i], 0.0) + s.fps * self.local_ms(i, s.mode) / 1000 / cfg.edge_inflight
        for i, s in enumerate(self.streams):
            cell = s.cell()
            f1 = cell[(s.mode, s.fps)]
            if s.mode in CLOUD:
                done = rate[i] / s.fps
                latency = rtt[i]
            else:
                rho = load[self.host[i]]
                done = min(1.0, 1 / rho) if rho > 0 else 1.0
                ms = self.local_ms(i, s.mode)
                # path_ms was measured at 3 YOLO11s streams x 15 FPS, its queueing included: add only the extra wait.
                rho_cal = min(.95, 45 * s.edge.local_ms["yolo11s_edge"] / 1000 / cfg.edge_inflight)
                extra = (_md1_ms(min(rho, .95) * 1000 / ms, 1000 / ms) - _md1_ms(rho_cal * 1000 / ms, 1000 / ms)
                         if rho < 1 else 1000 / s.fps)
                latency = s.edge.path_ms + ms + max(0.0, extra)
            gpu_use = rate[i] / caps[s.mode] if s.mode in CLOUD else 0.0
            ul, extra_rtt = links[self.host[i]]
            on_time = on_time_share(latency, cfg.deadline_ms, cfg.latency_sigma)
            ta = f1 * done * on_time
            out.append({"f1": f1, "done": done, "on_time": on_time, "ta": ta, "latency": latency,
                        "mode": s.mode, "fps": s.fps, "dataset": s.dataset, "edge": s.edge.name, "gpu": util,
                        "gpu_use": gpu_use, "uplink": ul, "net_rtt": extra_rtt, "downstream": self.downstream,
                        "ds_served": self.ds_served, "ds_slow": self.ds_slow, "new": cell.get("new", 0),
                        "ocr_queue": self.ocr_queue, "ocr_workers": self.ocr_workers, "ocr_reads": crops,
                        "probe_v": float(np.mean(cell["probe"])) if cell.get("probe") is not None else None})
            s.cursor += 1
        self.links = {h: (kind, k, cursor + 1) for h, (kind, k, cursor) in self.links.items()}
        self.net_ms = net_ms
        return out

    def _probe_rate(self):
        return self.cfg.probe_frames / self.cfg.probe_period_s if self.cfg.probe_period_s else 0.0

    def _probe_load(self, caps):
        if self.cfg.probe_selective:
            plan = getattr(self, "probe_plan", {})
            return sum(plan.get(i, 0) * self.cfg.probe_frames / self.cfg.decision_s / caps["dfine_x_cloud"]
                       for i, s in enumerate(self.streams) if s.mode != "dfine_x_cloud")
        rate = self._probe_rate()
        return sum(rate / caps["dfine_x_cloud"] for s in self.streams if s.mode != "dfine_x_cloud")

    def _probe(self, seconds=None):
        cfg = self.cfg
        drop = max(0.0, min(1.0, (self.gpu_util - .8) / .2))
        if cfg.probe_selective:
            plan = getattr(self, "probe_plan", {})
            for i, s in enumerate(self.streams):
                k = plan.get(i, 0) if s.mode != "dfine_x_cloud" else 0
                vals = [v for sec in seconds[len(seconds) - k:] for v in [sec[i]["probe_v"]] if v is not None] if k else []
                if vals and self.rng.random() >= drop:
                    s.probe_last, s.probe_age, s.probe_n = float(np.mean(vals)), 0.0, len(vals)
                    self._probe_history(s)
                elif s.probe_age is not None:
                    s.probe_age += cfg.decision_s
            return
        for s in self.streams:
            values = s.cell().get("probe")
            if cfg.probe_period_s and values is not None and self.rng.random() >= drop:
                s.probe_last, s.probe_age, s.probe_n = float(np.mean(values)), 0.0, 1
                self._probe_history(s)
            elif s.probe_age is not None:
                s.probe_age += cfg.decision_s

    @staticmethod
    def _probe_history(s):
        """Exponentially weighted probe and its spread; a probe of more seconds weighs more (as the round-26 planner)."""
        if s.probe_ema is None:
            s.probe_ema, s.probe_var = s.probe_last, 1.0
            return
        a = min(.8, .35 * s.probe_n)
        old = s.probe_ema
        s.probe_ema = (1 - a) * old + a * s.probe_last
        s.probe_var = (1 - a) * s.probe_var + a * (s.probe_last - old) ** 2

    def observe(self, variant):
        """Per-stream observation: own action and service, its Edge, the shared GPU, and (variant "probe*") the
        last probe; "*_price" adds the GPU price and (with the probe) the stream's probe rank among all streams.
        Everything is measurable live (Edge window state + Cloud scheduler's stream registry)."""
        variant = variant.split("_fps")[0]  # the FPS-objective variants observe like their base variant
        probes = sorted(s.probe_last for s in self.streams if s.probe_last is not None)
        n = len(self.streams)
        share = sum(s.mode in CLOUD for s in self.streams) / max(1, n)
        out = []
        for s in self.streams:
            last = s.last or {"done": 1.0, "on_time": 1.0, "latency": 0.0}
            row = [float(s.mode == m) for m in MODES] + [s.fps / 15, last["done"], last["on_time"],
                   min(1.0, last["latency"] / 400), s.edge.local_ms["yolo11s_edge"] / 30, s.edge.join_ms / 50,
                   min(1.0, self.gpu_util), share, n / 30]
            if variant.startswith("probe"):
                row += [0.0 if s.probe_last is None else min(1.0, s.probe_last / 20),
                        1.0 if s.probe_age is None else min(1.0, s.probe_age / 30)]
            if "_net" in variant:  # *_net and *_net_dr: the stream's site link (Edge-measured upload rate / transport delay)
                row += [last.get("mobile", 0.0), min(1.0, last.get("uplink", 0.0) / 50), min(1.0, last.get("net_rtt", 0.0) / 200)]
            if variant.endswith("_dyn"):  # Cloud GPU temperature (nvidia-smi) + this Edge's current model time / nominal
                i = self.streams.index(s)
                row += [min(1.0, self.gpu_temp / 100), min(1.0, self.local_ms(i, "yolo11s_edge") / s.edge.local_ms["yolo11s_edge"] / 2)]
            if variant.endswith(("_ocr", "_ocr2", "_ocr3")):  # OCR load on the shared GPU (OCR service stats) + this camera's new vehicles
                h = self.ocr_hist
                row += [min(1.0, float(np.mean(h))), float(np.clip(h[-1] - np.mean(h), -1, 1)),
                        min(1.0, (s.last or {}).get("new", 0.0) / 20)]
                if variant.endswith(("_ocr2", "_ocr3")):  # OCR workers configured on the Cloud
                    row.append(self.ocr_workers / 8)
            if variant.endswith("_ocr3"):  # round 27: probe history, the camera's late share so far, recent OCR high
                high = float(np.percentile(self.reads_window, 90)) if self.reads_window else 0.0
                row += [0.0 if s.probe_ema is None else min(1.0, s.probe_ema / 20), min(1.0, np.sqrt(s.probe_var) / 10),
                        float(s.probe_ema is None), min(1.0, s.late_sum / max(5, s.obs_t) / .2),
                        min(1.0, s.obs_t / (self.cfg.episode_decisions * self.cfg.decision_s)), min(1.0, high / 200)]
            if variant.endswith(("_price", "_price2", "_net", "_dr", "_mm", "_dyn", "_ocr", "_ocr2", "_ocr3")):
                row.append(min(1.0, self.price / 2))
                if variant.startswith("probe"):  # share of streams with a smaller probe (.5 when unknown)
                    row.append(.5 if s.probe_last is None or not probes else
                               sum(v < s.probe_last for v in probes) / len(probes))
            out.append(row)
        return np.array(out, dtype=np.float32)

    def step(self, actions):
        """actions: (mode, fps) per stream -> list of per-second records for the next decision_s seconds."""
        for s, (mode, fps) in zip(self.streams, actions):
            s.switched = (mode, fps) != (s.mode, s.fps)
            s.mode, s.fps = mode, fps
        seconds = [self._second() for _ in range(self.cfg.decision_s)]
        for i, s in enumerate(self.streams):
            s.last = {k: float(np.mean([sec[i][k] for sec in seconds])) for k in ("done", "on_time", "latency")}
            # Site link over the last 5 s as the Edge measures it (LAN: no mobile link)
            ul = [sec[i]["uplink"] for sec in seconds if sec[i]["uplink"] is not None]
            s.last.update(mobile=float(bool(ul)), uplink=float(np.mean(ul)) if ul else 0.0,
                          net_rtt=float(np.mean([sec[i]["net_rtt"] for sec in seconds])),
                          new=float(np.mean([sec[i]["new"] for sec in seconds])))
        self.ocr_hist = [sec[0]["downstream"] for sec in seconds]
        self.reads_hist = [sec[0]["ocr_reads"] for sec in seconds]  # OCR arrivals per second, as the OCR service counts
        self.reads_window = (self.reads_window + self.reads_hist)[-30:]
        for i, s in enumerate(self.streams):
            s.late_sum += sum(1 - sec[i]["done"] * sec[i]["on_time"] for sec in seconds)
            s.obs_t += len(seconds)
        self._probe(seconds)
        util = float(np.mean([sec[0]["gpu"] for sec in seconds]))
        self.price = max(0.0, self.price + self.cfg.price_step * (util - self.cfg.price_target))
        return seconds


def metrics(seconds, f1_target, deadline_share=.95):
    """seconds: list (per second) of per-stream records -> headline candidates."""
    n = len(seconds[0])
    per = [[sec[i] for sec in seconds] for i in range(n)]
    # FPS view (goodput as in Clockwork / Symphony; constrained throughput as in RegenHance): frames analysed AND
    # returned within the deadline per second; drop = share of the frames to analyse that were dropped or late.
    eff = [float(np.mean([r["fps"] * r["done"] * r["on_time"] for r in p])) for p in per]
    drop = [1 - float(np.mean([r["done"] * r["on_time"] for r in p])) for p in per]
    on_time = [float(np.mean([r["done"] * r["on_time"] for r in p])) for p in per]
    f1 = [float(np.mean([r["f1"] for r in p])) for p in per]
    ta = [float(np.mean([r["ta"] for r in p])) for p in per]
    return {"ta200": float(np.mean(ta)), "goodput": 15 * float(np.sum(ta)),
            "slo_streams": int(sum(o >= deadline_share and f >= f1_target for o, f in zip(on_time, f1))),
            "on_time": float(np.mean(on_time)), "f1": float(np.mean(f1)),
            "fps_goodput": float(np.sum(eff)), "drop": float(np.mean(drop)),
            # constrained goodput: only streams with F1 >= target and drop <= 5% count (accuracy proxy: F1, not mAP)
            "fps_goodput_c70": float(sum(e for e, f, d in zip(eff, f1, drop) if f >= .70 and d <= .05)),
            "fps_goodput_c60": float(sum(e for e, f, d in zip(eff, f1, drop) if f >= .60 and d <= .05)),
            "fps_goodput_c75": float(sum(e for e, f, d in zip(eff, f1, drop) if f >= .75 and d <= .05)),
            "share_ok70": float(np.mean([f >= .70 and d <= .05 for f, d in zip(f1, drop)])),
            "gpu_util": float(np.mean([sec[0]["gpu"] for sec in seconds])),
            # Cloud GPU power: linear fit on the 4 v50.5 testbed runs (gpu.csv mean W vs util: 81 W at .14, 138 W at .31)
            "gpu_w": float(np.mean([GPU_W[0] + GPU_W[1] * min(1.0, sec[0]["gpu"]) for sec in seconds])),
            "ta_per_gpu": float(np.mean(ta)) / max(.05, float(np.mean([sec[0]["gpu"] for sec in seconds])))}
