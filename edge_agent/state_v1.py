"""Simple real-measurement state/reward and deterministic test policy.

No counterfactual latency predictor, neural weights, cross-machine clock
subtraction or substitution of missing observations by measured zeros.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from time import monotonic_ns, time_ns
from typing import Any

from contracts import ModelMode, Placement
from contracts.control_v1 import finite, validate_feedback

FEATURES = (
    "transport_overhead_norm", "edge_backlog_norm", "cloud_inflight_norm",
    "active_path_latency_norm", "effective_fps_norm", "blur_difficulty_mean",
    "confidence_mean", "box_change_mean", "cloud_arrival_pressure_norm",
    "cloud_feedback_age_norm",
)


class FeedbackTracker:
    def __init__(self, *, window_ms=1000, clock=monotonic_ns):
        self.age_cap_ms = 2 * finite(window_ms, "window_ms", positive=True)
        self.clock = clock
        self.feedback = None
        self.received = None
        self.retired = set()
        self.rejected = 0

    def accept(self, value):
        value = validate_feedback(value)
        if self.feedback is not None:
            old = self.feedback
            instance = value["cloud_instance_id"]
            if instance in self.retired or (instance == old["cloud_instance_id"]
                                           and value["sequence"] <= old["sequence"]):
                self.rejected += 1
                return False
            if instance != old["cloud_instance_id"]:
                if len(self.retired) >= 64:
                    raise RuntimeError("too many Cloud restarts; restart this Edge session")
                self.retired.add(old["cloud_instance_id"])
        self.feedback, self.received = value, self.clock()
        return True

    def age(self, overhead_ms, *, now=None):
        if self.feedback is None:
            return None
        now = self.clock() if now is None else now
        # Without a transport observation only a lower bound is known; do not
        # silently call it a complete end-to-end age estimate.
        lower_bound = max(0, now - self.received) / 1e6 + self.feedback["measurement_to_send_ms"]
        estimate = None if overhead_ms is None else lower_bound + overhead_ms / 2
        return {"lower_bound_ms": lower_bound, "estimated_ms": estimate,
                "norm": None if estimate is None else min(1.0, estimate / self.age_cap_ms),
                "lower_bound_norm": min(1.0, lower_bound / self.age_cap_ms)}


@dataclass(frozen=True)
class Action:
    mode: ModelMode
    fps: float
    decision_id: str
    since_ns: int


class TimeWindowMonitor:
    def __init__(self, stream_key, *, window_ms=1000, fps_cap=30, latency_cap_ms=1000,
                 transport_cap_ms=100, max_samples=4096, clock=monotonic_ns,
                 error_weight=1.0, switch_weight=0.05, capability=None):
        self.key = tuple(stream_key)
        self.clock = clock
        # Static device profile (measured pipeline caps), copied into every window for the v2 state.
        self.capability = dict(capability or {})
        self.window_ns = round(finite(window_ms, "window_ms", positive=True) * 1e6)
        self.fps_cap = finite(fps_cap, "fps_cap", positive=True)
        self.latency_cap = finite(latency_cap_ms, "latency_cap_ms", positive=True)
        self.transport_cap = finite(transport_cap_ms, "transport_cap_ms", positive=True)
        self.error_weight = finite(error_weight, "error_weight")
        self.switch_weight = finite(switch_weight, "switch_weight")
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = max_samples
        self.start = clock()
        self.index = 0
        self.samples = {}
        self.events = []
        self.last_overhead = None
        self.last_overhead_at = None
        self.feedback = FeedbackTracker(window_ms=window_ms, clock=clock)
        self.overflow = 0
        # One ordered worker per stream: a high watermark rejects replay
        # without a rate-dependent dedup table growing at high source FPS.
        self.last_frame_id = -1
        self.duplicates = 0
        self.probe_last_ns = None  # last completed probe (docs/PLAN_PROBE_VI.md)

    @property
    def end(self):
        return self.start + self.window_ns

    def add(self, name, value):
        if value is None:
            return
        value = finite(value, name)
        if name in {"edge_backlog_norm", "blur_difficulty_mean", "raw_confidence",
                    "calibrated_confidence", "box_change_mean", "small_ratio"} and value > 1:
            raise ValueError(f"{name} outside [0,1]")
        sample = self.samples.setdefault(name, [])
        if len(sample) >= self.max_samples:
            self.overflow += 1
            return
        sample.append(value)

    def queue(self, pending, capacity):
        if capacity <= 0 or not 0 <= pending <= capacity:
            raise ValueError("invalid queue occupancy")
        self.add("edge_backlog_norm", pending / capacity)

    def observation(self, event):
        key = tuple(event["stream_key"])
        if key != self.key:
            raise ValueError("observation belongs to another stream")
        frame_id = event["frame_id"]
        now = self.clock()
        if type(frame_id) is not int or frame_id < 0:
            raise ValueError("invalid observation frame identity")
        if frame_id <= self.last_frame_id:
            self.duplicates += 1
            return False
        if len(self.events) >= self.max_samples:
            self.overflow += 1
            return False
        self.last_frame_id = frame_id
        self.events.append(dict(event))
        if event["status"] != "accepted":
            return True
        for name in ("transport_overhead_ms", "active_path_latency_ms", "edge_inference_ms",
                     "cloud_processing_ms", "box_change_mean", "box_gap_ms"):
            self.add(name, event.get(name))
        overhead = event.get("transport_overhead_ms")
        if overhead is not None:
            self.last_overhead, self.last_overhead_at = finite(overhead, "overhead"), now
        for value in event.get("raw_confidences", ()):
            self.add("raw_confidence", value)
        for value in event.get("calibrated_confidences", ()):
            self.add("calibrated_confidence", value)
        return True

    def finish(self, action, *, now=None, switches=0):
        now = self.clock() if now is None else now
        elapsed = (now - self.start) / 1e9
        if elapsed <= 0:
            raise ValueError("window duration must be positive")
        def mean(name):
            sample = self.samples.get(name, [])
            return sum(sample) / len(sample) if sample else None
        accepted = [event for event in self.events if event["status"] == "accepted"]
        current = [event for event in accepted if event["decision_id"] == action.decision_id]
        def current_mean(name):
            values = [event[name] for event in current if event.get(name) is not None]
            return sum(values) / len(values) if values else None
        timely = sum(event["timely"] for event in accepted)
        failures = sum(event["status"] != "accepted" or not event.get("timely", False)
                       for event in self.events)
        raw_values = [x for event in current for x in event.get("raw_confidences", ())]
        calibrated_values = [x for event in current for x in event.get("calibrated_confidences", ())]
        raw_conf = sum(raw_values) / len(raw_values) if raw_values else None
        calibrated = sum(calibrated_values) / len(calibrated_values) if calibrated_values else None
        # A raw mean is an explicitly labeled debug signal, never calibration.
        confidence = calibrated if calibrated is not None else raw_conf
        feedback = self.feedback.feedback
        # Old overhead is not valid indefinitely after a network failure.
        overhead = (self.last_overhead if self.last_overhead_at is not None
                    and now - self.last_overhead_at <= 10 * self.window_ns else None)
        age = self.feedback.age(overhead, now=now)
        raw = {
            "transport_overhead_ms": mean("transport_overhead_ms"),
            "edge_backlog_norm": mean("edge_backlog_norm"),
            "cloud_inflight_norm": None if feedback is None else feedback["cloud_inflight_norm"],
            "active_path_latency_ms": current_mean("active_path_latency_ms"),
            "effective_fps": len(accepted) / elapsed, "timely_fps": timely / elapsed,
            "blur_difficulty_mean": mean("blur_difficulty_mean"),
            "laplacian_variance_mean": mean("laplacian_variance"),
            "raw_confidence_mean": raw_conf, "calibrated_confidence_mean": calibrated,
            "box_change_mean": current_mean("box_change_mean"), "box_gap_ms": current_mean("box_gap_ms"),
            "cloud_arrival_pressure_raw": None if feedback is None else feedback["cloud_arrival_pressure_raw"],
            "cloud_arrival_pressure_norm": None if feedback is None else feedback["cloud_arrival_pressure_norm"],
            "cloud_feedback_age_ms": None if age is None else age["estimated_ms"],
            "cloud_feedback_age_lower_bound_ms": None if age is None else age["lower_bound_ms"],
            "edge_inference_ms": mean("edge_inference_ms"),
            "cloud_processing_ms": mean("cloud_processing_ms"),
            # v2 state inputs and dataset-only extras (window means; None when not measured).
            **{name: mean(name) for name in V2_RAW},
            # Probe: hardness (missed vehicles) and its costs; last value and age for the DRL v3 state.
            **{name: mean(name) for name in PROBE_RAW},
            "probe_missed_last": (self.samples.get("probe_missed") or [None])[-1],
            "probe_age_s": None if self.probe_last_ns is None else (now - self.probe_last_ns) / 1e9,
        }
        def norm(value, cap=1):
            return None if value is None else min(1.0, value / cap)
        state = {
            "transport_overhead_norm": norm(raw["transport_overhead_ms"], self.transport_cap),
            "edge_backlog_norm": raw["edge_backlog_norm"],
            "cloud_inflight_norm": raw["cloud_inflight_norm"],
            "active_path_latency_norm": norm(raw["active_path_latency_ms"], self.latency_cap),
            "effective_fps_norm": norm(raw["effective_fps"], self.fps_cap),
            "blur_difficulty_mean": raw["blur_difficulty_mean"], "confidence_mean": confidence,
            "box_change_mean": raw["box_change_mean"],
            "cloud_arrival_pressure_norm": raw["cloud_arrival_pressure_norm"],
            "cloud_feedback_age_norm": None if age is None else age["norm"],
        }
        by_action = {}
        for event in self.events:
            item = by_action.setdefault(event["decision_id"], {
                "model_id": event["model_id"], "accepted": 0, "timely": 0, "failed": 0})
            item["accepted"] += event["status"] == "accepted"
            item["timely"] += event["status"] == "accepted" and event.get("timely", False)
            item["failed"] += event["status"] != "accepted" or not event.get("timely", False)
        error_fraction = failures / len(self.events) if self.events else 0.0
        components = {"fps": norm(raw["timely_fps"], self.fps_cap),
                      "errors": self.error_weight * error_fraction,
                      "switch": self.switch_weight * switches, "quality": None}
        result = {
            "control_v1": 1, "stream_key": list(self.key), "window_id": self.index,
            "window_start_mono_ns": self.start, "window_end_mono_ns": now,
            "window_duration_s": elapsed, "state": state, "raw": raw,
            "availability_mask": {key: value is not None for key, value in state.items()},
            "sample_counts": {key: len(value) for key, value in self.samples.items()},
            "context": {"model_mode": action.mode.value, "target_fps": action.fps,
                        "decision_id": action.decision_id},
            "confidence_source": "calibrated" if calibrated is not None else "raw_debug" if raw_conf is not None else "unavailable",
            "cloud_feedback": feedback, "feedback_age_estimate_partial": age is not None and age["estimated_ms"] is None,
            "feedback_age_cap_ms": self.feedback.age_cap_ms,
            "reward": components["fps"] - components["errors"] - components["switch"],
            "reward_components": components, "reward_mode": "throughput_debug",
            "quality_constraint_active": False, "training_ready": False,
            "action_results": by_action, "frames": self.events, "capability": self.capability,
            "media_fps": self.fps_cap,
            "capacity_drops": self.overflow, "duplicate_observations": self.duplicates,
        }
        self.start = now
        self.index += 1
        self.samples, self.events = {}, []
        return result


PROBE_RAW = ("probe_missed", "probe_rtt_ms", "probe_jpeg_ms", "probe_edge_ms")
V2_RAW = ("cpu_max_core_pct", "cpu_total_pct", "ram_pct", "temp_c", "encoder_write_ms", "brightness",
          "contrast", "laplacian_small", "motion", "vehicle_count", "small_ratio", "conf_mean", "uncertain_ratio")

# Modes with a verified adapter on both sides of the control path.
SUPPORTED_MODES = (ModelMode.YOLO_EDGE, ModelMode.YOLO11S_EDGE, ModelMode.DFINE_CLOUD,
                   ModelMode.RTDETR_CLOUD, ModelMode.LWDETR_CLOUD, ModelMode.DFINEX_CLOUD,
                   ModelMode.RTDETR_SPLIT_CLOUD)
POLICIES = ("observe", "rules", "cycle", "explore", "drl", "cloud")


class RuleController:
    """Reproducible smoke/explore policies and a trained-actor hook; not an accuracy guarantee.

    explore: uniform random (mode, fps) every cooldown, seeded, for dataset collection.
    drl: actor.act(observation_v2(last 5 windows), mask) picks a mode (single FPS level); Cloud modes
    are masked while Cloud feedback is stale, exactly like the stale fallback of the other policies.
    cloud: follow this stream's Cloud scheduler command (docs/PLAN_CONTROL_V3_VI.md) when it is valid (allowed
    mode and FPS, not expired); without one, a Cloud stream falls back to the first Edge model.
    """
    def __init__(self, initial_mode, initial_fps, *, allowed_modes, fps_levels, stream_key,
                 policy="observe", cooldown_s=3, clock=monotonic_ns, seed=None, actor=None):
        self.modes = tuple(allowed_modes)
        self.levels = tuple(sorted(set(finite(x, "fps", positive=True) for x in fps_levels)))
        if initial_mode not in self.modes or not self.levels or initial_fps not in self.levels:
            raise ValueError("initial action must belong to allowed modes/FPS levels")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("allowed modes must be distinct")
        if any(mode not in SUPPORTED_MODES for mode in self.modes):
            raise ValueError(f"control v1 supports {', '.join(m.value for m in SUPPORTED_MODES)}")
        if policy not in POLICIES:
            raise ValueError("unknown policy")
        self.edge_modes = tuple(m for m in self.modes if m.placement is Placement.EDGE)
        self.cloud_modes = tuple(m for m in self.modes if m.placement is Placement.CLOUD)
        # Fixed action order shared with training: modes outer, FPS levels inner.
        self.actions = tuple((mode, fps) for mode in self.modes for fps in self.levels)
        if policy == "drl":
            if actor is None:
                raise ValueError("drl policy requires an actor")
            if len(self.levels) != 1:
                raise ValueError("drl policy chooses the model only; give exactly one FPS level")
            if tuple(actor.actions) != tuple(m.value for m in self.modes):
                raise ValueError("actor actions differ from allowed modes")
        from edge_agent.drl_policy import HISTORY
        self.history = deque(maxlen=HISTORY)
        self.actor = actor
        self.rng = random.Random(seed)
        self.policy = policy
        self.clock = clock
        self.cooldown_ns = round(finite(cooldown_s, "cooldown_s", positive=True) * 1e9)
        self.key = ":".join(map(str, stream_key))
        self.sequence = 0
        self.rejected_commands = 0
        self.action = Action(initial_mode, initial_fps, f"control-v1:{self.key}:0", clock())

    def decide(self, snapshot, *, now=None):
        now = self.clock() if now is None else now
        self.history.append(snapshot)
        current = self.action
        state, raw = snapshot["state"], snapshot["raw"]
        feedback = snapshot["cloud_feedback"]
        age = raw["cloud_feedback_age_lower_bound_ms"]
        age_cap = snapshot.get("feedback_age_cap_ms", 2 * snapshot["window_duration_s"] * 1000)
        stale = (feedback is None or age is None or age >= age_cap
                 or state["cloud_feedback_age_norm"] == 1.0)
        mode, fps, reason = current.mode, current.fps, "hold"
        if self.policy == "observe":
            return current, reason
        if self.policy == "cloud":
            return self._follow(feedback, stale, current, now)
        on_cloud = current.mode.placement is Placement.CLOUD
        # A stale/disconnected Cloud overrides cooldown only when local is available.
        if stale and on_cloud and self.edge_modes:
            mode, reason = self.edge_modes[0], "cloud_feedback_stale"
        elif now - current.since_ns < self.cooldown_ns:
            return current, "cooldown"
        elif self.policy == "cycle":
            # Debug cycle tests executor transitions, not model ranking/quality.
            next_mode = self.modes[(self.modes.index(current.mode) + 1) % len(self.modes)]
            if next_mode.placement is Placement.CLOUD and stale:
                return current, "cloud_feedback_stale"
            mode, reason = next_mode, "debug_cycle"
        elif self.policy in ("explore", "drl"):
            mask = [not (stale and m.placement is Placement.CLOUD) for m, _ in self.actions]
            if self.policy == "explore":
                choices = [a for a, ok in zip(self.actions, mask) if ok and a != (current.mode, current.fps)]
                if not choices:
                    return current, "no_alternative"
                (mode, fps), reason = self.rng.choice(choices), "explore"
            else:
                from edge_agent.drl_policy import observation_v2
                mode_mask = [not (stale and m.placement is Placement.CLOUD) for m in self.modes]
                index = self.actor.act(observation_v2(self.history, current.mode.value), mode_mask)
                mode, reason = self.modes[index], "drl"
        else:
            pressure, inflight = state["cloud_arrival_pressure_norm"], state["cloud_inflight_norm"]
            cloud_busy = (pressure is not None and pressure >= .8) or (inflight is not None and inflight >= .9)
            backlog = state["edge_backlog_norm"]
            if cloud_busy and on_cloud and self.edge_modes:
                mode, reason = self.edge_modes[0], "cloud_busy"
            elif (not on_cloud and backlog is not None and backlog >= .5
                  and not stale and not cloud_busy and pressure is not None
                  and inflight is not None and self.cloud_modes):
                mode, reason = self.cloud_modes[0], "local_backlog"
            latency = raw["active_path_latency_ms"]
            index = self.levels.index(current.fps)
            if mode == current.mode and latency is not None:
                if (latency > 1000 / current.fps or backlog is not None and backlog >= .5) and index:
                    fps, reason = self.levels[index - 1], "reduce_fps"
                elif latency < .6 * 1000 / current.fps and backlog is not None and backlog < .25 and index + 1 < len(self.levels):
                    fps, reason = self.levels[index + 1], "increase_fps"
        if (mode, fps) == (current.mode, current.fps):
            return current, reason
        self.sequence += 1
        self.action = Action(mode, fps, f"control-v1:{self.key}:{self.sequence}", now)
        return self.action, reason


    def _follow(self, feedback, stale, current, now):
        command = None if feedback is None else feedback.get("command")
        valid = (command is not None and not stale and command["valid_until_utc_ns"] > time_ns()
                 and command["model_mode"] in {m.value for m in self.modes} and float(command["fps"]) in self.levels)
        if valid:
            if command["decision_id"] == current.decision_id:
                return current, "cloud_command"
            self.sequence += 1
            self.action = Action(ModelMode(command["model_mode"]), float(command["fps"]), command["decision_id"], now)
            return self.action, "cloud_command"
        self.rejected_commands += command is not None
        if current.mode.placement is Placement.CLOUD and self.edge_modes:
            self.sequence += 1
            self.action = Action(self.edge_modes[0], current.fps, f"control-v1:{self.key}:{self.sequence}", now)
            return self.action, "no_valid_command"
        return current, "no_valid_command"


class PrometheusV1:
    def __init__(self, key):
        from prometheus_client import CollectorRegistry, Gauge
        self.registry = CollectorRegistry()
        labels = ["node", "camera", "stream", "epoch"]
        self.labels = tuple(map(str, key))
        self.feature = Gauge("edge_control_v1_state", "Normalized real state", labels + ["feature"], registry=self.registry)
        self.available = Gauge("edge_control_v1_available", "Feature availability", labels + ["feature"], registry=self.registry)
        self.raw = Gauge("edge_control_v1_raw", "Raw diagnostic measurements", labels + ["metric"], registry=self.registry)
        self.raw_available = Gauge("edge_control_v1_raw_available", "Raw measurement availability", labels + ["metric"], registry=self.registry)
        self.reward = Gauge("edge_control_v1_reward", "Throughput debug reward, not accuracy constrained", labels, registry=self.registry)
        self.server = self.thread = None

    def start(self, port):
        from prometheus_client import start_http_server
        self.server, self.thread = start_http_server(port, registry=self.registry)

    def publish(self, snapshot):
        for name, value in snapshot["state"].items():
            self.available.labels(*self.labels, name).set(value is not None)
            if value is not None:
                self.feature.labels(*self.labels, name).set(value)
        for name, value in snapshot["raw"].items():
            self.raw_available.labels(*self.labels, name).set(value is not None)
            if value is not None:
                self.raw.labels(*self.labels, name).set(value)
        self.reward.labels(*self.labels).set(snapshot["reward"])

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=5)
