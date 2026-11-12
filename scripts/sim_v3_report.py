"""Round-15b report (5 seeds per DRL variant): tables mean +- std and paper figures from the sim_v3_baselines jsonl.

  PYTHONPATH=. python -B scripts/sim_v3_report.py results/drl_v3_cal/report15b

Figures: training curves (mean +- std over seeds), constrained FPS goodput and drop vs N (LAN | LAN+5G; DRL = mean
over seeds with a +- std band, baselines = mean over episodes with +- std bars), model mix per N, one LAN+5G episode
over time (5G uplink vs streams the policy keeps on the Cloud). Seed for the testbed: best mean train reward over the
last 100 updates (train data only, never test).
"""
import json
import re
import sys
from pathlib import Path

import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

CAL = Path("results/drl_v3_cal")
SCEN = {"LAN": "results/drl_v3/baselines_test.jsonl", "LAN+5G": "results/drl_v3/baselines_test_net_traces5g.jsonl"}
LOGS = {"LAN": CAL / "eval15b_lan.log", "LAN+5G": CAL / "eval15b_lan5g.log"}
DRL = {"DRL (LAN+5G train)": "probe_net_5g_fps3", "DRL-LAN (ablation)": "probe_lan_fps3"}
BASE = {"rules_static": "Static rule", "heuristic_net": "Net heuristic", "jcab": "JCAB", "heur_thresh": "Threshold heuristic"}
MIX = (("dfine_x_cloud", "D-FINE-X (Cloud)"), ("rtdetr_cloud", "RT-DETR (Cloud)"), ("yolo11s_edge", "YOLO11s (Edge)"),
       ("yolo_edge", "YOLOv8n (Edge)"))


def load(scen):
    rows = [json.loads(x) for x in Path(SCEN[scen]).read_text(encoding="utf-8").splitlines()]
    out = {}  # label -> N -> list of rows (seeds for DRL, one row for a baseline)
    for r in rows:
        p = r["policy"]
        label = next((v for k, v in BASE.items() if p.startswith(k)), None)
        if p.startswith("DRL"):
            label = next(k for k, v in DRL.items() if re.search(v + r"_s\d$", p))
        if label:
            out.setdefault(label, {}).setdefault(r["n"], []).append(r)
    return out


def stat(rows, key="fps_goodput_c60"):
    """DRL: mean / std over seeds (each seed = mean of its episodes); baseline: over episodes."""
    if len(rows) > 1:
        v = [r[key] for r in rows]
    else:
        v = [e[key] for e in rows[0]["episodes"]]
    return float(np.mean(v)), float(np.std(v, ddof=1))


def paired_p(drl_rows, base_row, key="fps_goodput_c60"):
    """DRL (episode values averaged over seeds) vs a baseline on the same 10 simulator seeds: Wilcoxon signed-rank."""
    d = np.mean([[e[key] for e in r["episodes"]] for r in drl_rows], axis=0)
    b = np.array([e[key] for e in base_row["episodes"]])
    return float(stats.wilcoxon(d, b).pvalue) if np.any(d != b) else 1.0


def tables(data, out):
    lines = []
    for scen, d in data.items():
        ns = sorted(next(iter(d.values())))
        lines.append(f"\n## {scen}: constrained FPS goodput (F1 >= .6, drop <= 5%), mean +- std; drop % in ()\n")
        cols = list(BASE.values()) + list(DRL)
        lines.append("| N | " + " | ".join(cols) + " | DRL vs best baseline | p (Wilcoxon) |")
        lines.append("|---" * (len(cols) + 3) + "|")
        for n in ns:
            cells, means = [], {}
            for c in cols:
                m, s = stat(d[c][n])
                dr = np.mean([r["drop"] for r in d[c][n]]) * 100
                means[c] = m
                cells.append(f"{m:.1f} +- {s:.1f} ({dr:.0f})")
            best = max(BASE.values(), key=lambda c: means[c])
            drl = "DRL (LAN+5G train)"
            gain = (means[drl] / means[best] - 1) * 100 if means[best] else float("nan")
            p = paired_p(d[drl][n], d[best][n][0])
            lines.append(f"| {n} | " + " | ".join(cells) + f" | {gain:+.0f}% vs {best} | {p:.3f} |")
        lines.append(f"\nModel mix (share of stream-seconds) and mean FPS, {scen}:\n")
        lines.append("| N | policy | " + " | ".join(n for _, n in MIX) + " | mean FPS | Cloud GPU |")
        lines.append("|---" * (len(MIX) + 4) + "|")
        for n in ns:
            for c in ("Threshold heuristic",) + tuple(DRL):
                rows = d[c][n]
                mix = " | ".join(f"{np.mean([r['mix_' + k] for r in rows]):.2f}" for k, _ in MIX)
                lines.append(f"| {n} | {c} | {mix} | {np.mean([r['fps_mean'] for r in rows]):.1f} | "
                             f"{np.mean([r['gpu_util'] for r in rows]):.2f} |")
    (out / "TABLES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def training(out):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    pick = {}
    for (label, v), color in zip(DRL.items(), ("C0", "C3")):
        curves = []
        for f in sorted(CAL.glob(f"train_{v}_s[0-9].jsonl")):
            r = [json.loads(x)["reward"] for x in f.read_text(encoding="utf-8").splitlines()]
            curves.append(np.convolve(r, np.ones(20) / 20, mode="valid"))
            pick.setdefault(v, []).append((float(np.mean(r[-100:])), f.stem.split("_")[-1]))
        c = np.array(curves)
        x = np.arange(c.shape[1]) + 20
        ax.plot(x, c.mean(0), color=color, label=f"{label} ({len(c)} seeds)")
        ax.fill_between(x, c.mean(0) - c.std(0), c.mean(0) + c.std(0), color=color, alpha=.2)
    ax.set_xlabel("PPO update")
    ax.set_ylabel("Training reward (20-update moving avg)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out / "fig_training_reward.png", dpi=200)
    return {v: max(p) for v, p in pick.items()}, pick


def per_n(data, out, key, ylabel, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (scen, d) in zip(axes, data.items()):
        ns = sorted(next(iter(d.values())))
        for i, c in enumerate(BASE.values()):
            m, s = zip(*(stat(d[c][n], key) for n in ns))
            ax.errorbar(ns, m, yerr=s, marker="o", ms=3, capsize=2, color=f"C{i + 1}", label=c)
        for (c, ls), color in zip((("DRL (LAN+5G train)", "-"), ("DRL-LAN (ablation)", "--")), ("C0", "C9")):
            m, s = map(np.array, zip(*(stat(d[c][n], key) for n in ns)))
            ax.plot(ns, m, ls, lw=2.2, color=color, label=c + " (5 seeds)")
            ax.fill_between(ns, m - s, m + s, color=color, alpha=.25)
        if key == "fps_goodput_c60":
            ax.plot(ns, [15 * n for n in ns], ":", color="grey", label="Max (N x 15 FPS)")
        ax.set_title(scen)
        ax.set_xlabel("Number of streams N")
        ax.grid(alpha=.3)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)


def mix_fig(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    for row, (scen, d) in zip(axes, data.items()):
        ns = sorted(next(iter(d.values())))
        for ax, c in zip(row, ("Threshold heuristic", "DRL (LAN+5G train)")):
            bottom = np.zeros(len(ns))
            for i, (k, name) in enumerate(MIX):
                v = np.array([np.mean([r["mix_" + k] for r in d[c][n]]) for n in ns])
                ax.bar([str(n) for n in ns], v, bottom=bottom, color=f"C{i}", label=name)
                bottom += v
            ax.set_title(f"{scen}: {c}")
            ax.set_xlabel("N")
    axes[0][0].set_ylabel("Share of stream-seconds")
    axes[1][0].set_ylabel("Share of stream-seconds")
    axes[0][1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_model_mix.png", dpi=200)


def timeline_data(out, drl_actor, theta, n=12, seed=0):
    """One LAN+5G test episode: each 5G site's uplink and how many of its streams each policy keeps on the Cloud.
    Runs in its own process: torch and matplotlib drawing in one process abort on Windows (OMP Error #15)."""
    from drl.sim_v3 import CLOUD, SimV3, TableV3
    from drl.train_v3 import config, load_policy
    from scripts.sim_v3_baselines import TABLE, PROBES, HeuristicThreshold, JCAB
    test = TableV3.load(TABLE, "test", probe_paths=PROBES)
    train = TableV3.load(TABLE, "train", probe_paths=PROBES)
    pol, variant = load_policy(drl_actor)
    runs = {"DRL (LAN+5G train)": (pol, config(variant, "test", True, traces="5g")),
            "Threshold heuristic": (HeuristicThreshold(JCAB(train).profile, theta), config("probe", "test", True, traces="5g"))}
    trace = {}
    for name, (p, cfg) in runs.items():
        sim = SimV3(test, cfg, seed)
        sim.reset(n)
        sites = [h for h, (k, _, _) in sim.links.items() if k != "lan"]
        rec = []
        for _ in range(cfg.episode_decisions):
            secs = sim.step(p(sim))
            for s in secs:
                rec.append({h: (next(r["uplink"] for i, r in enumerate(s) if sim.host[i] == h),
                                sum(r["mode"] in CLOUD for i, r in enumerate(s) if sim.host[i] == h),
                                np.mean([r["done"] * r["on_time"] for i, r in enumerate(s) if sim.host[i] == h]))
                            for h in sites})
        trace[name] = (sites, [{str(h): v for h, v in r.items()} for r in rec])
    (out / "timeline.json").write_text(json.dumps(trace), encoding="utf-8")


def timeline(out, drl_actor, theta):
    subprocess.run([sys.executable, "-B", __file__, "--timeline-data", str(out), drl_actor, str(theta)], check=True)
    trace = json.loads((out / "timeline.json").read_text(encoding="utf-8"))
    sites = [str(h) for h in trace["DRL (LAN+5G train)"][0]]
    if not sites:
        print("timeline: no 5G site in this episode, pick another seed")
        return
    fig, axes = plt.subplots(3, len(sites), figsize=(5 * len(sites), 7), sharex=True, squeeze=False)
    for j, h in enumerate(sites):
        t = np.arange(len(trace["DRL (LAN+5G train)"][1]))
        axes[0][j].plot(t, [r[h][0] for r in trace["DRL (LAN+5G train)"][1]], color="k")
        axes[0][j].set_title(f"Edge instance {h} on 5G")
        for (name, (_, rec)), color in zip(trace.items(), ("C0", "C4")):
            axes[1][j].step(t, [r[h][1] for r in rec], where="post", color=color, label=name)
            axes[2][j].plot(t, [r[h][2] for r in rec], color=color, label=name)
        axes[2][j].set_xlabel("Time (s)")
    axes[0][0].set_ylabel("5G uplink (Mbit/s)")
    axes[1][0].set_ylabel("Streams on Cloud")
    axes[2][0].set_ylabel("On-time share")
    axes[1][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig_timeline_5g.png", dpi=200)


def main():
    if sys.argv[1:2] == ["--timeline-data"]:
        return timeline_data(Path(sys.argv[2]), sys.argv[3], float(sys.argv[4]))
    out = Path(sys.argv[1] if len(sys.argv) > 1 else CAL / "report15b")
    out.mkdir(parents=True, exist_ok=True)
    data = {s: load(s) for s in SCEN}
    for scen, d in data.items():  # sanity: every policy at every N, 5 seeds per DRL variant, 10 episodes each
        for c, by_n in d.items():
            for n, rows in by_n.items():
                assert len(rows) == (5 if c in DRL else 1), (scen, c, n, len(rows))
                assert all(len(r["episodes"]) == 10 for r in rows), (scen, c, n)
    tables(data, out)
    best, pick = training(out)
    print("\nmean train reward, last 100 updates:", {v: sorted(p, reverse=True) for v, p in pick.items()})
    print("testbed seed (best train reward):", best)
    per_n(data, out, "fps_goodput_c60", "Constrained FPS goodput (frames/s)", "fig_goodput_vs_n.png")
    per_n(data, out, "drop", "Dropped or late frames (share)", "fig_drop_vs_n.png")
    mix_fig(data, out)
    theta = float(re.search(r"theta tuned on train \(fps_goodput_c60\): ([\d.]+)", LOGS["LAN+5G"].read_text()).group(1))
    timeline(out, str(CAL / f"actor_probe_net_5g_fps3_{best['probe_net_5g_fps3'][1]}.pt"), theta)
    print("written to", out)


if __name__ == "__main__":
    main()
