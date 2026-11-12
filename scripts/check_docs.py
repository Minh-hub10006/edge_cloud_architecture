"""Offline release checks: Markdown links, Bash syntax and documented CLI flags.

No deployment commands are executed. Bash runs with -n on temporary snippets.
CLI options are read from source AST, without loading model/runtime dependencies.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    "README.md", "IMPLEMENTATION_PLAN.md", "docs/README.md",
    "docs/CURRENT_RUNBOOK_VI.md", "docs/CONTROL_V1_VI.md",
    "docs/BENCHMARK_RUNBOOK_VI.md", "docs/BENCHMARK_PLAN_VI.md",
    "docs/BENCHMARK_BYTETRACK_VI.md",
    "docs/BENCHMARK_OVERLAP_VI.md",
    "docs/CLOUD_MODEL_ADDITION_PLAN_VI.md", "docs/MULTIMODEL_RUNBOOK_VI.md",
    "docs/RTDETRV2_CANDIDATE_RUNBOOK_VI.md",
    "docs/LWDETR_CANDIDATE_RUNBOOK_VI.md", "docs/FOUR_MODEL_DAY1_VI.md",
    "docs/RTMDET_QNN_DIAGNOSTIC_VI.md",
    "docs/RTMDET_REBUILD_VI.md",
    "docs/RTMDET_FLOAT_CPU_RUNBOOK_VI.md",
    "docs/YOLO11S_CANDIDATE_VI.md",
    "docs/YOLO11S_INTEGRATION_VI.md", "docs/FOUR_MODEL_TWO_DAY_GATES_VI.md",
    "docs/G0_BASELINE_20261008_VI.md",
    "docs/DRL_MONITORING_VI.md", "docs/DRL_STATUS_ROADMAP_VI.md",
    "docs/CHANGELOG_20261002_VI.md", "docs/ARCHITECTURE_OVERVIEW.md",
    "docs/ARCHITECTURE_DETAILED_VI.md", "docs/PLAYBACK_FIX_20261002_VI.md",
)
CLI_SOURCES = {
    "edge_agent.control_v1": "edge_agent/control_v1.py",
    "edge_agent.phase7": "edge_agent/phase7.py",
    "benchmark": "benchmark/__main__.py",
    "scripts/candidate_artifact.py": "scripts/candidate_artifact.py",
    "scripts/rtmdet_qnn_smoke.py": "scripts/rtmdet_qnn_smoke.py",
    "scripts/rebuild_rtmdet_small.py": "scripts/rebuild_rtmdet_small.py",
    "scripts/prepare_rtmdet_rebuild_calibration.py": "scripts/prepare_rtmdet_rebuild_calibration.py",
    "scripts/check_rtmdet_snpe_outputs.py": "scripts/check_rtmdet_snpe_outputs.py",
    "scripts/rtmdet_quant_sweep.py": "scripts/rtmdet_quant_sweep.py",
    "scripts/rtmdet_float_cpu_smoke.py": "scripts/rtmdet_float_cpu_smoke.py",
    "scripts/build_yolo11s_candidate.py": "scripts/build_yolo11s_candidate.py",
    "scripts/package_yolo11s_candidate.py": "scripts/package_yolo11s_candidate.py",
    "scripts/yolo11s_snpe_candidate.py": "scripts/yolo11s_snpe_candidate.py",
}


def cli_flags(path):
    flags = {"--help"}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            flags.update(arg.value for arg in node.args
                         if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                         and arg.value.startswith("--"))
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bash", help="Bash executable; syntax checking only")
    args = parser.parse_args()
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = args.bash or (str(git_bash) if git_bash.is_file() else shutil.which("bash"))
    if not bash:
        raise SystemExit("Bash is required for syntax validation; use --bash")
    allowed = {module: cli_flags(ROOT / path) for module, path in CLI_SOURCES.items()}
    errors = []
    links = snippets = commands = 0
    for name in CURRENT_DOCS:
        path = ROOT / name
        content = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\[[^\]\n]+\]\(([^)\n]+)\)", content):
            target = match.group(1).strip().strip("<>")
            if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
                continue
            target = unquote(target.split("#", 1)[0])
            if not (path.parent / target).exists():
                errors.append(f"{name}: broken link {target}")
            links += 1
        for block in re.findall(r"^```bash\s*\n(.*?)^```", content, re.M | re.S):
            with tempfile.TemporaryDirectory(prefix="platform-doc-check-") as tmp:
                script = Path(tmp) / "snippet.sh"
                script.write_text(block, encoding="utf-8", newline="\n")
                checked = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
                if checked.returncode:
                    errors.append(f"{name}: Bash syntax: {checked.stderr.strip()}")
            snippets += 1
            logical = block.replace("\\\n", " ")
            for line in logical.splitlines():
                match = re.search(r"\s-m\s+(edge_agent\.control_v1|edge_agent\.phase7|benchmark)\b(.*)", line)
                if not match:
                    match = re.search(r"\s(scripts/(?:candidate_artifact|rtmdet_qnn_smoke|rebuild_rtmdet_small|prepare_rtmdet_rebuild_calibration|check_rtmdet_snpe_outputs|rtmdet_quant_sweep|rtmdet_float_cpu_smoke)\.py)\b(.*)", line)
                if not match:
                    continue
                module, rest = match.groups()
                unknown = set(re.findall(r"(?<!\S)(--[a-z][a-z0-9-]*)", rest)) - allowed[module]
                if unknown:
                    errors.append(f"{name}: undocumented CLI options in {module}: {sorted(unknown)}")
                commands += 1
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(CURRENT_DOCS)} docs, {links} local links, {snippets} Bash snippets, {commands} CLI commands")


if __name__ == "__main__":
    main()
