"""Build a portable source ZIP with POSIX entry paths and content manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint", type=Path, help="optional pre-task source ZIP; embed exact change inventory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    allowed = {".py", ".js", ".cjs", ".sh", ".md", ".toml", ".proto",
               ".yml", ".yaml", ".json", ".jsonl", ".cpp"}
    excluded = {"__pycache__", "results", "artifacts", "models", "build", "node_modules"}
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in allowed
                   and not any(part.startswith(".") or part in excluded for part in p.relative_to(root).parts))
    manifest = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    delta = None
    if args.checkpoint:
        with zipfile.ZipFile(args.checkpoint) as before:
            baseline = json.loads(before.read("RELEASE_MANIFEST.json"))
            for name, digest in baseline.items():
                if hashlib.sha256(before.read(name)).hexdigest() != digest:
                    raise ValueError(f"checkpoint hash mismatch: {name}")
        removed = sorted(set(baseline) - set(manifest))
        if removed:
            raise ValueError(f"source removed since checkpoint; review before packaging: {removed}")
        delta = {
            "checkpoint_name": args.checkpoint.name,
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "scope": "source additions/modifications against pre-task dirty-worktree snapshot; not git HEAD",
            "changed": {name: {"before_sha256": baseline.get(name), "after_sha256": digest}
                        for name, digest in manifest.items() if baseline.get(name) != digest},
            "removed": removed,
        }
    with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
        archive.writestr("RELEASE_MANIFEST.json", json.dumps(manifest, indent=2))
        if delta is not None:
            archive.writestr("RELEASE_CHECKPOINT_DELTA.json", json.dumps(delta, indent=2))
    with zipfile.ZipFile(args.output) as archive:
        assert "web/cloud_app.py" in archive.namelist()
        assert "web/live_sync.js" in archive.namelist()
        assert not any("\\" in name for name in archive.namelist())
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    print(f"{hashlib.sha256(args.output.read_bytes()).hexdigest()}  {args.output}")
    print(f"Verified {len(files)} source entries and manifest")
    if delta is not None:
        print(f"Checkpoint delta: {len(delta['changed'])} additions/modifications; no source deletions")


if __name__ == "__main__":
    main()
