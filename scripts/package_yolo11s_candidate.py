"""Seal an already parity-gated YOLO11s320 Float packet into a transferable ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo11s_contract import digest, verify_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    args = parser.parse_args()
    root = verify_inventory(args.packet)
    manifest = json.loads((root / "candidate.json").read_text(encoding="utf-8"))
    if manifest.get("kind") != "yolo11s320_candidate" or manifest.get("float_gate") != "PASS":
        parser.error("only a verified YOLO11s320 Float packet may be packaged")
    output = args.output.resolve()
    source = args.source_output.resolve()
    if (output.exists() or source.exists() or output == source
            or output.is_relative_to(root) or source.is_relative_to(root)):
        parser.error("choose new output paths outside the source packet")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for relative in sorted(set(manifest_files(root)) | {"BUNDLE_SHA256.json"}):
            path = root / relative
            archive.write(path, Path("yolo11s320_packet") / relative)
    with zipfile.ZipFile(output) as archive:
        if archive.read("yolo11s320_packet/BUNDLE_SHA256.json") != (root / "BUNDLE_SHA256.json").read_bytes():
            raise ValueError("packaged inventory mismatch")
        for relative, expected in json.loads((root / "BUNDLE_SHA256.json").read_text(encoding="utf-8")).items():
            if hashlib.sha256(archive.read(f"yolo11s320_packet/{relative}")).hexdigest() != expected:
                raise ValueError("packaged file checksum mismatch: " + relative)
    source.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in ("scripts/yolo11s_snpe_candidate.py", "yolo11s_contract.py"):
            archive.write(Path(__file__).resolve().parents[1] / relative, relative)
    print(f"{digest(output)}  {output}")
    print(f"Verified {len(manifest_files(root))} packet files")
    print(f"Runner source: {source} ({digest(source)})")


def manifest_files(root):
    return json.loads((root / "BUNDLE_SHA256.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
