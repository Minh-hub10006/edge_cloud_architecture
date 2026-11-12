"""Bind a source checkpoint and three validated RTMDet frames into one handoff ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

REFERENCE_FILES = ("source.png", "reference_input.npy", "reference_outputs.npz", "rebuild_report.json")
REFERENCE_FOLDERS = {120: "float_reference_frame120_v2", 600: "float_reference_frame600", 1200: "float_reference_frame1200"}


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"unsafe package member: {name}")
    return name


def build(source_zip, reference_root, output):
    checksum_file = Path(str(output) + ".sha256")
    if Path(output).exists() or checksum_file.exists():
        raise FileExistsError("handoff ZIP/checksum must be new; no overwrite")
    entries = {}
    with zipfile.ZipFile(source_zip) as source:
        if source.testzip() is not None or len(source.namelist()) != len(set(source.namelist())):
            raise ValueError("source ZIP integrity/duplicate gate failed")
        manifest = json.loads(source.read("RELEASE_MANIFEST.json"))
        for name, wanted in manifest.items():
            data = source.read(safe_name(name))
            if hashlib.sha256(data).hexdigest() != wanted:
                raise ValueError(f"source checksum mismatch: {name}")
            entries[name] = data
        for name in ("RELEASE_MANIFEST.json", "RELEASE_CHECKPOINT_DELTA.json"):
            entries[name] = source.read(name)
    if "scripts/rtmdet_quant_sweep.py" not in entries:
        raise ValueError("source ZIP lacks sweep script")
    for number, name in REFERENCE_FOLDERS.items():
        folder = Path(reference_root) / name
        expected = dict(line.split(None, 1)[::-1] for line in
                        (folder / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if line)
        for member in REFERENCE_FILES:
            data = (folder / member).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected.get(member):
                raise ValueError(f"reference checksum mismatch: {folder / member}")
            entries[f"sweep_references/frame_{number:06d}/{member}"] = data
    bound = {name: hashlib.sha256(data).hexdigest() for name, data in entries.items()}
    entries["SWEEP_BUNDLE_MANIFEST.json"] = json.dumps(bound, indent=2).encode("utf-8")
    entries["SWEEP_SHA256SUMS"] = "".join(f"{value}  {name}\n" for name, value in sorted(bound.items())).encode("utf-8")
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as target:
        for name, data in sorted(entries.items()):
            target.writestr(safe_name(name), data)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("handoff ZIP CRC gate failed")
        for name, wanted in bound.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != wanted:
                raise ValueError(f"handoff ZIP checksum mismatch: {name}")
    checksum = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    with checksum_file.open("x", encoding="ascii") as stream:
        stream.write(f"{checksum}  {Path(output).name}\n")
    print(f"{checksum}  {output}")
    print(f"Verified {len(bound)} source/reference entries; {Path(output).stat().st_size} bytes")
    return bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    build(args.source_zip, args.reference_root, args.output)


if __name__ == "__main__":
    main()
