"""Check packaged source manifest without modifying the source tree."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def verify(root):
    root = Path(root).resolve(strict=True)
    records = json.loads((root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    if not isinstance(records, dict) or not records:
        raise ValueError("release manifest must be a nonempty mapping")
    for name, expected in records.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError("unsafe manifest path")
        file = (root / name).resolve(strict=True)
        if not file.is_relative_to(root) or not file.is_file():
            raise ValueError("source file outside release")
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError("release hash mismatch: " + name)
    return len(records)


if __name__ == "__main__":
    print("SOURCE_MANIFEST_PASS", verify(Path(__file__).resolve().parents[1]))
