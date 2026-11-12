"""Cloud-only display/tracker A/B packet; reuse the working Edge v3 DSP daemon."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile


def extract_verified(packet, lane):
    with zipfile.ZipFile(packet) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('duplicate ZIP entries')
        for name in names:
            path = PurePosixPath(name)
            mode = archive.getinfo(name).external_attr >> 16
            if ('\\' in name or ':' in name or path.is_absolute() or '..' in path.parts
                    or path.as_posix() != name or mode & 0o170000 not in (0, 0o100000)):
                raise ValueError('unsafe ZIP entry: ' + name)
        manifest = json.loads(archive.read('RELEASE_MANIFEST.json'))
        if set(names) - set(manifest) - {'RELEASE_MANIFEST.json', 'RELEASE_CHECKPOINT_DELTA.json'}:
            raise ValueError('unmanifested ZIP entry')
        for name, digest in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('source manifest mismatch: ' + name)
        archive.extractall(lane)
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, default=Path('/home/stu0/yolo11s_tracker_review_v6.zip'))
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run on Cloud249, not PowerShell')
    packet = args.packet.resolve(strict=True)
    digest, name = Path(str(packet) + '.sha256').read_text(encoding='utf-8').strip().split(maxsplit=1)
    if name != packet.name or hashlib.sha256(packet.read_bytes()).hexdigest() != digest:
        raise ValueError('packet hash mismatch; no deployment')
    parent = Path('/home/stu0/edge_video_platform_benchmark_overlap_20261004').resolve(strict=True)
    lane = Path(tempfile.mkdtemp(prefix=packet.stem + '.', dir=parent))
    print(f'SOURCE_MANIFEST_PASS {extract_verified(packet, lane)}', flush=True)
    print(f'CLOUD_LANE={lane}', flush=True)
    print('Edge unchanged. Compare smooth_ms=0 and smooth_ms=120; add review=1 before publishing.', flush=True)
    return subprocess.call([sys.executable, '-B', 'scripts/run_yolo11s_live.py', 'cloud'], cwd=lane)


if __name__ == '__main__':
    raise SystemExit(main())
