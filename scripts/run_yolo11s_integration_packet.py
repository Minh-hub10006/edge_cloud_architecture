"""Verify/extract v3 in a fresh lane, then run Edge smoke/live or Cloud receiver."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

PACKET_NAME = 'edge_video_platform_yolo11s_integration_20261009_v3.zip'


def extract_verified(packet, lane):
    with zipfile.ZipFile(packet) as archive:
        seen = set()
        for name in archive.namelist():
            path = PurePosixPath(name)
            if (name in seen or '\\' in name or path.is_absolute() or '..' in path.parts
                    or ':' in name or not path.parts or path.as_posix() != name):
                raise ValueError('unsafe or duplicate ZIP path: ' + name)
            seen.add(name)
            # The packet is source-only: never materialize links or special files.
            mode = archive.getinfo(name).external_attr >> 16
            if mode & 0o170000 not in (0, 0o100000):
                raise ValueError('non-regular ZIP entry: ' + name)
        manifest = json.loads(archive.read('RELEASE_MANIFEST.json'))
        if seen - set(manifest) - {'RELEASE_MANIFEST.json', 'RELEASE_CHECKPOINT_DELTA.json'}:
            raise ValueError('unmanifested source entry')
        for name, expected in manifest.items():
            if name not in seen or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('source manifest mismatch: ' + name)
        archive.extractall(lane)
    return len(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--node', choices=('edge', 'cloud'), default='edge')
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('run this launcher on Edge246, not Windows')
    home = Path('/home/ubuntu' if args.node == 'edge' else '/home/stu0')
    packet = (args.packet or home / PACKET_NAME).resolve(strict=True)
    expected, filename = Path(str(packet) + '.sha256').read_text(encoding='utf-8').strip().split(maxsplit=1)
    if filename != packet.name or len(expected) != 64:
        raise ValueError('invalid ZIP sidecar')
    actual = hashlib.sha256(packet.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('ZIP checksum mismatch; stop without deployment')
    print(f'{packet.name}: OK', flush=True)
    parent = (home / 'edge_video_platform_benchmark_overlap_20261004').resolve(strict=True)
    lane = Path(tempfile.mkdtemp(prefix='yolo11s_integration_v3.', dir=parent))
    print(f'INTEGRATION_LANE={lane}', flush=True)
    count = extract_verified(packet, lane)
    print(f'SOURCE_MANIFEST_PASS {count}', flush=True)
    if args.node == 'cloud':
        return subprocess.run([sys.executable, '-B', 'scripts/run_yolo11s_live.py', 'cloud'],
                              cwd=lane, check=False).returncode
    code = subprocess.run(['bash', 'scripts/run_yolo11s_integration_smoke_246.sh'],
                          cwd=lane, check=False).returncode
    if code != 0 or not args.live:
        return code
    return subprocess.run([sys.executable, '-B', 'scripts/run_yolo11s_live.py', 'edge'],
                          cwd=lane, check=False).returncode


if __name__ == '__main__':
    raise SystemExit(main())
