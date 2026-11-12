"""Read-only daemon/SDK evidence after G1 build1007; no inference or service changes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SDK = Path('/home/ubuntu/sigm/qairt/2.43.0.260128')
LANE = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s_integration.GtT1jX')
DAEMON = Path('/home/ubuntu/edge_artifacts/bin/snpe-daemon')
PACKET = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/npu_run/edge_packet')
PINNED_DLC = '1bb6a516a06e23bb9a6022c0b537574745aee603adc2539ec8a885d78231d8b3'


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def excerpts(path, cues, before=8, after=12):
    if not path.is_file():
        return {'path': str(path), 'exists': False}
    lines = path.read_text(errors='replace').splitlines()
    selected = set()
    for i, line in enumerate(lines):
        if any(cue in line for cue in cues):
            selected.update(range(max(0, i-before), min(len(lines), i+after+1)))
    return {'path': str(path), 'sha256': digest(path),
            'lines': [f'{i+1}: {lines[i]}' for i in sorted(selected)]}


def command(argv, env):
    try:
        result = subprocess.run(argv, env=env, capture_output=True, text=True,
                                errors='replace', timeout=10, check=False)
        return {'argv': argv, 'exit': result.returncode,
                'stdout': result.stdout[:16000], 'stderr': result.stderr[:4000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'error': str(exc)}


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run on Edge246 Linux only; no changes made')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(SDK / 'lib/aarch64-oe-linux-gcc11.2') + ':' + env.get('LD_LIBRARY_PATH', '')
    report = {'status': 'DAEMON_BUILD_EVIDENCE_ONLY', 'inference_run': False,
              'services_modified': False, 'model_modified': False,
              'daemon': {'path': str(DAEMON), 'exists': DAEMON.is_file()},
              'sdk': str(SDK), 'source': [], 'headers': []}
    if DAEMON.is_file():
        report['daemon']['sha256'] = digest(DAEMON)
        report['daemon']['ldd'] = command(['ldd', str(DAEMON)], env)
        report['daemon']['help'] = command([str(DAEMON), '--help'], env)
    dlc = PACKET / 'yolo11s320_w8a8_qcs6490.dlc'
    report['dlc'] = {'path': str(dlc), 'exists': dlc.is_file()}
    if dlc.is_file():
        actual = digest(dlc)
        report['dlc'].update(sha256=actual, pinned_match=actual == PINNED_DLC)
    for folder in ('include/SNPE', 'include/zdl'):
        for relative in ('SNPE/SNPEBuilder.h', 'SNPE/SNPEUtil.h',
                         'DlSystem/PlatformConfig.h', 'DlContainer/DlContainer.h'):
            path = SDK / folder / relative
            if path.is_file():
                report['headers'].append(excerpts(path, ('SetOutputTensors', 'NetworkInit',
                    'SetPlatformOptions', 'IsRuntimeAvailable', 'NetworkNames', 'UnsignedPD'), 12, 5))
    for path in (Path('/home/ubuntu/edge_multimodel/native/snpe_daemon/snpe_daemon.cpp'),
                 Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/native/snpe_daemon/snpe_daemon.cpp')):
        report['source'].append(excerpts(path, ('build_snpe(', 'SetOutputTensors',
            'SetOutputLayers', 'SetPlatformConfig', 'SetPlatformOptions', 'falling back'), 5, 12))
    report['cache_info'] = excerpts(PACKET / 'qcs6490_info.log',
                                    ('Info of graph:', 'Total Subnets:', 'Identifier'), 0, 2)
    smoke_report = LANE / 'adapter_smoke/report.json'
    if smoke_report.is_file():
        smoke = json.loads(smoke_report.read_text())
        report['smoke_failure'] = {k: smoke.get(k) for k in ('status', 'completed', 'error')}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
