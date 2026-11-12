"""Read SDK-native configuration and loader identities; no build/inference/write."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

SDK = Path('/home/ubuntu/sigm/qairt/2.43.0.260128')
LANE = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s_integration_v2.jb42mx09')
RETRY = LANE / 'daemon_env_retry.jp17newy'
CUES = ('Snpe_SNPEBuilder_Build', 'Snpe_Util_IsRuntimeAvailable',
        'Snpe_PlatformConfig_SetPlatformOptions', 'unsignedPD',
        'Snpe_SNPEBuilder_SetPlatformConfig', 'Snpe_SNPEBuilder_SetRuntimeProcessorOrder',
        'Snpe_Util_Initialize', 'Snpe_SNPEBuilder_SetCPUFallback',
        'Snpe_SNPEBuilder_SetCpuFallback')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            result.update(chunk)
    return result.hexdigest()


def command(argv, env):
    try:
        result = subprocess.run(argv, env=env, capture_output=True, text=True,
                                errors='replace', timeout=15, check=False)
        return {'argv': argv, 'exit': result.returncode,
                'stdout': result.stdout[:14000], 'stderr': result.stderr[:4000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': argv, 'error': str(exc)}


def resolved_snpe(ldd_stdout):
    return re.findall(r'libSNPE\.so\s+=>\s+(\S+)\s+\(', ldd_stdout)


def source_excerpt(path):
    lines = path.read_text(errors='replace').splitlines()
    selected = set()
    for i, line in enumerate(lines):
        if any(cue in line for cue in CUES):
            selected.update(range(max(0, i-10), min(len(lines), i+16)))
    if not selected:
        return None
    return {'path': str(path), 'sha256': digest(path),
            'truncated': len(selected) > 240,
            'lines': [f'{i+1}: {lines[i]}' for i in sorted(selected)[:240]]}


def collect(env):
    report = {'status': 'NATIVE_DIFF_EVIDENCE_ONLY', 'inference_run': False,
              'build_run': False, 'services_modified': False, 'source_modified': False}
    previous = RETRY / 'report.json'
    report['retry_report'] = json.loads(previous.read_text(encoding='utf-8'))
    report['binaries'] = []
    for path in (SDK / 'bin/aarch64-oe-linux-gcc11.2/snpe-net-run',
                 LANE / 'build/snpe-daemon-yolo11s-v2'):
        ldd = command(['ldd', str(path)], env)
        item = {'path': str(path), 'sha256': digest(path), 'ldd': ldd,
                'dynamic_section': command(['readelf', '-d', str(path)], env),
                'resolved_libSNPE': []}
        for library in resolved_snpe(ldd.get('stdout', '')):
            resolved = Path(library).resolve(strict=True)
            item['resolved_libSNPE'].append({'ldd_path': library,
                                             'real_path': str(resolved), 'sha256': digest(resolved)})
        report['binaries'].append(item)
    report['sdk_samples'] = []
    root = SDK / 'examples/SNPE/NativeCpp/SampleCode_CAPI'
    report['sample_root'] = str(root)
    report['sample_root_exists'] = root.is_dir()
    samples = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.cpp', '.c', '.h', '.hpp'))
    report['sample_scan_truncated'] = len(samples) > 100
    for path in samples[:100]:
        excerpt = source_excerpt(path)
        if excerpt:
            report['sdk_samples'].append(excerpt)
    report['backend_inventory'] = [str(path) for path in sorted(
        (SDK / 'lib/aarch64-oe-linux-gcc11.2').glob('*Htp*.so'))]
    return report


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run read-only collector on Edge246 Linux')
    env = os.environ.copy()
    target = str(SDK / 'lib/aarch64-oe-linux-gcc11.2')
    env['LD_LIBRARY_PATH'] = f"{target}:{sys.prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    env['ADSP_LIBRARY_PATH'] = ';'.join((str(SDK / 'lib/hexagon-v68/unsigned'), target,
                                       '/usr/lib/rfsa/adsp', '/dsp', '/opt'))
    print(json.dumps(collect(env), indent=2))


if __name__ == '__main__':
    main()
