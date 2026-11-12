"""Resume the pinned v2 lane with the previously successful NPU environment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

LANE = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s_integration_v2.jb42mx09')
SDK = Path('/home/ubuntu/sigm/qairt/2.43.0.260128')
PACKET = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/npu_run/edge_packet')
DLC_SHA = '1bb6a516a06e23bb9a6022c0b537574745aee603adc2539ec8a885d78231d8b3'
DAEMON_SHA = '6e56ded92b5d8907b65889745d17f4c09c51ef20d909814a3367dad63465e006'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def candidate_env(base, prefix, sdk, lane, daemon):
    env = dict(base)
    target = str(sdk / 'lib/aarch64-oe-linux-gcc11.2')
    env['QAIRT_ROOT'] = str(sdk)
    env['LD_LIBRARY_PATH'] = f"{target}:{prefix}/lib:{env.get('LD_LIBRARY_PATH', '')}"
    env['ADSP_LIBRARY_PATH'] = ';'.join((str(sdk / 'lib/hexagon-v68/unsigned'), target,
                                       '/usr/lib/rfsa/adsp', '/dsp', '/opt'))
    env['PYTHONPATH'] = str(lane) + ':' + str(sdk / 'lib/python')
    env['SNPE_PERSISTENT'] = '1'
    env['SNPE_PERSISTENT_MODE'] = 'daemon'
    env['SNPE_DAEMON_BIN'] = str(daemon)
    return env


def run(argv, env, log, timeout=120):
    with log.open('x', encoding='utf-8') as stream:
        try:
            result = subprocess.run([str(arg) for arg in argv], env=env, cwd=LANE,
                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    tail = log.read_text(errors='replace').splitlines()[-14:]
    print(f'{log.name}: EXIT={code}', flush=True)
    print('\n'.join(tail), flush=True)
    return code


def disable_core_dumps():
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def library_evidence(env):
    evidence = []
    target = SDK / 'lib/aarch64-oe-linux-gcc11.2'
    for path in sorted(target.glob('libSnpeHtp*.so'))[:12]:
        item = {'path': str(path), 'sha256': digest(path)}
        try:
            result = subprocess.run(['ldd', str(path)], env=env, capture_output=True,
                                    text=True, timeout=10, check=False)
            item.update(exit=result.returncode, dependencies=(result.stdout + result.stderr)[:6000])
        except (OSError, subprocess.TimeoutExpired) as exc:
            item['error'] = str(exc)
        evidence.append(item)
    return evidence


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run on Edge246 Linux only')
    # Avoid core dumps from SDK cleanup on a failed build; no system-wide change.
    disable_core_dumps()
    dlc = PACKET / 'yolo11s320_w8a8_qcs6490.dlc'
    daemon = LANE / 'build/snpe-daemon-yolo11s-v2'
    if digest(dlc) != DLC_SHA or digest(daemon) != DAEMON_SHA:
        raise ValueError('model or v2 daemon checksum mismatch; stop without changes')
    subprocess.run([sys.executable, '-B', str(LANE / 'scripts/verify_release_snapshot.py')],
                   cwd=LANE, check=True)
    root = Path(tempfile.mkdtemp(prefix='daemon_env_retry.', dir=LANE))
    print(f'RETRY_DIR={root}', flush=True)
    env = candidate_env(os.environ, sys.prefix, SDK, LANE, daemon)
    report = {'status': 'STARTED', 'production_ready': False, 'diagnostic_only': True,
              'parity_gate': 'FAIL', 'quality_status': 'NOT_EVALUATED',
              'model_sha256': DLC_SHA, 'daemon_sha256': DAEMON_SHA,
              'runtime_requested': 'DSP only; no CPU fallback', 'env': {
                  key: env[key] for key in ('LD_LIBRARY_PATH', 'ADSP_LIBRARY_PATH')},
              'no_recompile': True, 'no_quantization': True}
    try:
        import numpy as np
        raw = PACKET / 'frame_000120/input_nhwc.raw'
        if raw.stat().st_size != 1228800:
            raise ValueError('pinned packet raw input size mismatch')
        report['input_sha256'] = digest(raw)
        inputs = root / 'inputs.txt'
        inputs.write_text(str(raw) + '\n', encoding='utf-8')
        control = SDK / 'bin/aarch64-oe-linux-gcc11.2/snpe-net-run'
        control_exit = run([control, '--container', dlc, '--input_list', inputs,
            '--output_dir', root / 'control', '--use_dsp', '--profiling_level', 'detailed',
            '--set_output_tensors', 'yolo11s320_float boxes,scores'], env, root / 'net_run.log')
        report['control_exit'] = control_exit
        if control_exit != 0:
            report['status'] = 'NPU_CONTROL_ENV_FAILED'
            return 1
        for name, count in (('boxes', 8400), ('scores', 168000)):
            path = root / f'control/Result_0/{name}.raw'
            values = np.fromfile(path, dtype=np.float32)
            if path.stat().st_size != count * 4 or not np.isfinite(values).all():
                raise ValueError('invalid control tensor size/finiteness: ' + name)
            if name == 'scores' and ((values < 0) | (values > 1)).any():
                raise ValueError('invalid control score range')
        report['control_contract'] = 'PASS_NOT_QUALITY_ADMISSION'
        build_exit = run([daemon, '--dlc', dlc, '--graph', 'yolo11s320_float',
            '--outputs', 'boxes,scores', '--runtime', 'dsp', '--perf', 'burst',
            '--init-cache', '--build-check'], env, root / 'daemon_build.log')
        report['daemon_build_exit'] = build_exit
        if build_exit != 0:
            report['status'] = 'CONTROL_PASS_DAEMON_BUILD_FAILED'
            return 1
        smoke_exit = run([sys.executable, '-B', LANE / 'scripts/yolo11s_edge_smoke.py',
            '--dlc', dlc, '--video', '/home/ubuntu/edge_multimodel/videos/clips_1min/traffic_1min.mp4',
            '--out', root / 'adapter_smoke', '--frames', '60', '--warmup', '3',
            '--allow-diagnostic'], env, root / 'adapter_smoke.log', timeout=300)
        report['smoke_exit'] = smoke_exit
        report['smoke_report'] = str(root / 'adapter_smoke/report.json')
        report['status'] = ('ENV_RETRY_ADAPTER_PASS_QUALITY_NOT_ADMITTED' if smoke_exit == 0
                            else 'DAEMON_BUILD_PASS_ADAPTER_FAILED')
        return 0 if smoke_exit == 0 else 1
    except Exception as exc:
        report.update(status='RETRY_ERROR', error={'type': type(exc).__name__, 'message': str(exc)})
        return 1
    finally:
        if report['status'] != 'ENV_RETRY_ADAPTER_PASS_QUALITY_NOT_ADMITTED':
            try:
                report['backend_libraries'] = library_evidence(env)
            except Exception as exc:
                report['library_evidence_error'] = str(exc)
        (root / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'status': report['status'], 'report': str(root / 'report.json')}), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
