"""Bounded cache on/off native build comparison in a new Edge246 directory."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

LANE = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s_integration_v2.jb42mx09')
SDK = Path('/home/ubuntu/sigm/qairt/2.43.0.260128')
PACKET = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/npu_run/edge_packet')
DLC_SHA = '1bb6a516a06e23bb9a6022c0b537574745aee603adc2539ec8a885d78231d8b3'
DAEMON_SHA = '6e56ded92b5d8907b65889745d17f4c09c51ef20d909814a3367dad63465e006'
HEADER_SHA = '8a1aa8926b3e6a99874d787d192175b74f4cccdf9a3be4cbc3f14f7478616ad2'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def patch_source(source):
    anchor = '        } else if (arg == "--help" || arg == "-h") {'
    if source.count(anchor) != 1 or '--no-init-cache' in source:
        raise ValueError('unexpected native CLI source; stop')
    source = source.replace(anchor, '        } else if (arg == "--no-init-cache") {\n'
                            '            init_cache = false;\n' + anchor)
    anchor = '    Snpe_Util_InitializeLogging(SNPE_LOG_LEVEL_ERROR);'
    if source.count(anchor) != 1:
        raise ValueError('unexpected native logging source; stop')
    return source.replace(anchor, '    std::cerr << "[cache-probe] init_cache=" << '
                          '(init_cache ? "on" : "off") << std::endl;\n' + anchor)


def main():
    if sys.platform != 'linux':
        raise SystemExit('Run with tracking Python on Edge246 Linux only')
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    dlc = PACKET / 'yolo11s320_w8a8_qcs6490.dlc'
    old_binary = LANE / 'build/snpe-daemon-yolo11s-v2'
    for path, expected in ((dlc, DLC_SHA), (old_binary, DAEMON_SHA),
                           (SDK / 'include/SNPE/SNPE/SNPEBuilder.h', HEADER_SHA)):
        if digest(path) != expected:
            raise ValueError(f'checksum mismatch: {path}; stop')
    subprocess.run([sys.executable, '-B', str(LANE / 'scripts/verify_release_snapshot.py')],
                   cwd=LANE, check=True)
    original = LANE / 'native/snpe_daemon_candidate/snpe_daemon.cpp'
    patched = patch_source(original.read_text(encoding='utf-8'))
    root = Path(tempfile.mkdtemp(prefix='cache_mode_probe.', dir=LANE))
    print(f'PROBE_DIR={root}', flush=True)
    report = {'status': 'STARTED', 'diagnostic_only': True, 'production_ready': False,
              'parity_gate': 'FAIL', 'quality_status': 'NOT_EVALUATED',
              'model_sha256': DLC_SHA, 'original_source_sha256': digest(original),
              'original_daemon_sha256': DAEMON_SHA,
              'runtime': 'DSP only; no CPU fallback', 'quantization_run': False,
              'baseline_modified': False, 'cases': {}, 'adapter_gate': 'NOT_RUN'}
    target = SDK / 'lib/aarch64-oe-linux-gcc11.2'
    env = dict(os.environ)
    env.update(QAIRT_ROOT=str(SDK),
               LD_LIBRARY_PATH=f'{target}:{sys.prefix}/lib:' + env.get('LD_LIBRARY_PATH', ''),
               ADSP_LIBRARY_PATH=';'.join((str(SDK / 'lib/hexagon-v68/unsigned'), str(target),
                                          '/usr/lib/rfsa/adsp', '/dsp', '/opt')),
               PYTHONPATH=f'{LANE}:{SDK}/lib/python',
               SNPE_PERSISTENT='1', SNPE_PERSISTENT_MODE='daemon', SNPE_DAEMON='1')

    def run(argv, name, timeout=120):
        args = [str(arg) for arg in argv]
        log = root / (name + '.log')
        with log.open('x', encoding='utf-8') as stream:
            try:
                code = subprocess.run(args, cwd=LANE, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                code = 124
        report['cases'][name] = {'argv': args, 'exit': code, 'log': str(log)}
        print(f'{name}: EXIT={code}', flush=True)
        print('\n'.join(log.read_text(errors='replace').splitlines()[-16:]), flush=True)
        return code

    try:
        source = root / 'snpe_daemon_cache_probe.cpp'
        source.write_text(patched, encoding='utf-8')
        report['probe_source_sha256'] = digest(source)
        binary = root / 'snpe-daemon-cache-probe'
        if run(['g++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                '-I' + str(SDK / 'include/SNPE'), '-I' + str(SDK / 'include/zdl'), source,
                '-L' + str(target), '-Wl,-rpath,' + str(target), '-lSNPE', '-lpthread',
                '-o', binary], 'compile') != 0:
            report['status'] = 'COMPILE_FAILED'
            return 1
        report['probe_binary_sha256'] = digest(binary)
        common = [binary, '--dlc', dlc, '--graph', 'yolo11s320_float', '--outputs',
                  'boxes,scores', '--runtime', 'dsp', '--perf', 'burst', '--build-check']
        results = {}
        for name, flag in (('cache_on', '--init-cache'), ('cache_off', '--no-init-cache')):
            code = run(common + [flag], name)
            log = (root / (name + '.log')).read_text(errors='replace')
            mode = name.removeprefix('cache_')
            results[name] = code == 0 and 'BUILD_CHECK_PASS' in log and f'init_cache={mode}' in log
        report['build_gates'] = results
        if not results['cache_off']:
            report['status'] = 'CACHE_OFF_BUILD_FAILED'
            return 1
        report['cache_hypothesis'] = ('SUPPORTED_BY_BUILD_AB' if not results['cache_on']
                                      else 'INCONCLUSIVE_BOTH_BUILDS_PASS')
        # The v2 manager always supplies --init-cache. Append the off flag last
        # in an isolated launcher, so adapter smoke uses the tested cache mode.
        wrapper = root / 'daemon-cache-off'
        wrapper.write_text('#!/bin/sh\nexec ' + shlex.quote(str(binary)) +
                           ' "$@" --no-init-cache\n', encoding='utf-8')
        wrapper.chmod(0o700)
        env['SNPE_DAEMON_BIN'] = str(wrapper)
        report['candidate_launcher'] = str(wrapper)
        code = run([sys.executable, '-B', LANE / 'scripts/yolo11s_edge_smoke.py',
                    '--dlc', dlc, '--video', '/home/ubuntu/edge_multimodel/videos/clips_1min/traffic_1min.mp4',
                    '--out', root / 'adapter_smoke', '--frames', '60', '--warmup', '3',
                    '--allow-diagnostic'], 'adapter_smoke', timeout=300)
        smoke_path = root / 'adapter_smoke/report.json'
        smoke = json.loads(smoke_path.read_text()) if smoke_path.exists() else {}
        report['adapter_report'] = str(smoke_path)
        passed = code == 0 and smoke.get('completed') == 60 and 'error' not in smoke
        report['adapter_gate'] = 'PASS' if passed else 'FAIL'
        report['status'] = ('CACHE_OFF_ADAPTER_PASS_QUALITY_NOT_ADMITTED' if passed
                            else 'CACHE_OFF_BUILD_PASS_ADAPTER_FAILED')
        return 0 if passed else 1
    except Exception as exc:
        report.update(status='PROBE_ERROR', error={'type': type(exc).__name__, 'message': str(exc)})
        return 1
    finally:
        report['model_unchanged'] = digest(dlc) == DLC_SHA
        report['pinned_v2_daemon_unchanged'] = digest(old_binary) == DAEMON_SHA
        path = root / 'report.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'status': report['status'], 'report': str(path),
                          'build_gates': report.get('build_gates'),
                          'adapter_gate': report['adapter_gate']}), flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
