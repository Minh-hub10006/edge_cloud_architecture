"""Host-only checks for the diagnostic gate; does not simulate SNPE numerics."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('cache_probe', ROOT / 'scripts/probe_yolo11s_cache_mode_246.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
# Historical v2 parser shape: the operator probe targets its pinned273-file lane.
SOURCE = '''void builder() {}\nint main(int argc, char** argv) {
    bool init_cache = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--init-cache") {
            init_cache = true;
        } else if (arg == "--help" || arg == "-h") {
            return 0;
        }
    }
    Snpe_Util_InitializeLogging(SNPE_LOG_LEVEL_ERROR);
}\n'''


class CacheProbeTest(unittest.TestCase):
    def test_patch_is_single_use_and_preserves_builder(self):
        changed = probe.patch_source(SOURCE)
        self.assertEqual(changed.count('arg == "--no-init-cache"'), 1)
        self.assertEqual(changed.split('int main(')[0], SOURCE.split('int main(')[0])
        with self.assertRaises(ValueError):
            probe.patch_source(changed)

    def run_case(self, on, off, smoke_exit=0):
        with tempfile.TemporaryDirectory() as temp:
            lane = Path(temp)
            native = lane / 'native/snpe_daemon_candidate'
            native.mkdir(parents=True)
            (native / 'snpe_daemon.cpp').write_text(SOURCE, encoding='utf-8')
            smoke_calls = []

            def fake_run(argv, **kwargs):
                args = [str(arg) for arg in argv]
                stream = kwargs.get('stdout')
                if 'g++' == args[0]:
                    return types.SimpleNamespace(returncode=0)
                if '--build-check' in args:
                    enabled = '--init-cache' in args
                    ok = on if enabled else off
                    stream.write('[cache-probe] init_cache=' + ('on' if enabled else 'off') + '\n')
                    stream.write('BUILD_CHECK_PASS\n' if ok else 'build failed\n')
                    return types.SimpleNamespace(returncode=0 if ok else 2)
                if any(arg.endswith('yolo11s_edge_smoke.py') for arg in args):
                    smoke_calls.append(args)
                    wrapper = Path(kwargs['env']['SNPE_DAEMON_BIN']).read_text(encoding='utf-8')
                    self.assertIn('"$@" --no-init-cache', wrapper)
                    out = Path(args[args.index('--out') + 1])
                    out.mkdir()
                    (out / 'report.json').write_text(json.dumps({'completed': 60 if smoke_exit == 0 else 0}))
                    return types.SimpleNamespace(returncode=smoke_exit)
                self.assertTrue(any(arg.endswith('verify_release_snapshot.py') for arg in args))
                return types.SimpleNamespace(returncode=0)

            def fake_digest(path):
                if path.name.endswith('.dlc'):
                    return probe.DLC_SHA
                if path.name == 'SNPEBuilder.h':
                    return probe.HEADER_SHA
                return probe.DAEMON_SHA

            resource = types.SimpleNamespace(RLIMIT_CORE=4, setrlimit=lambda *args: None)
            with patch.object(probe, 'LANE', lane), patch.object(probe, 'digest', fake_digest), \
                    patch.object(probe.sys, 'platform', 'linux'), \
                    patch.dict('sys.modules', resource=resource), \
                    patch.object(probe.subprocess, 'run', fake_run), contextlib.redirect_stdout(io.StringIO()):
                result = probe.main()
            report = json.loads(next(lane.glob('cache_mode_probe.*/report.json')).read_text())
            return result, report, smoke_calls

    def test_build_fail_never_runs_smoke(self):
        code, report, calls = self.run_case(False, False)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'CACHE_OFF_BUILD_FAILED')
        self.assertFalse(calls)

    def test_off_pass_runs_smoke_without_quality_admission(self):
        code, report, calls = self.run_case(False, True)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report['cache_hypothesis'], 'SUPPORTED_BY_BUILD_AB')
        self.assertFalse(report['production_ready'])
        self.assertEqual(report['parity_gate'], 'FAIL')

    def test_both_pass_does_not_claim_cache_cause(self):
        _, report, _ = self.run_case(True, True)
        self.assertEqual(report['cache_hypothesis'], 'INCONCLUSIVE_BOTH_BUILDS_PASS')

    def test_smoke_failure_is_not_admitted(self):
        code, report, _ = self.run_case(False, True, smoke_exit=1)
        self.assertEqual(code, 1)
        self.assertEqual(report['status'], 'CACHE_OFF_BUILD_PASS_ADAPTER_FAILED')


if __name__ == '__main__':
    unittest.main()
