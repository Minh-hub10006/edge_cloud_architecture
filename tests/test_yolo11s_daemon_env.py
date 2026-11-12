import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from legacy.adapters.npu.snpe_env import setup_qairt_env, setup_yolo11s_candidate_env
from scripts import resume_yolo11s_daemon_env_246 as retry


class CandidateEnvTests(unittest.TestCase):
    def test_legacy_env_unchanged_candidate_matches_successful_video(self):
        with patch.dict(os.environ, {'QAIRT_ROOT': '/sdk', 'ADSP_LIBRARY_PATH': 'baseline',
                                     'LD_LIBRARY_PATH': '/original'}), patch('sys.prefix', '/tracking'):
            baseline = setup_qairt_env()
            candidate = setup_yolo11s_candidate_env()
            self.assertEqual(baseline['ADSP_LIBRARY_PATH'], 'baseline')
            self.assertEqual(candidate['ADSP_LIBRARY_PATH'],
                '/sdk/lib/hexagon-v68/unsigned;/sdk/lib/aarch64-oe-linux-gcc11.2;/usr/lib/rfsa/adsp;/dsp;/opt')
            self.assertEqual(candidate['LD_LIBRARY_PATH'], '/sdk/lib/aarch64-oe-linux-gcc11.2:/tracking/lib:/original')
            self.assertEqual(os.environ['ADSP_LIBRARY_PATH'], 'baseline')

    def test_resume_wrapper_has_same_library_paths_and_no_cpu_fallback(self):
        env = retry.candidate_env({'LD_LIBRARY_PATH': '/original'}, '/tracking', PurePosixPath('/sdk'),
                                  PurePosixPath('/lane'), PurePosixPath('/lane/daemon'))
        self.assertEqual(env['LD_LIBRARY_PATH'], '/sdk/lib/aarch64-oe-linux-gcc11.2:/tracking/lib:/original')
        self.assertIn('/sdk/lib/aarch64-oe-linux-gcc11.2;', env['ADSP_LIBRARY_PATH'])
        self.assertTrue(env['ADSP_LIBRARY_PATH'].endswith(';/opt'))
        self.assertEqual(env['SNPE_DAEMON_BIN'], '/lane/daemon')

    def run_retry(self, exits):
        with tempfile.TemporaryDirectory() as folder:
            lane = Path(folder)/'lane'
            packet = Path(folder)/'packet'
            lane.mkdir()
            (packet/'frame_000120').mkdir(parents=True)
            (packet/'frame_000120/input_nhwc.raw').write_bytes(bytes(1228800))
            def fake_digest(path):
                return retry.DAEMON_SHA if path.name == 'snpe-daemon-yolo11s-v2' else retry.DLC_SHA
            commands = []
            def fake_run(argv, env, log, timeout=120):
                commands.append([str(arg) for arg in argv])
                code = exits[len(commands)-1]
                if log.name == 'net_run.log' and code == 0:
                    output = log.parent/'control/Result_0'
                    output.mkdir(parents=True)
                    for name, count in (('boxes', 8400), ('scores', 168000)):
                        (output/f'{name}.raw').write_bytes(np.full(count, .5, np.float32).tobytes())
                return code
            with patch.object(retry, 'LANE', lane), patch.object(retry, 'PACKET', packet), patch.object(
                    retry.sys, 'platform', 'linux'), patch.object(retry, 'disable_core_dumps'), patch.object(
                    retry, 'digest', side_effect=fake_digest), patch.object(retry.subprocess, 'run'), patch.object(
                    retry, 'run', side_effect=fake_run), patch.object(retry, 'library_evidence', return_value=[]):
                code = retry.main()
            report = json.loads(next(lane.glob('daemon_env_retry.*/report.json')).read_text(encoding='utf-8'))
            return code, commands, report

    def test_failed_control_stops_before_daemon(self):
        code, commands, report = self.run_retry([1])
        self.assertEqual(code, 1)
        self.assertEqual(len(commands), 1)
        self.assertIn('--use_dsp', commands[0])
        self.assertNotIn('--enable_cpu_fallback', commands[0])
        self.assertEqual(report['status'], 'NPU_CONTROL_ENV_FAILED')

    def test_failed_daemon_stops_before_smoke(self):
        code, commands, report = self.run_retry([0, 2])
        self.assertEqual(code, 1)
        self.assertEqual(len(commands), 2)
        self.assertIn('--build-check', commands[1])
        self.assertEqual(report['status'], 'CONTROL_PASS_DAEMON_BUILD_FAILED')

    def test_success_runs_persistent_smoke_without_quality_admission(self):
        code, commands, report = self.run_retry([0, 0, 0])
        self.assertEqual(code, 0)
        self.assertEqual(len(commands), 3)
        self.assertIn('--allow-diagnostic', commands[2])
        self.assertFalse(report['production_ready'])
        self.assertEqual(report['parity_gate'], 'FAIL')
        self.assertEqual(report['status'], 'ENV_RETRY_ADAPTER_PASS_QUALITY_NOT_ADMITTED')


if __name__ == '__main__':
    unittest.main()
