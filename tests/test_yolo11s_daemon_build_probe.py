import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import yolo11s_daemon_build_probe as probe


class DaemonBuildProbeTests(unittest.TestCase):
    def test_excerpts_hash_and_bounded_context(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'SDK.h'
            content = '\n'.join(['irrelevant'] * 20 + ['SetOutputTensors'] + ['tail'] * 20)
            path.write_text(content)
            result = probe.excerpts(path, ('SetOutputTensors',), 2, 3)
            self.assertEqual(result['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(result['lines']), 6)
            self.assertIn('21: SetOutputTensors', result['lines'])

    def test_missing_file_is_evidence_not_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(probe.excerpts(Path(folder) / 'missing', ('x',))['exists'])

    def test_command_has_deadline_and_no_shell(self):
        with patch.object(probe.subprocess, 'run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = 'ok'
            run.return_value.stderr = ''
            self.assertEqual(probe.command(['ldd', 'candidate'], {})['exit'], 0)
            self.assertEqual(run.call_args.kwargs['timeout'], 10)
            self.assertNotIn('shell', run.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
