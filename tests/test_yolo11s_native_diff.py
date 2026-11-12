from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import collect_yolo11s_native_diff_246 as collector


class NativeDiffTests(unittest.TestCase):
    def test_ldd_resolution_requires_actual_resolved_library(self):
        self.assertEqual(collector.resolved_snpe('libSNPE.so => /sdk/libSNPE.so (0x123)'), ['/sdk/libSNPE.so'])
        self.assertEqual(collector.resolved_snpe('libSNPE.so => not found'), [])

    def test_sample_excerpt_includes_platform_setup_and_no_unrelated_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'build.cpp'
            path.write_text('platform setup\nSnpe_SNPEBuilder_Build(builder);\ncleanup')
            result = collector.source_excerpt(path)
            self.assertEqual(len(result['lines']), 3)
            self.assertFalse(result['truncated'])
            path.write_text('not a builder')
            self.assertIsNone(collector.source_excerpt(path))

    def test_readonly_subprocess_has_timeout_and_no_shell(self):
        with patch.object(collector.subprocess, 'run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = 'elf'
            run.return_value.stderr = ''
            self.assertEqual(collector.command(['readelf', '-d', 'binary'], {})['exit'], 0)
            self.assertEqual(run.call_args.kwargs['timeout'], 15)
            self.assertNotIn('shell', run.call_args.kwargs)


if __name__ == '__main__':
    unittest.main()
