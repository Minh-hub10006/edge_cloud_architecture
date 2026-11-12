import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.run_yolo11s_integration_packet import extract_verified


class IntegrationPacketTests(unittest.TestCase):
    def packet(self, path, files, digests=None):
        with zipfile.ZipFile(path, 'x') as archive:
            for name, data in files.items():
                archive.writestr(name, data)
            manifest = digests if digests is not None else {
                name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
            archive.writestr('RELEASE_MANIFEST.json', json.dumps(manifest))

    def test_verified_regular_sources_include_native_cpp(self):
        with tempfile.TemporaryDirectory() as folder:
            packet = Path(folder)/'source.zip'
            lane = Path(folder)/'lane'
            self.packet(packet, {'native/candidate.cpp': b'source', 'scripts/test.py': b'pass'})
            self.assertEqual(extract_verified(packet, lane), 2)
            self.assertEqual((lane/'native/candidate.cpp').read_bytes(), b'source')

    def test_wrong_manifest_does_not_extract(self):
        with tempfile.TemporaryDirectory() as folder:
            packet = Path(folder)/'source.zip'
            lane = Path(folder)/'lane'
            self.packet(packet, {'main.py': b'source'}, {'main.py': '0'*64})
            with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                extract_verified(packet, lane)
            self.assertFalse(lane.exists())

    def test_traversal_alias_and_unmanifested_files_rejected(self):
        for name, digests in (('../escape.py', None), ('scripts//main.py', None), ('extra.py', {})):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                packet = Path(folder)/'source.zip'
                lane = Path(folder)/'lane'
                self.packet(packet, {name: b'source'}, digests)
                with self.assertRaises(ValueError):
                    extract_verified(packet, lane)
                self.assertFalse(lane.exists())


if __name__ == '__main__':
    unittest.main()
