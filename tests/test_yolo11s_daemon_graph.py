from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from legacy.adapters.npu import snpe_daemon_manager as manager
from legacy.adapters.npu.snpe_dlc_session import SnpeDlcSession


class GraphDaemonTests(unittest.TestCase):
    def test_graph_scoped_binary_gate_precedes_socket_reuse(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {
                'SNPE_DAEMON_SOCK_DIR': folder}), patch.object(manager, 'SnpeDaemonClient') as client, patch.object(
                manager.subprocess, 'run', return_value=MagicMock(returncode=1, stdout='old')), patch.object(
                manager.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'legacy binary rejected'):
                manager.SnpeDaemonHandle('model.dlc', socket_path=str(Path(folder)/'x.sock'),
                    runtime='dsp', perf_profile='burst', bin_path='old-daemon',
                    output_tensors=('boxes', 'scores'), input_bytes=1228800, graph_name='yolo11s320_float')
            client.return_value.ping.assert_not_called()
            process.assert_not_called()

    def test_graph_name_passed_without_changing_output_names(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {
                'SNPE_DAEMON_SOCK_DIR': folder}), patch.object(manager, 'SnpeDaemonClient') as client, patch.object(
                manager.subprocess, 'run', return_value=MagicMock(
                    returncode=0, stdout='SNPE_DAEMON_GRAPH_DSP_V3\n')), patch.object(manager.subprocess, 'Popen') as process:
            client.return_value.ping.return_value = True
            manager.SnpeDaemonHandle('model.dlc', socket_path=str(Path(folder)/'x.sock'),
                runtime='dsp', perf_profile='burst', bin_path='candidate-daemon',
                output_tensors=('boxes', 'scores'), input_bytes=1228800, graph_name='yolo11s320_float')
            argv = process.call_args.args[0]
            self.assertEqual(argv[argv.index('--graph')+1], 'yolo11s320_float')
            self.assertEqual(argv[argv.index('--runtime')+1], 'dsp')
            self.assertEqual(argv[argv.index('--outputs')+1], 'boxes,scores')
            self.assertIn('--no-init-cache', argv)
            self.assertNotIn('--init-cache', argv)

    def test_candidate_binary_does_not_replace_baseline(self):
        with patch.dict('os.environ', {'SNPE_DAEMON_BIN': '/baseline',
                                     'YOLO11S_SNPE_DAEMON_BIN': '/candidate'}), patch.object(
                manager.os.path, 'isfile', return_value=True):
            self.assertEqual(manager._daemon_bin(), '/baseline')
            self.assertEqual(manager._daemon_bin('yolo11s320_float'), '/candidate')
            self.assertEqual(manager._daemon_bin('yolov8n_320_split'), '/candidate')  # Edge252 YOLOv8n

    def test_v2_daemon_rejected_before_cache_off_launch(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {
                'SNPE_DAEMON_SOCK_DIR': folder}), patch.object(manager, 'SnpeDaemonClient'), patch.object(
                manager.subprocess, 'run', return_value=MagicMock(returncode=0, stdout='SNPE_DAEMON_GRAPH_DSP_V2')), patch.object(
                manager.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'candidate daemon v3'):
                manager.SnpeDaemonHandle('model.dlc', socket_path=str(Path(folder)/'x.sock'),
                    runtime='dsp', perf_profile='burst', bin_path='v2',
                    output_tensors=('boxes', 'scores'), input_bytes=1228800, graph_name='yolo11s320_float')
            process.assert_not_called()

    def test_strict_session_forwards_graph(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'candidate.dlc'
            path.write_bytes(b'fake')
            with patch.object(manager, 'try_get_daemon_client', return_value=MagicMock()) as client:
                session = SnpeDlcSession(str(path), require_daemon=True, output_tensors=('boxes', 'scores'),
                                        input_bytes=1228800, graph_name='yolo11s320_float')
                self.assertEqual(session.mode, 'snpe_daemon')
                self.assertEqual(client.call_args.kwargs['graph_name'], 'yolo11s320_float')

    def test_explicit_missing_binary_never_falls_back_to_baseline(self):
        with patch.dict('os.environ', {'SNPE_DAEMON_BIN': '/missing/task-local-daemon'}), patch.object(
                manager.os.path, 'isfile', return_value=False):
            with self.assertRaises(FileNotFoundError):
                manager._daemon_bin()

    def test_candidate_native_source_has_no_cpu_assignment_or_legacy_serve(self):
        source = (Path(__file__).resolve().parents[1]/'native/snpe_daemon_candidate/snpe_daemon.cpp').read_text(encoding='utf-8')
        self.assertNotIn('runtime = SNPE_RUNTIME_CPU_FLOAT32;', source)
        self.assertIn('(legacy_output_api && !build_check)', source)
        self.assertIn('runtime_str != "dsp"', source)
        self.assertIn('Snpe_SNPEBuilder_SetOutputTensorsForNetwork(builder, graph.c_str(), outs)', source)
        self.assertIn('checked("NetworkInit"', source)


if __name__ == '__main__':
    unittest.main()


class SharedDaemonTests(unittest.TestCase):
    """Edge246, 4 streams (Oct 28): a busy shared daemon missed one ping and a stream's cleanup killed it."""

    def _handle(self, folder, ping):
        sock = Path(folder) / 'x.sock'
        sock.write_text('')
        with patch.dict('os.environ', {'SNPE_DAEMON_SOCK_DIR': folder}), patch.object(
                manager, 'SnpeDaemonClient') as client, patch.object(manager.subprocess, 'Popen') as process, patch.object(
                manager.time, 'sleep'):
            client.return_value.ping.side_effect = ping
            handle = manager.SnpeDaemonHandle('model.dlc', socket_path=str(sock), runtime='dsp', perf_profile='burst',
                                              bin_path='daemon', output_tensors=('boxes', 'scores'), input_bytes=4)
            return handle, client, process, sock

    def test_busy_daemon_is_reused_after_missed_pings(self):
        with tempfile.TemporaryDirectory() as folder:
            handle, _client, process, sock = self._handle(folder, [False, False, True])
            process.assert_not_called()
            self.assertTrue(sock.exists())

    def test_reused_daemon_survives_close(self):
        with tempfile.TemporaryDirectory() as folder:
            handle, client, _process, sock = self._handle(folder, [True])
            handle.close()
            client.return_value.shutdown.assert_not_called()
            self.assertTrue(sock.exists())

    def test_each_process_gets_its_own_daemon(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict('os.environ', {'SNPE_DAEMON_SOCK_DIR': folder}),                 patch.object(manager, 'SnpeDaemonHandle') as handle, patch.object(manager, '_DAEMONS', {}),                 patch.object(manager, '_daemon_bin', return_value=__file__):
            socks = []
            for pid in (101, 102):
                with patch.object(manager.os, 'getpid', return_value=pid):
                    manager.get_daemon_handle('model.dlc')
                socks.append(handle.call_args.kwargs['socket_path'])
                manager._DAEMONS.clear()
            self.assertNotEqual(socks[0], socks[1])
