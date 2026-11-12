"""Repeat the known zero-drop 15 FPS trial without rebuilding Edge or the model."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import time

ROOT = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s_integration_v3.9dxyacpe')
DLC = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/npu_run/edge_packet/yolo11s320_w8a8_qcs6490.dlc')


def main():
    if sys.platform != 'linux':
        raise RuntimeError('Run on Edge246')
    binary = ROOT / 'build/snpe-daemon-yolo11s-v3'
    video = Path('/home/ubuntu/edge_multimodel/videos/clips_1min/traffic_1min.mp4')
    for path in (binary, DLC, video):
        if not path.is_file():
            raise FileNotFoundError(path)
    lane = Path(tempfile.mkdtemp(prefix='live_box_review.', dir=ROOT))
    epoch = str(time.time_ns())
    env = dict(os.environ, PYTHONPATH=str(ROOT), QAIRT_ROOT='/home/ubuntu/sigm/qairt/2.43.0.260128',
               SNPE_PERSISTENT='1', SNPE_PERSISTENT_MODE='daemon', YOLO11S_SNPE_DAEMON_BIN=str(binary))
    url = f'http://192.168.180.249:8005/v1/streams/edge-246/traffic-01/yolo11s-15fps/{epoch}/live?fps=15&buffer_ms=500'
    print(f'RAW_URL={url}&smooth_ms=0&review=1', flush=True)
    print(f'SMOOTH_URL={url}&smooth_ms=120', flush=True)
    input('Open BOTH URLs, then press Enter to publish 60 seconds: ')
    argv = [sys.executable, '-B', '-u', '-m', 'edge_agent.phase7', '--video', str(video),
            '--frames', '1800', '--mode', 'yolo11s_edge', '--transport', 'grpc',
            '--grpc-target', '192.168.180.249:50073', '--rtsp-base', 'rtsp://192.168.180.249:8554',
            '--node-id', 'edge-246', '--camera-id', 'traffic-01', '--stream-id', 'yolo11s-15fps',
            '--stream-epoch', epoch, '--yolo11s-dlc', str(DLC), '--yolo11s-allow-diagnostic',
            '--yolo11s-confidence', '.1', '--yolo11s-nms-iou', '.5', '--width', '1280', '--height', '720',
            '--preview-fps', '15', '--bitrate-kbps', '2500', '--shutdown-padding-frames', '15',
            '--monitor', '--metrics-port', '0', '--monitor-jsonl', str(lane / 'monitor.jsonl')]
    print(f'REVIEW_DIR={lane}', flush=True)
    with (lane / 'edge.log').open('x', encoding='utf-8') as log:
        proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                log.write(line)
            return proc.wait()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


if __name__ == '__main__':
    raise SystemExit(main())
