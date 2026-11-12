"""Run the shared YOLO11s live path after the cache-off adapter gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DLC = Path('/home/ubuntu/edge_video_platform_benchmark_overlap_20261004/yolo11s320_split_diag.NHBFPo/npu_run/edge_packet/yolo11s320_w8a8_qcs6490.dlc')
DLC_SHA = '1bb6a516a06e23bb9a6022c0b537574745aee603adc2539ec8a885d78231d8b3'
VIDEO = Path('/home/ubuntu/edge_multimodel/videos/clips_1min/traffic_1min.mp4')
CLOUD_HOST = '192.168.180.249'
HTTP_PORT = 8005
GRPC_PORT = 50073


def reachable(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def child_env():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    return env


def cloud():
    for port in (HTTP_PORT, GRPC_PORT):
        if reachable('127.0.0.1', port):
            raise RuntimeError(f'candidate port {port} occupied; stop, retain existing process')
    env = child_env()
    env.update(CLOUD_TRANSPORT='grpc', CLOUD_MODELS='none', CLOUD_GRPC_BIND=f'0.0.0.0:{GRPC_PORT}',
               CLOUD_CONTROL_V1='0', CLOUD_BENCHMARK='0', CLOUD_PREVIEW_FPS='30',
               CLOUD_RTSP_BASE='rtsp://127.0.0.1:8554',
               CLOUD_MEDIAMTX_WEBRTC_BASE=f'http://{CLOUD_HOST}:8889',
               CLOUD_RUN_ID='yolo11s-cache-off-v3',
               # Offline replay of the v4 browser trace: median track life 1 -> 20 frames.
               CLOUD_BYTETRACK_OPTIONS='{"match_thresh":0.9,"track_high_thresh":0.25,"new_track_thresh":0.2}',
               # v5 trace: 433/463 overlaps were car/bus/truck labels on one vehicle (median IoU .84).
               CLOUD_VEHICLE_MERGE_IOU='0.6')
    env.pop('CLOUD_KAFKA_NAMESPACE', None)
    env.pop('CLOUD_ENABLE_OVERLAY_SMOKE', None)
    # Fail before starting listeners if shared receiver/tracker dependencies are missing.
    subprocess.run([sys.executable, '-B', '-c',
        'import av,grpc,starlette,uvicorn; from cloud_core.tracking import CloudByteTrack; '
        'CloudByteTrack().prepare(); print("RECEIVER_DEPENDENCIES_PASS",flush=True)'],
        cwd=ROOT, env=env, check=True)
    media = None
    media_log = None
    try:
        if not reachable('127.0.0.1', 8554):
            binary = Path('/home/stu0/edge_video_platform/tools/mediamtx/mediamtx')
            if not binary.is_file():
                raise FileNotFoundError(f'MediaMTX missing: {binary}; start existing MediaMTX before retry')
            if reachable('127.0.0.1', 8889):
                raise RuntimeError('WebRTC port8889 occupied while RTSP8554 absent; inspect existing MediaMTX')
            media_log = (ROOT / 'live_mediamtx.log').open('x', encoding='utf-8')
            media_env = dict(env, MTX_WEBRTCADDITIONALHOSTS=CLOUD_HOST)
            media = subprocess.Popen([str(binary), str(ROOT / 'media/mediamtx.yml')],
                                     cwd=ROOT, env=media_env, stdout=media_log, stderr=subprocess.STDOUT)
            for _ in range(40):
                if media.poll() is not None:
                    raise RuntimeError('MediaMTX exited; inspect live_mediamtx.log')
                if reachable('127.0.0.1', 8554) and reachable('127.0.0.1', 8889):
                    break
                time.sleep(.25)
        if not reachable('127.0.0.1', 8554) or not reachable('127.0.0.1', 8889):
            raise RuntimeError('MediaMTX RTSP8554/WebRTC8889 not ready')
        print(f'LIVE_URL=http://{CLOUD_HOST}:{HTTP_PORT}/v1/streams/edge-246/traffic-01/yolo11s-live/live?fps=30&buffer_ms=500', flush=True)
        # Foreground process: Ctrl+C stops this receiver and only MediaMTX owned here.
        return subprocess.call([sys.executable, '-B', '-m', 'uvicorn', 'web.cloud_app:create_app',
                                '--factory', '--host', '0.0.0.0', '--port', str(HTTP_PORT)],
                               cwd=ROOT, env=env)
    finally:
        if media is not None and media.poll() is None:
            media.terminate()
            try:
                media.wait(timeout=5)
            except subprocess.TimeoutExpired:
                media.kill()
                media.wait()
        if media_log is not None:
            media_log.close()


def edge(frames):
    smoke = json.loads((ROOT / 'adapter_smoke/report.json').read_text(encoding='utf-8'))
    if smoke.get('completed') != 60 or smoke.get('error') or smoke.get('status') != 'PERSISTENT_ADAPTER_CONTRACT_PASS_QUALITY_NOT_ADMITTED':
        raise RuntimeError('integration adapter60 gate not PASS')
    if hashlib.sha256(DLC.read_bytes()).hexdigest() != DLC_SHA:
        raise ValueError('pinned model checksum mismatch')
    if not VIDEO.is_file() or not shutil.which('ffmpeg'):
        raise RuntimeError('video or ffmpeg missing')
    with urlopen(f'http://{CLOUD_HOST}:{HTTP_PORT}/healthz', timeout=5) as response:
        health = json.load(response)
    if (health.get('status') != 'ok' or health.get('transport') != 'grpc'
            or health.get('run_id') != 'yolo11s-cache-off-v3'
            or health.get('cloud_model_modes') != [] or 'yolo11s_edge' not in health.get('wire_modes', [])):
        raise RuntimeError('wrong Cloud receiver/source/config; start v3 Cloud launcher first')
    for port in (GRPC_PORT, 8554):
        if not reachable(CLOUD_HOST, port):
            raise RuntimeError(f'Cloud port {port} unreachable')
    root = Path(tempfile.mkdtemp(prefix='live_yolo11s.', dir=ROOT))
    env = child_env()
    env.update(QAIRT_ROOT='/home/ubuntu/sigm/qairt/2.43.0.260128',
               SNPE_PERSISTENT='1', SNPE_PERSISTENT_MODE='daemon', SNPE_DAEMON='1',
               YOLO11S_SNPE_DAEMON_BIN=str(ROOT / 'build/snpe-daemon-yolo11s-v3'))
    argv = [sys.executable, '-B', '-u', '-m', 'edge_agent.phase7', '--mode', 'yolo11s_edge',
            '--video', str(VIDEO), '--frames', str(frames), '--transport', 'grpc',
            '--grpc-target', f'{CLOUD_HOST}:{GRPC_PORT}', '--rtsp-base', f'rtsp://{CLOUD_HOST}:8554',
            '--node-id', 'edge-246', '--camera-id', 'traffic-01', '--stream-id', 'yolo11s-live',
            '--stream-epoch', str(time.time_ns()), '--decision-id', 'yolo11s-v3-live',
            '--yolo11s-dlc', str(DLC), '--yolo11s-allow-diagnostic', '--yolo11s-confidence', '.1',
            '--yolo11s-nms-iou', '.5', '--width', '1280', '--height', '720', '--preview-fps', '30',
            '--bitrate-kbps', '2500', '--shutdown-padding-frames', '30', '--monitor',
            '--metrics-port', '0', '--monitor-jsonl', str(root / 'monitor.jsonl')]
    report = {'status': 'STARTED', 'diagnostic_only': True, 'production_ready': False,
              'quality_status': 'NOT_EVALUATED', 'parity_gate': 'FAIL', 'init_cache': False,
              'model_sha256': DLC_SHA, 'receiver': health, 'argv': argv, 'visual_gate': 'REVIEW_REQUIRED'}
    print(f'LIVE_DIR={root}', flush=True)
    print(f'LIVE_URL=http://{CLOUD_HOST}:{HTTP_PORT}/v1/streams/edge-246/traffic-01/yolo11s-live/live?fps=30&buffer_ms=500', flush=True)
    try:
        with (root / 'edge.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True, bufsize=1)
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                code = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        report['exit'] = code
        text = (root / 'edge.log').read_text(encoding='utf-8')
        marker = text.rfind('\n{\n')
        summary = json.JSONDecoder().raw_decode(text[marker + 1:])[0] if marker >= 0 else {}
        report['pipeline'] = summary
        stats = summary.get('inference', {})
        passed = (code == 0 and summary.get('captured') == frames
                  and summary.get('model_mode') == 'yolo11s_edge'
                  and stats.get('frames_inferred', 0) > 0
                  and all(stats.get(k) == 0 for k in ('model_errors', 'transport_errors', 'cloud_rejections'))
                  and summary.get('media_ref_send_failed') == 0)
        report['status'] = 'LIVE_PIPELINE_PASS_VISUAL_REVIEW_REQUIRED' if passed else 'LIVE_PIPELINE_REVIEW_REQUIRED'
        return 0 if passed else 1
    finally:
        path = root / 'report.json'
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'status': report['status'], 'report': str(path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('node', choices=('edge', 'cloud'))
    parser.add_argument('--frames', type=int, default=1800)
    args = parser.parse_args()
    if sys.platform != 'linux' or not 300 <= args.frames <= 1800:
        parser.error('Linux only; frames must be300..1800')
    subprocess.run([sys.executable, '-B', ROOT / 'scripts/verify_release_snapshot.py'], cwd=ROOT, check=True)
    return cloud() if args.node == 'cloud' else edge(args.frames)


if __name__ == '__main__':
    raise SystemExit(main())
