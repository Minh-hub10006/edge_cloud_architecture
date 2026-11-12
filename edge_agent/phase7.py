"""Run one real Edge model and publish its independent H.264 preview."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from time import monotonic_ns, time_ns

from contracts import ModelMode, Placement, TransportKind
from contracts.reference_sink import MediaReferenceSink
from edge_agent import (
    BlurDifficultyNormalizer, EdgeFanout, StateObservation,
    StatePrometheusExporter, WindowStateAggregator, blur_difficulty,
)
from inference import phase7_edge_catalog
from inference.edge_worker import run_edge_inference_worker
from inference.frame_refs import MediaReferenceMap
from media.ffmpeg_publisher import FFmpegH264Publisher
from media.opencv_source import OpenCVFrameSource
from media.profile import MediaProfile, rtsp_publish_url
from media.pipeline import MediaCadence


def _tracked_boxes(result):
    if result is None:
        return None
    tracked = {
        int(box.track_id): (box.left, box.top, box.right, box.bottom)
        for box in result.boxes if box.track_id is not None
    }
    return tracked or None


def _box_change(previous, current):
    if previous is None or current is None:
        return None
    if not previous and not current:
        return 0.0
    old_ids, new_ids = set(previous), set(current)
    changed = len(old_ids ^ new_ids) / max(1, len(old_ids | new_ids))
    shared = old_ids & new_ids
    if shared:
        ious = []
        for track_id in shared:
            ax1, ay1, ax2, ay2 = previous[track_id]
            bx1, by1, bx2, by2 = current[track_id]
            iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = iw * ih
            area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            union = area_a + area_b - inter
            ious.append(inter / union if union > 0 else 0.0)
        geometry = 1.0 - sum(ious) / len(ious)
    else:
        geometry = 1.0
    return min(1.0, max(0.0, 0.5 * changed + 0.5 * geometry))


async def _run(args: argparse.Namespace) -> None:
    mode = ModelMode(args.mode)
    if getattr(args, "monitor_window_size", 30) <= 0:
        raise ValueError("--monitor-window-size must be positive")
    if getattr(args, "monitor_max_frames", 4096) <= 0 or getattr(args, "monitor_max_windows", 32) <= 0:
        raise ValueError("monitor retention limits must be positive")
    if not 0 <= getattr(args, "metrics_port", 9108) <= 65535:
        raise ValueError("--metrics-port must be between 0 and 65535")
    blur_blurry_variance = getattr(args, "blur_blurry_variance", None)
    blur_sharp_variance = getattr(args, "blur_sharp_variance", None)
    if blur_blurry_variance is not None or blur_sharp_variance is not None:
        if blur_blurry_variance is None or blur_sharp_variance is None:
            raise ValueError("set both --blur-blurry-variance and --blur-sharp-variance")
        blur_normalizer = BlurDifficultyNormalizer(
            blur_blurry_variance, blur_sharp_variance,
        )
    else:
        blur_normalizer = None
    key = (args.node_id, args.camera_id, args.stream_id, args.stream_epoch)
    profile = MediaProfile(
        width=args.width, height=args.height, fps=args.preview_fps,
        bitrate_kbps=args.bitrate_kbps, gop_frames=args.preview_fps,
    )
    source = OpenCVFrameSource(
        args.video, node_id=args.node_id, camera_id=args.camera_id,
        stream_id=args.stream_id, stream_epoch=args.stream_epoch,
        max_frames=args.frames,
    )
    fanout = EdgeFanout(media_capacity=args.media_queue, analytics_capacity=args.analytics_queue)
    references = MediaReferenceMap(capacity=args.reference_capacity)
    catalog = phase7_edge_catalog(
        yolo_dlc=args.yolo_dlc or None, yolo_imgsz=args.yolo_imgsz,
        yolo_layout=args.yolo_layout, confidence=args.confidence,
        rtmdet_dlc=getattr(args, "rtmdet_dlc", "") or None,
        yolo11s_dlc=getattr(args, "yolo11s_dlc", "") or None,
        yolo11s_allow_diagnostic=getattr(args, "yolo11s_allow_diagnostic", False),
        yolo11s_confidence=getattr(args, "yolo11s_confidence", .1),
        yolo11s_nms_iou=getattr(args, "yolo11s_nms_iou", .5),
    )
    detector = None
    if mode.placement is Placement.EDGE:
        detector = await asyncio.to_thread(catalog.load, mode)

    transport = TransportKind.configured(args.transport)
    if transport is TransportKind.GRPC:
        from contracts.grpc_client import GrpcCloudClient
        client = GrpcCloudClient(args.grpc_target, timeout_s=args.rpc_timeout)
        try:
            await asyncio.wait_for(client.channel.channel_ready(), timeout=args.rpc_timeout)
        except Exception:
            await client.close()
            raise
    else:
        from contracts.kafka_transport import KafkaCloudClient
        client = KafkaCloudClient(
            args.kafka_bootstrap, node_id=args.node_id, timeout_s=args.rpc_timeout,
            max_pending=args.kafka_max_pending,
        )
        await client.start()
    reference_sink = MediaReferenceSink(client, capacity=args.reference_capacity)
    reference_sink.start()

    monitor = (WindowStateAggregator(
        max_windows=args.monitor_max_windows,
        max_observations_per_window=args.monitor_max_frames,
    ) if getattr(args, "monitor", False) else None)
    metrics_exporter = None
    metrics_port = getattr(args, "metrics_port", 9108)
    if monitor is not None and metrics_port:
        try:
            metrics_exporter = StatePrometheusExporter(
                node=args.node_id, camera=args.camera_id, stream=args.stream_id,
                epoch=args.stream_epoch,
                model=(detector.model_id if detector is not None else args.cloud_model_id),
                transport=transport.value,
            )
            metrics_exporter.start(metrics_port)
            print(f"[Edge monitor] Prometheus scrape endpoint :{metrics_port}/", flush=True)
        except Exception as exc:
            print(f"[Edge monitor] Prometheus disabled: {type(exc).__name__}: {exc}", flush=True)
            metrics_exporter = None
    monitor_log = (open(args.monitor_jsonl, "a", encoding="utf-8")
                   if monitor is not None and getattr(args, "monitor_jsonl", "") else None)
    active_monitor_window = {"id": None}
    previous_tracks = {"value": None}

    def finish_monitor_window(window_id: int) -> None:
        if monitor is None:
            return
        try:
            state = monitor.finish(key, window_id)
        except ValueError:
            return
        record = json.dumps(state.as_dict(), separators=(",", ":"), allow_nan=False)
        print(f"[DRL monitor] {record}", flush=True)
        if monitor_log is not None:
            monitor_log.write(record + "\n")
            monitor_log.flush()
        if metrics_exporter is not None:
            metrics_exporter.publish(state, monitor)

    def observe_frame(frame, measured: dict) -> None:
        if monitor is None:
            return
        frame_id = frame.identity.source_frame_id
        window_id = frame_id // args.monitor_window_size
        previous_id = active_monitor_window["id"]
        if previous_id is not None and window_id > previous_id:
            finish_monitor_window(previous_id)
            active_monitor_window["id"] = window_id
        elif previous_id is None:
            active_monitor_window["id"] = window_id

        result = measured.get("result")
        tracks = _tracked_boxes(result)
        change = _box_change(previous_tracks["value"], tracks)
        # Compare only adjacent completed results with at least one tracked box.
        # Untracked detections are excluded; a missing tracked frame breaks the
        # comparison instead of stretching IoU over an unknown time gap.
        previous_tracks["value"] = tracks
        confidence_values = tuple(
            float(box.calibrated_confidence) for box in (result.boxes if result else ())
            if box.calibrated_confidence is not None
        )
        blur = blur_difficulty(frame.pixels, blur_normalizer) if blur_normalizer else None
        observation = StateObservation(
            stream_key=key, window_id=window_id, frame_id=frame_id,
            monotonic_ns=monotonic_ns(),
            media_matched=measured.get("media_matched"),
            transport_rtt_ms=measured.get("transport_rtt_ms"),
            edge_backlog_norm=measured.get("edge_backlog_norm"),
            cloud_inflight_norm=measured.get("cloud_inflight_norm"),
            edge_inference_ms=measured.get("edge_inference_ms"),
            cloud_e2e_ms=measured.get("cloud_e2e_ms"),
            blur_difficulty=blur, calibrated_confidences=confidence_values,
            box_change=change, completed=bool(measured.get("completed")),
            model_id=(result.model_id if result is not None else
                      detector.model_id if detector is not None else args.cloud_model_id),
            transport=transport.value,
        )
        monitor.record(observation)

    async def on_media_reference(reference):
        await references.publish(reference)
        await reference_sink.submit(reference)

    publisher = FFmpegH264Publisher(
        profile=profile, stream_key=key,
        output=rtsp_publish_url(args.rtsp_base, key),
        on_frame_submitted=on_media_reference,
        frame_identity=True,
    )

    stop = asyncio.Event()
    inference_stop = asyncio.Event()
    captured = {"count": 0, "media_skipped": 0}
    shutdown_padding_frames = max(0, getattr(args, "shutdown_padding_frames", profile.fps))
    shutdown_drain_s = max(0.0, getattr(args, "shutdown_drain_s", 3.0))
    tasks: list[asyncio.Task] = []

    async def capture():
        async for frame in source:
            # Both modes infer preview frames only, preserving display identity.
            fanout.publish(frame, analytics=False)
            captured["count"] += 1

    async def publish_media():
        cadence = MediaCadence(profile.fps)
        last_frame = None
        while not stop.is_set() or fanout.media.pending:
            try:
                frame = await asyncio.wait_for(fanout.media.get(), timeout=0.05)
            except asyncio.TimeoutError:
                continue
            if cadence.accept(frame.cadence_time_ns):
                await publisher.publish(frame)
                last_frame = frame
                fanout.analytics.put_latest(frame)
            else:
                captured["media_skipped"] += 1
        if last_frame is not None:
            for _ in range(shutdown_padding_frames):
                await publisher.publish_padding(last_frame)
                await asyncio.sleep(1 / profile.fps)
        inference_stop.set()
        try:
            await asyncio.wait_for(asyncio.shield(inference_task), timeout=shutdown_drain_s)
        except asyncio.TimeoutError:
            pass
        # A bounded drain avoids keeping an idle RTSP publisher open forever
        # if a Cloud reply never arrives.
        await publisher.close()
        try:
            await asyncio.wait_for(asyncio.shield(inference_task), timeout=max(.1, shutdown_drain_s))
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Inference did not finish within bounded shutdown drain") from exc

    try:
        async with publisher:
            media_task = asyncio.create_task(publish_media())
            inference_task = asyncio.create_task(run_edge_inference_worker(
                fanout.analytics, mode=mode, client=client, references=references,
                detector=detector, decision_id=args.decision_id,
                cloud_model_id=args.cloud_model_id,
                reference_wait_s=args.reference_wait, ticket_ttl_s=args.ticket_ttl,
                stop=inference_stop,
                latest_only=True, max_frame_age_s=getattr(args, "max_frame_age", 0.5),
                preview_fps=profile.fps,
                on_observation=observe_frame if monitor is not None else None,
            ))
            capture_task = asyncio.create_task(capture())
            tasks.extend((capture_task, media_task, inference_task))
            done, _ = await asyncio.wait(
                (capture_task, media_task, inference_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task.exception() is not None:
                    raise task.exception()
            await capture_task
            stop.set()
            _, inference_stats = await asyncio.gather(media_task, inference_task)
            if monitor is not None and active_monitor_window["id"] is not None:
                finish_monitor_window(active_monitor_window["id"])
        # Drain compact reference messages before shutting down the channel.
        await reference_sink.close()
        print(json.dumps({
            "node_id": args.node_id,
            "model_mode": mode.value,
            "transport": transport.value,
            "model_id": detector.model_id if detector is not None else args.cloud_model_id,
            "captured": captured["count"],
            "media_encoded": publisher.frames_submitted,
            "media_padding_encoded": publisher.padding_frames_submitted,
            "media_queue_dropped": fanout.media.dropped,
            "analytics_queue_dropped": fanout.analytics.dropped,
            "media_fps_skipped": captured["media_skipped"],
            "media_ref_queue_dropped": reference_sink.dropped,
            "media_ref_send_failed": reference_sink.failed,
            "inference": inference_stats,
            "monitor": (None if monitor is None else {
                "windows_retained": len(monitor.recent()),
                "duplicate_observations": monitor.duplicate_observations,
                "capacity_drops": monitor.capacity_drops,
                "metrics_port": metrics_port or None,
                "jsonl": getattr(args, "monitor_jsonl", "") or None,
            }),
            "rtsp_url": publisher.output,
        }, indent=2))
    finally:
        stop.set()
        inference_stop.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await reference_sink.close()
        await client.close()
        if metrics_exporter is not None:
            metrics_exporter.close()
        if monitor_log is not None:
            monitor_log.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 real Edge-to-Cloud inference smoke")
    parser.add_argument("--video", required=True)
    parser.add_argument("--mode", choices=[mode.value for mode in ModelMode], default=ModelMode.YOLO_EDGE.value)
    parser.add_argument("--node-id", default="edge-246")
    parser.add_argument("--camera-id", default="traffic-01")
    parser.add_argument("--stream-id", default="preview-hq")
    parser.add_argument("--stream-epoch", type=int, default=1)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--decision-id", default="phase7-fixed")
    parser.add_argument("--grpc-target", default="192.168.180.249:50051")
    parser.add_argument("--transport", choices=[kind.value for kind in TransportKind], default=None,
                        help="data-plane transport; defaults to TRANSPORT_MODE or grpc")
    parser.add_argument("--kafka-bootstrap", default=os.getenv("KAFKA_BOOTSTRAP_SERVERS", ""))
    parser.add_argument("--kafka-max-pending", type=int, default=64)
    parser.add_argument("--rtsp-base", default="rtsp://192.168.180.249:8554")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview-fps", type=int, default=15)
    parser.add_argument("--bitrate-kbps", type=int, default=2500)
    parser.add_argument("--media-queue", type=int, default=8)
    parser.add_argument("--analytics-queue", type=int, default=8)
    parser.add_argument("--reference-capacity", type=int, default=2048)
    parser.add_argument("--reference-wait", type=float, default=0.25)
    parser.add_argument("--ticket-ttl", type=float, default=1.5)
    parser.add_argument("--max-frame-age", type=float, default=0.5)
    parser.add_argument("--rpc-timeout", type=float, default=6.0)
    parser.add_argument("--yolo-dlc", default="")
    parser.add_argument("--rtmdet-dlc", default="", help="RTMDet-Small DLC with checked .dlc.json sidecar")
    parser.add_argument("--yolo11s-dlc", default="")
    parser.add_argument("--yolo11s-allow-diagnostic", action="store_true", help="explicit opt-in to parity FAIL candidate")
    parser.add_argument("--yolo11s-confidence", type=float, default=.1)
    parser.add_argument("--yolo11s-nms-iou", type=float, default=.5)
    parser.add_argument("--yolo-imgsz", type=int, default=320)
    parser.add_argument("--yolo-layout", choices=["nchw", "nhwc"], default="nchw")
    parser.add_argument("--cloud-model-id", default="dfine_full")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--shutdown-padding-frames", type=int, default=15)
    parser.add_argument("--shutdown-drain-s", type=float, default=3.0)
    parser.add_argument("--monitor", action="store_true",
                        help="aggregate window state and emit debug JSON; does not enable a policy")
    parser.add_argument("--monitor-window-size", type=int, default=30,
                        help="source frames per monitor window")
    parser.add_argument("--monitor-jsonl", default="",
                        help="optional append-only path for per-window state JSONL")
    parser.add_argument("--metrics-port", type=int, default=9108,
                        help="Prometheus endpoint port when --monitor is enabled; 0 disables it")
    parser.add_argument("--blur-blurry-variance", type=float, default=None,
                        help="camera/resolution-calibrated Laplacian variance for blurry threshold")
    parser.add_argument("--blur-sharp-variance", type=float, default=None,
                        help="camera/resolution-calibrated Laplacian variance for sharp threshold")
    parser.add_argument("--monitor-max-frames", type=int, default=4096)
    parser.add_argument("--monitor-max-windows", type=int, default=32)
    # Build a bounded aggregator with explicit configured limits at runtime.
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
