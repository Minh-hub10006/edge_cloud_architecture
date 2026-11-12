"""Explicit benchmark CLI; no implicit deployment or production mutation."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from .config import RunConfig, validate_campaign


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in model/pipeline benchmark; no deployment or cleanup")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a future multi-node campaign config")
    validate.add_argument("config", type=Path)
    coordinate = commands.add_parser("coordinate", help="READY/START barrier for manually launched LAN workers")
    coordinate.add_argument("--nodes", required=True, help="comma-separated distinct Edge IDs")
    coordinate.add_argument("--model-key", required=True)
    coordinate.add_argument("--placement", required=True, choices=["edge", "cloud"])
    coordinate.add_argument("--profile", choices=["pipeline_capacity", "realtime"], default="pipeline_capacity")
    coordinate.add_argument("--transport", required=True, choices=["grpc", "kafka"])
    coordinate.add_argument("--campaign-id", default=None)
    coordinate.add_argument("--frames", type=int, default=900)
    coordinate.add_argument("--warmup-frames", type=int, default=30)
    coordinate.add_argument("--repetitions", type=int, default=3)
    coordinate.add_argument("--listen", default="127.0.0.1")
    coordinate.add_argument("--port", type=int, default=9200)
    coordinate.add_argument("--ready-timeout-s", type=float, default=180)
    coordinate.add_argument("--measure-timeout-s", type=float, default=600)
    coordinate.add_argument("--out", type=Path, default=Path("results/benchmark"))
    model = commands.add_parser("model", help="unpaced model adapter capacity at its native placement")
    model.add_argument("--adapter", required=True, choices=["yolo_snpe", "yolo11s_snpe", "dfine_onnx", "rtmdet_snpe", "rtdetrv2_onnx", "lwdetr_onnx"])
    model.add_argument("--artifact", required=True, type=Path)
    model.add_argument("--video", required=True, type=Path)
    model.add_argument("--node-id", required=True)
    model.add_argument("--model-key", required=True)
    model.add_argument("--campaign-id", default=None)
    model.add_argument("--frames", type=int, default=900)
    model.add_argument("--warmup-frames", type=int, default=30)
    model.add_argument("--repetitions", type=int, default=3)
    model.add_argument("--cache-frames", type=int, default=16)
    model.add_argument("--cache-mib", type=int, default=256)
    model.add_argument("--resource-sample-s", type=float, default=1)
    model.add_argument("--imgsz", type=int, default=320)
    model.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
    model.add_argument("--confidence", type=float, default=0.25)
    model.add_argument("--out", type=Path, default=Path("results/benchmark"))
    model.add_argument("--run-timeout-s", type=float, default=600,
                       help="finite timeout for all repetitions; kills only owned worker, not shared daemon")
    model.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    pipeline = commands.add_parser("pipeline", help="data-plane throughput completed at Cloud ByteTrack")
    pipeline.add_argument("--mode", required=True, choices=["yolo_edge", "yolo11s_edge", "rtmdet_edge", "dfine_cloud", "rtdetr_cloud", "rtdetrv2_small_cloud", "lwdetr_large_cloud"])
    pipeline.add_argument("--profile", choices=["pipeline_capacity", "realtime"], default="pipeline_capacity")
    pipeline.add_argument("--transport", required=True, choices=["grpc", "kafka"])
    pipeline.add_argument("--model-key", required=True)
    pipeline.add_argument("--artifact", type=Path, help="Edge DLC only; Cloud artifact stays on Cloud")
    pipeline.add_argument("--video", required=True, type=Path)
    pipeline.add_argument("--node-id", required=True)
    pipeline.add_argument("--camera-id", default="traffic-01")
    pipeline.add_argument("--grpc-target", default="192.168.180.249:50061")
    pipeline.add_argument("--kafka-bootstrap", default="192.168.180.95:9092")
    pipeline.add_argument("--kafka-namespace", default="benchmark")
    pipeline.add_argument("--rtsp-base", default="rtsp://192.168.180.249:8554")
    pipeline.add_argument("--media-mode", choices=["rtsp", "none"], default="rtsp",
                          help="none removes YOLO preview media; D-FINE requires rtsp input")
    pipeline.add_argument("--width", type=int, default=1280)
    pipeline.add_argument("--height", type=int, default=720)
    pipeline.add_argument("--fps", "--media-fps", dest="fps", type=int, default=30,
                          help="RTSP timestamp FPS (or realtime offered rate); capacity has no FPS cap")
    pipeline.add_argument("--bitrate-kbps", type=int, default=2400)
    pipeline.add_argument("--gop", type=int, default=30)
    pipeline.add_argument("--queue-depth", type=int, default=16)
    pipeline.add_argument("--max-inflight", type=int, default=4,
                          help="bounded outstanding tracked requests per stream (1 restores serial requests)")
    pipeline.add_argument("--no-split-preprocess", action="store_true",
                          help="run YOLO preprocessing inside its inference lane for diagnosis")
    pipeline.add_argument("--loop-source", action="store_true",
                          help="rewind the clip at EOF so measured frames may exceed its length (content repeats)")
    pipeline.add_argument("--source-prefetch", type=int, default=0,
                          help="decode this many frames ahead in a reader thread (0 = baseline serial read)")
    pipeline.add_argument("--priming-frames", type=int, default=30)
    pipeline.add_argument("--padding-frames", type=int, default=15)
    pipeline.add_argument("--rpc-timeout-s", type=float, default=10)
    pipeline.add_argument("--frame-ttl-s", type=float, default=10)
    pipeline.add_argument("--phase-timeout-s", type=float, default=600)
    pipeline.add_argument("--coordinator", help="trusted LAN host:port; local paths never received remotely")
    pipeline.add_argument("--coordinator-timeout-s", type=float, default=180)
    for command in (pipeline,):
        command.add_argument("--campaign-id", default=None)
        command.add_argument("--frames", type=int, default=900)
        command.add_argument("--warmup-frames", type=int, default=30)
        command.add_argument("--repetitions", type=int, default=3)
        command.add_argument("--resource-sample-s", type=float, default=1)
        command.add_argument("--imgsz", type=int, default=320)
        command.add_argument("--layout", choices=["nchw", "nhwc"], default="nchw")
        command.add_argument("--confidence", type=float, default=0.25)
        command.add_argument("--out", type=Path, default=Path("results/benchmark"))
        command.add_argument("--run-timeout-s", type=float, default=600)
        command.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    for candidate_command in (model, pipeline):
        candidate_command.add_argument("--allow-diagnostic", action="store_true", help="explicit parity FAIL candidate opt-in")
        candidate_command.add_argument("--nms-iou", type=float, default=.5)
    args = parser.parse_args(argv)
    if args.command == "validate":
        validate_campaign(json.loads(args.config.read_text(encoding="utf-8")))
        print("Campaign schema OK; launch matching coordinate/pipeline workers manually")
        return 0
    if args.command == "coordinate":
        from .remote import coordinate as run_coordinator
        cfg = RunConfig(args.campaign_id or uuid4().hex, "coordinator", "coordinator", args.model_key,
                        args.placement, args.frames, args.warmup_frames, args.profile, args.transport)
        report = asyncio.run(run_coordinator(cfg, args.nodes.split(","), args.out / ("campaign-" + cfg.campaign_id),
            host=args.listen, port=args.port, repetitions=args.repetitions,
            ready_timeout_s=args.ready_timeout_s, measure_timeout_s=args.measure_timeout_s))
        return 0 if report["status"] == "PASS" else 1
    if not 1 <= args.repetitions <= 100 or not 1 <= getattr(args, "cache_frames", 1) <= 1024:
        parser.error("repetitions must be 1..100 and cache frames 1..1024")
    if not 1 <= getattr(args, "cache_mib", 1) <= 4096 or args.imgsz <= 0 or not 0 <= args.confidence <= 1:
        parser.error("invalid cache/input/confidence")
    if args.resource_sample_s != 0 and not 0.1 <= args.resource_sample_s < float("inf"):
        parser.error("resource sample must be 0 (disabled) or finite >=0.1")
    if not 0 < args.run_timeout_s < float("inf"):
        parser.error("run timeout must be finite and positive")
    if args.command == "pipeline" and args.mode == "rtdetrv2_small_cloud":
        args.mode = "rtdetr_cloud"  # keep wire enum 4 and existing route compatibility
    placement = "edge" if (args.adapter in {"yolo_snpe", "yolo11s_snpe", "rtmdet_snpe"} if args.command == "model" else args.mode in {"yolo_edge", "yolo11s_edge", "rtmdet_edge"}) else "cloud"
    campaign_id = args.campaign_id or uuid4().hex
    # Validate before importing model runtimes or touching artifacts.
    RunConfig(campaign_id, "validate", args.node_id, args.model_key, placement,
              args.frames, args.warmup_frames, profile=getattr(args, "profile", "model_capacity"),
              transport=getattr(args, "transport", "none"))
    if args.command == "pipeline":
        from .pipeline import PipelineOptions
        from media.profile import MediaProfile, rtsp_publish_url
        from contracts.kafka_transport import namespaced_topics
        PipelineOptions(args.fps, args.queue_depth, args.priming_frames, args.padding_frames,
                        args.rpc_timeout_s, args.frame_ttl_s, args.phase_timeout_s, args.resource_sample_s,
                        args.max_inflight, not args.no_split_preprocess)
        if args.media_mode == "rtsp":
            MediaProfile(args.width, args.height, args.fps, args.bitrate_kbps, args.gop)
            rtsp_publish_url(args.rtsp_base, (args.node_id, args.camera_id, "validate", 0))
        if args.transport == "kafka" and not args.kafka_namespace:
            parser.error("benchmark requires a nonempty Kafka namespace to isolate live traffic")
        namespaced_topics(args.kafka_namespace)
        if placement == "edge" and args.artifact is None:
            parser.error("Edge inference requires --artifact DLC")
        if placement == "cloud" and args.media_mode == "none":
            parser.error("Cloud inference requires --media-mode=rtsp for input pixels")
        if not 0 < args.coordinator_timeout_s < float("inf"):
            parser.error("coordinator timeout must be finite and positive")
    if not args._worker:
        # Isolate native inference from the CLI so a hung model can be bounded.
        directory = args.out / ("supervised-" + uuid4().hex)
        directory.mkdir(parents=True, exist_ok=False)
        forwarded = list(sys.argv[1:] if argv is None else argv)
        command = [sys.executable, "-m", "benchmark", *forwarded, "--_worker",
                   "--campaign-id", campaign_id, "--out", str(directory)]
        print(json.dumps({"worker_output_root": str(directory.resolve()),
                          "timeout_s": args.run_timeout_s}), flush=True)
        try:
            code = subprocess.run(command, timeout=args.run_timeout_s, check=False).returncode
            state = "FINISHED" if code == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            code, state = 1, "TIMEOUT_INCOMPLETE"
        except KeyboardInterrupt:
            code, state = 130, "INTERRUPTED_INCOMPLETE"
        except OSError as exc:
            code, state = 1, f"WORKER_START_FAILED: {exc}"
        with (directory / "supervisor.json").open("x", encoding="utf-8") as target:
            json.dump({"campaign_id": campaign_id, "node_id": args.node_id,
                       "state": state, "exit_code": code, "timeout_s": args.run_timeout_s}, target, indent=2)
        print(json.dumps({"supervisor_state": state, "exit_code": code}), flush=True)
        return code
    if args.command == "pipeline":
        return asyncio.run(_pipeline_runs(args, campaign_id, placement))
    from .model import load_adapter, measure_model, preload_video, sha256
    artifact = args.artifact.expanduser().resolve(strict=True)
    video = args.video.expanduser().resolve(strict=True)
    identity = {"artifact": str(artifact), "artifact_sha256": sha256(artifact),
                "video": str(video), "video_sha256": sha256(video),
                "adapter": args.adapter, "model_key": args.model_key}
    frames = preload_video(video, args.cache_frames, args.cache_mib * 1024 * 1024)
    identity["source_shape"] = list(frames[0].shape)
    diagnostic_options = ({"allow_diagnostic": args.allow_diagnostic, "nms_iou": args.nms_iou}
                          if args.adapter == "yolo11s_snpe" else {})
    detector, runtime = load_adapter(args.adapter, artifact, imgsz=args.imgsz,
                                     layout=args.layout, confidence=args.confidence, **diagnostic_options)
    if args.adapter == "yolo11s_snpe" and args.model_key != detector.model_id:
        raise ValueError("YOLO11s model-key does not match pinned candidate identity")
    identity.update(model_id=detector.model_id, runtime=runtime)
    failed = False
    for repetition in range(1, args.repetitions + 1):
        run_id = uuid4().hex
        config = RunConfig(campaign_id, run_id, args.node_id, args.model_key, placement,
                           args.frames, args.warmup_frames, repetition=repetition)
        directory = args.out / run_id
        summary = measure_model(config, detector, frames, directory,
                                identity=identity, sample_s=args.resource_sample_s)
        print(json.dumps({"output": str(directory.resolve()), "summary": summary}, ensure_ascii=False))
        failed |= summary["execution_status"] != "PASS"
        if summary["fatal_error"]:
            break
    return 1 if failed else 0


async def _pipeline_runs(args, campaign_id, placement):
    from dataclasses import asdict
    from time import time
    from contracts import ModelMode
    from media.ffmpeg_publisher import FFmpegH264Publisher
    from media.opencv_source import OpenCVFrameSource
    from media.profile import MediaProfile, rtsp_publish_url
    from .client import connect
    from .model import load_adapter, sha256
    from .pipeline import PipelineOptions, measure_pipeline
    video = args.video.expanduser().resolve(strict=True)
    identity = {"video": str(video), "video_sha256": sha256(video), "model_key": args.model_key,
                "source_loop": args.loop_source, "source_prefetch": args.source_prefetch}
    detector = None
    if placement == "edge":
        artifact = args.artifact.expanduser().resolve(strict=True)
        adapter = {"rtmdet_edge": "rtmdet_snpe", "yolo11s_edge": "yolo11s_snpe"}.get(args.mode, "yolo_snpe")
        diagnostic_options = ({"allow_diagnostic": args.allow_diagnostic, "nms_iou": args.nms_iou}
                              if args.mode == "yolo11s_edge" else {})
        detector, runtime = await asyncio.to_thread(load_adapter, adapter, artifact,
            imgsz=args.imgsz, layout=args.layout, confidence=args.confidence, **diagnostic_options)
        if args.mode == "yolo11s_edge" and args.model_key != detector.model_id:
            raise ValueError("YOLO11s model-key does not match pinned candidate identity")
        identity.update(artifact=str(artifact), artifact_sha256=sha256(artifact), runtime=runtime)
    profile = (MediaProfile(args.width, args.height, args.fps, args.bitrate_kbps, args.gop)
               if args.media_mode == "rtsp" else None)
    options = PipelineOptions(args.fps, args.queue_depth, args.priming_frames, args.padding_frames,
                              args.rpc_timeout_s, args.frame_ttl_s, args.phase_timeout_s, args.resource_sample_s,
                              args.max_inflight, not args.no_split_preprocess)
    target = args.grpc_target if args.transport == "grpc" else args.kafka_bootstrap
    failed = False
    coordinator = None
    if args.coordinator:
        from .remote import WorkerConnection
        coordinator = await WorkerConnection.open(args.coordinator, args.node_id, args.coordinator_timeout_s)
    try:
        for repetition in range(1, args.repetitions + 1):
            run_id = uuid4().hex
            config = RunConfig(campaign_id, run_id, args.node_id, args.model_key, placement,
                args.frames, args.warmup_frames, args.profile, args.transport, repetition)
            if coordinator:
                config = await coordinator.config(config)
                run_id = config.run_id
            key = (args.node_id, args.camera_id, "bench-" + run_id, int(time()))
            source = OpenCVFrameSource(str(video), node_id=key[0], camera_id=key[1],
                stream_id=key[2], stream_epoch=key[3], realtime_file=False, loop=args.loop_source, prefetch=args.source_prefetch,
                max_frames=(args.priming_frames if args.media_mode == "rtsp" else 0)
                           + args.warmup_frames + args.frames)
            publisher = (FFmpegH264Publisher(profile=profile, output=rtsp_publish_url(args.rtsp_base, key),
                                            stream_key=key, frame_identity=True)
                         if args.media_mode == "rtsp" else None)
            client = await connect(args.transport, target, args.node_id, args.rpc_timeout_s, args.kafka_namespace)
            try:
                summary = await measure_pipeline(config, source, publisher, client, args.out / run_id,
                    options=options, mode=ModelMode(args.mode), detector=detector, identity=identity,
                    media_config={**(asdict(profile) if publisher is not None else {}),
                                  "rtsp_url": publisher.output if publisher is not None else None,
                                  "target": target,
                                  "kafka_namespace": args.kafka_namespace if args.transport == "kafka" else None,
                                  "frame_identity": "pixel-v1" if publisher is not None else None},
                    ready=coordinator.ready if coordinator else None)
            finally:
                await client.close()
            if coordinator:
                await coordinator.done(summary)
            print(json.dumps({"output": str((args.out / run_id).resolve()), "summary": summary}, ensure_ascii=False), flush=True)
            failed |= summary["execution_status"] != "PASS"
            if summary["fatal_error"]:
                break
    finally:
        if coordinator:
            await coordinator.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
