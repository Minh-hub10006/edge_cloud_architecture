"""Minimal Cloud ASGI app that connects detector results to ResultHub/WebSocket."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from typing import Any

from cloud_core import ResultHub, SubmitStatus, result_to_payload
from contracts import Box, DetectionResult, FrameIdentity, ModelMode
from media.profile import media_path
from .overlay_page import render_overlay_page


def detection_result_from_payload(data: dict[str, Any]) -> DetectionResult:
    frame = FrameIdentity(
        node_id=str(data["node_id"]),
        camera_id=str(data["camera_id"]),
        stream_id=str(data["stream_id"]),
        stream_epoch=int(data["stream_epoch"]),
        source_frame_id=int(data["source_frame_id"]),
        capture_utc_ns=int(data["capture_utc_ns"]),
        source_pts=int(data["source_pts"]),
    )
    boxes = tuple(
        Box(
            left=float(item["left"]), top=float(item["top"]),
            right=float(item["right"]), bottom=float(item["bottom"]),
            confidence=float(item["confidence"]), class_id=int(item["class_id"]),
            class_name=str(item["class_name"]),
            calibrated_confidence=(
                None if item.get("calibrated_confidence") is None
                else float(item["calibrated_confidence"])
            ),
        )
        for item in data["boxes"]
    )
    return DetectionResult(
        frame=frame,
        model_mode=ModelMode(data["model_mode"]),
        model_id=str(data["model_id"]),
        decision_id=str(data["decision_id"]),
        boxes=boxes,
        inference_ms=float(data["inference_ms"]),
        produced_utc_ns=int(data["produced_utc_ns"]),
        request_id=data.get("request_id"),
        preview_pts=(None if data.get("preview_pts") is None else int(data["preview_pts"])),
        schema_version=int(data["schema_version"]),
        attributes=data.get("attributes", {}),
        confidence_calibration_id=data.get("confidence_calibration_id"),
    )


def create_app(hub: ResultHub | None = None):
    """Create a small ASGI app for detector adapters and metadata clients.

    The result endpoint is intended as a local integration seam for this
    prototype. Production model adapters can call ``hub.submit`` directly.
    """

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route, WebSocketRoute
    from starlette.websockets import WebSocketDisconnect

    result_hub = hub or ResultHub()

    @asynccontextmanager
    async def lifespan(app):
        selected = os.getenv("CLOUD_TRANSPORT", "").strip().lower()
        if not selected:
            selected = "grpc" if os.getenv("CLOUD_GRPC_ENABLED", "0").strip() == "1" else "disabled"
        if selected not in {"grpc", "kafka", "disabled"}:
            raise RuntimeError("CLOUD_TRANSPORT must be grpc, kafka, or disabled")
        if selected == "disabled":
            yield
            return

        # Shared model/cache path; only the message transport differs.
        if hasattr(result_hub.tracker, "prepare"):
            await asyncio.to_thread(result_hub.tracker.prepare)
        import av  # noqa: F401 - fail startup clearly if the RTSP decoder is absent
        from cloud_core import CloudFrameCache
        from cloud_core.grpc_service import CloudDataPlaneServicer
        from cloud_core.rtsp_frame_tap import RTSPFrameTapManager
        from contracts.proto import edge_video_platform_pb2_grpc as pb_grpc
        from inference import phase7_cloud_catalog

        frame_cache = CloudFrameCache(
            max_frames_per_stream=int(os.getenv("CLOUD_FRAME_CACHE_SIZE", "64")),
            ttl_ns=int(float(os.getenv("CLOUD_FRAME_CACHE_TTL_S", "5")) * 1_000_000_000),
        )
        async def retire_stream(key):
            await result_hub.close_stream(key)
            servicer.last_inference.pop(key, None)

        frame_taps = RTSPFrameTapManager(
            frame_cache,
            rtsp_base=os.getenv("CLOUD_RTSP_BASE", "rtsp://127.0.0.1:8554"),
            fps=int(os.getenv("CLOUD_PREVIEW_FPS", "15")),
            queue_size=int(os.getenv("CLOUD_FRAME_TAP_QUEUE", "8")),
            idle_timeout_s=float(os.getenv("CLOUD_STREAM_IDLE_S", "30")),
            on_stream_idle=retire_stream,
        )
        catalog = phase7_cloud_catalog(dfine_onnx=os.getenv("DFINE_FULL_ONNX") or None,
            rtdetrv2_onnx=os.getenv("RTDETRV2_SMALL_ONNX") or None,
            lwdetr_onnx=os.getenv("LWDETR_LARGE_ONNX") or None,
            dfinex_onnx=os.getenv("DFINEX_ONNX") or None,
            enabled=tuple(item.strip() for item in os.getenv("CLOUD_MODELS", "dfine").split(",")))
        adapters = {
            mode: await asyncio.to_thread(catalog.load, mode)
            for mode in catalog.modes()
        }
        servicer = CloudDataPlaneServicer(
            hub=result_hub, frame_cache=frame_cache, frame_taps=frame_taps,
            cloud_adapters=adapters,
            max_cloud_inflight=int(os.getenv("CLOUD_MAX_INFLIGHT", os.getenv("CLOUD_DFINE_MAX_INFLIGHT", "2"))),
            preview_fps=int(os.getenv("CLOUD_PREVIEW_FPS", "15")),
        )
        app.state.frame_cache = frame_cache
        app.state.servicer = servicer
        app.state.frame_taps = frame_taps
        app.state.data_plane_transport = selected
        if os.getenv("CLOUD_BENCHMARK", "0") == "1":
            from benchmark.cloud import configure
            await asyncio.to_thread(configure, servicer)
        server = None
        kafka_worker = None
        control_service = control_worker = control_sampler = scheduler_task = scheduler_log = None
        if os.getenv("CLOUD_CONTROL_V1", "0") == "1":
            if servicer.benchmark_enabled:
                raise RuntimeError("run CONTROL_V1 and benchmark in separate Cloud processes")
            import json
            from cloud_core.control_v1 import CloudTelemetry, ControlService, sample_forever
            capacities = json.loads(os.getenv("CLOUD_MODEL_CAPACITY_RPS", "{}"))
            if not isinstance(capacities, dict):
                raise ValueError("CLOUD_MODEL_CAPACITY_RPS must be a JSON object of measured capacities")
            telemetry = CloudTelemetry(servicer, capacities,
                window_ms=float(os.getenv("CONTROL_V1_WINDOW_MS", "1000")))
            control_service = ControlService(servicer, telemetry)
            servicer.control_ticket_observer = telemetry
            app.state.control_v1 = control_service
            # docs/PLAN_CONTROL_V3_VI.md: per-stream commands from one Cloud scheduler (gRPC only).
            scheduler_spec = os.getenv("CLOUD_SCHEDULER", "")
            if scheduler_spec:
                if selected != "grpc":
                    raise RuntimeError("CLOUD_SCHEDULER needs the gRPC transport")
                from cloud_core.scheduler_v3 import Scheduler, StreamRegistry, load_policy
                scheduler_log_path = os.getenv("CLOUD_SCHEDULER_LOG", "")
                scheduler_log = open(scheduler_log_path, "x", encoding="utf-8") if scheduler_log_path else None
                control_service.registry = StreamRegistry()
                control_service.scheduler = Scheduler(
                    control_service.registry, telemetry,
                    load_policy(scheduler_spec, float(os.getenv("CLOUD_RULES_CAPACITY_RPS", "85"))),
                    period_s=float(os.getenv("CLOUD_SCHEDULER_PERIOD_S", "5")), log=scheduler_log)
        if selected == "grpc":
            import grpc
            from contracts.proto import edge_video_platform_pb2_grpc as pb_grpc
            server = grpc.aio.server(options=(
                ("grpc.max_send_message_length", 8 * 1024 * 1024),
                ("grpc.max_receive_message_length", 8 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 10_000),
            ))
            pb_grpc.add_EdgeCloudDataPlaneServicer_to_server(servicer, server)
            if servicer.benchmark_enabled:
                from benchmark.cloud import install_grpc
                install_grpc(server, servicer)
            if control_service is not None:
                from cloud_core.control_v1 import install_grpc
                install_grpc(server, control_service)
            bind = os.getenv("CLOUD_GRPC_BIND", "0.0.0.0:50051")
            if server.add_insecure_port(bind) == 0:
                raise RuntimeError(f"could not bind Cloud gRPC data plane at {bind}")
            await server.start()
            app.state.grpc_server = server
            endpoint = f"gRPC at {bind}"
        else:
            from contracts.kafka_transport import KafkaCloudWorker
            bootstrap = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
            namespace = os.getenv("CLOUD_KAFKA_NAMESPACE", "")
            if namespace:
                if not servicer.benchmark_enabled:
                    raise RuntimeError("CLOUD_KAFKA_NAMESPACE requires CLOUD_BENCHMARK=1")
                kafka_worker = KafkaCloudWorker(servicer, bootstrap, topic_namespace=namespace)
            else:
                kafka_worker = KafkaCloudWorker(servicer, bootstrap)
            await kafka_worker.start()
            app.state.kafka_worker = kafka_worker
            endpoint = f"Kafka at {bootstrap}"
            if control_service is not None:
                from contracts.control_v1 import KAFKA_NAMESPACE
                control_worker = KafkaCloudWorker(control_service, bootstrap,
                                                  topic_namespace=KAFKA_NAMESPACE)
                try:
                    await control_worker.start()
                except BaseException:
                    await control_worker.close()
                    await kafka_worker.close()
                    await control_service.close()
                    await frame_taps.close()
                    raise
                app.state.control_kafka_worker = control_worker
        control_log = None
        if control_service is not None:
            # Exclusive create: never append a new Cloud session onto a previous dataset file.
            log_path = os.getenv("CONTROL_V1_TELEMETRY_JSONL", "")
            control_log = open(log_path, "x", encoding="utf-8") if log_path else None
            control_sampler = asyncio.create_task(sample_forever(
                control_service.telemetry, kafka_worker=control_worker, log=control_log))
            app.state.control_sampler = control_sampler
            if control_service.scheduler is not None:
                scheduler_task = asyncio.create_task(control_service.scheduler.run())
                print(f"[Cloud] scheduler {control_service.scheduler.policy.name} every "
                      f"{control_service.scheduler.period_s:g} s", flush=True)
        loaded = ", ".join(
            f"{mode.value}={adapter.model_id} ({getattr(adapter, 'runtime', 'unknown')})"
            for mode, adapter in adapters.items()
        )
        print(f"[Cloud] {selected} data plane active ({endpoint}); {loaded}", flush=True)
        try:
            yield
        finally:
            if control_sampler is not None:
                control_sampler.cancel()
                await asyncio.gather(control_sampler, return_exceptions=True)
            if scheduler_task is not None:
                scheduler_task.cancel()
                await asyncio.gather(scheduler_task, return_exceptions=True)
            if scheduler_log is not None:
                scheduler_log.close()
            if control_log is not None:
                control_log.close()
            if control_worker is not None:
                await control_worker.close()
            if kafka_worker is not None:
                await kafka_worker.close()
            if server is not None:
                await server.stop(grace=5)
            sampler = getattr(servicer, "benchmark_sampler", None)
            if sampler is not None:
                await asyncio.to_thread(sampler.close)
            if control_service is not None:
                await control_service.close()
            await frame_taps.close()

    async def health(_request: Request):
        control_worker = getattr(_request.app.state, "control_kafka_worker", None)
        control_sampler = getattr(_request.app.state, "control_sampler", None)
        if ((control_worker and any(task.done() for task in control_worker._tasks))
                or (control_sampler is not None and control_sampler.done())):
            return JSONResponse({"status": "error", "detail": "Control v1 telemetry/worker stopped"}, status_code=503)
        worker = getattr(_request.app.state, "kafka_worker", None)
        if worker and any(task.done() for task in worker._tasks):
            return JSONResponse({"status": "error", "detail": "Kafka worker stopped; check Cloud log"}, status_code=503)
        service = getattr(_request.app.state, "servicer", None)
        return JSONResponse({"status": "ok",
            "transport": getattr(_request.app.state, "data_plane_transport", "disabled"),
            "cloud_model_modes": [mode.value for mode in service.cloud_adapters] if service else [],
            "wire_modes": [mode.value for mode in ModelMode],
            "run_id": os.getenv("CLOUD_RUN_ID", ""),
            "tracker_options": getattr(result_hub.tracker, "options", None),
            "vehicle_merge_iou": getattr(result_hub.tracker, "vehicle_merge_iou", None),
            "class_allowlist": sorted(getattr(result_hub.tracker, "class_allowlist", None) or []) or None})

    async def snapshot(request: Request):
        key = (request.path_params["node_id"], request.path_params["camera_id"],
               request.path_params["stream_id"], int(request.path_params["epoch"]))
        service = getattr(request.app.state, "servicer", None)
        entry = service.last_inference.get(key) if service else None
        if entry is None:
            return JSONResponse({"error": "no recent Cloud inference frame for this stream"}, status_code=404)
        pixels, result = entry
        def annotate():
            import cv2
            image = pixels.copy()
            h, w = image.shape[:2]
            for box in result.boxes:
                a, b = (round(box.left*w), round(box.top*h)), (round(box.right*w), round(box.bottom*h))
                cv2.rectangle(image, a, b, (80, 240, 100), 2)
            cv2.putText(image, f"source frame {result.frame.source_frame_id} | pts {result.preview_pts}",
                        (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise RuntimeError("JPEG encode failed")
            return encoded.tobytes()
        from starlette.responses import Response
        return Response(await asyncio.to_thread(annotate), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Source-Frame-Id": str(result.frame.source_frame_id)})

    async def stream_status(_request: Request):
        return JSONResponse({"streams": result_hub.stream_status()}, headers={"Cache-Control": "no-store"})

    async def live_dashboard(_request: Request):
        import os
        from starlette.responses import HTMLResponse
        from web.live_dashboard import render_dashboard
        try:
            nodes = [n.strip() for n in os.getenv("CLOUD_LIVE_NODES", "edge-246,edge-252,edge-247").split(",") if n.strip()]
            html = render_dashboard(nodes, fps=float(os.getenv("CLOUD_PREVIEW_FPS", "15")))
        except ValueError as exc:
            return JSONResponse({"error": f"CLOUD_LIVE_NODES/CLOUD_PREVIEW_FPS: {exc}"}, status_code=500)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    def render_live_page(request: Request, key):
        try:
            media_path(key)  # validate path segments before embedding in URLs
            fps = float(request.query_params.get("fps", "15"))
            # Browser connects directly to MediaMTX for WHEP; override when it
            # is exposed behind a different hostname or TLS reverse proxy.
            import json
            import os
            # v50: video stays on each Edge's MediaMTX ({"edge-246": "http://192.168.180.246:8889", ...}).
            media_base = (json.loads(os.getenv("CLOUD_EDGE_WEBRTC_BASES") or "{}").get(key[0])
                          or os.getenv("CLOUD_MEDIAMTX_WEBRTC_BASE"))
            allow_smoke = (
                os.getenv("CLOUD_ENABLE_OVERLAY_SMOKE") == "1"
                and request.query_params.get("smoke") == "1"
            )
            if not media_base:
                scheme = "https" if request.url.scheme == "https" else "http"
                media_base = f"{scheme}://{request.url.hostname}:8889"
            websocket_path = (
                f"/v1/streams/{key[0]}/{key[1]}/{key[2]}/{key[3]}/metadata"
            )
            html = render_overlay_page(
                key, media_base=media_base, fps=fps, websocket_path=websocket_path,
                allow_smoke=allow_smoke,
                buffer_ms=int(request.query_params.get("buffer_ms", "500")),
                smooth_ms=int(request.query_params.get("smooth_ms", "0")),
                review=request.query_params.get("review") == "1",
                embed=request.query_params.get("embed") == "1",
            )
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        from starlette.responses import HTMLResponse
        return HTMLResponse(html)

    async def live_page(request: Request):
        try:
            key = (
                request.path_params["node_id"], request.path_params["camera_id"],
                request.path_params["stream_id"], int(request.path_params["epoch"]),
            )
            return render_live_page(request, key)
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)

    async def live_latest_page(request: Request):
        node_id = request.path_params["node_id"]
        camera_id = request.path_params["camera_id"]
        stream_id = request.path_params["stream_id"]
        try:
            media_path((node_id, camera_id, stream_id, 0))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        taps = getattr(request.app.state, "frame_taps", None)
        active = taps.active_streams() if taps is not None else {}
        matches = [(key, updated) for key, updated in active.items()
                   if key[:3] == (node_id, camera_id, stream_id)]
        if not matches:
            return JSONResponse(
                {"error": "No active stream epoch yet; start Edge publishing, then reload this page."},
                status_code=404,
                headers={"Cache-Control": "no-store"},
            )
        key, _ = max(matches, key=lambda item: item[1])
        return render_live_page(request, key)

    async def ingest_result(request: Request):
        try:
            result = detection_result_from_payload(await request.json())
        except (KeyError, TypeError, ValueError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        outcome = await result_hub.submit(result)
        body: dict[str, Any] = {"status": outcome.status.value}
        if outcome.result is not None:
            body["result"] = result_to_payload(outcome.result)
        code = 200 if outcome.status is SubmitStatus.ACCEPTED else 409
        return JSONResponse(body, status_code=code)

    async def metadata_socket(websocket):
        key = (
            websocket.path_params["node_id"],
            websocket.path_params["camera_id"],
            websocket.path_params["stream_id"],
            int(websocket.path_params["epoch"]),
        )
        await websocket.accept()
        subscriber_id, queue = result_hub.subscribe(key)
        async def forward():
            while True:
                payload = await queue.get()
                try:
                    await websocket.send_json(payload)
                finally:
                    queue.task_done()
        async def watch_disconnect():
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
        tasks = [asyncio.create_task(forward()), asyncio.create_task(watch_disconnect())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                try:
                    task.result()
                except WebSocketDisconnect:
                    pass
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            result_hub.unsubscribe(subscriber_id)

    return Starlette(lifespan=lifespan, routes=[
        Route("/healthz", health, methods=["GET"]),
        Route("/v1/streams/{node_id:str}/{camera_id:str}/{stream_id:str}/{epoch:int}/snapshot", snapshot),
        Route("/v1/results", ingest_result, methods=["POST"]),
        Route("/v1/streams/status", stream_status, methods=["GET"]),
        Route("/live", live_dashboard, methods=["GET"]),
        Route(
            "/v1/streams/{node_id:str}/{camera_id:str}/{stream_id:str}/{epoch:int}/live",
            live_page, methods=["GET"],
        ),
        Route(
            "/v1/streams/{node_id:str}/{camera_id:str}/{stream_id:str}/live",
            live_latest_page, methods=["GET"],
        ),
        WebSocketRoute(
            "/v1/streams/{node_id:str}/{camera_id:str}/{stream_id:str}/{epoch:int}/metadata",
            metadata_socket,
        ),
    ])


def main() -> None:
    import os
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("CLOUD_APP_HOST", "127.0.0.1"),
        port=int(os.getenv("CLOUD_APP_PORT", "8765")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
