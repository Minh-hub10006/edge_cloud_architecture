"""Conversions between immutable domain contracts and generated protobufs."""

from __future__ import annotations

import json
from typing import Any

from contracts import Box, CloudInferenceTicket, DetectionResult, FrameIdentity, MediaFrameRef, ModelMode
from contracts.proto import edge_video_platform_pb2 as pb


_MODE_TO_PROTO = {
    ModelMode.YOLO_EDGE: pb.YOLO_EDGE,
    ModelMode.RTMDET_EDGE: pb.RTMDET_EDGE,
    ModelMode.DFINE_CLOUD: pb.DFINE_CLOUD,
    ModelMode.RTDETR_CLOUD: pb.RTDETR_CLOUD,
    ModelMode.LWDETR_CLOUD: pb.LWDETR_CLOUD,
    ModelMode.YOLO11S_EDGE: pb.YOLO11S_EDGE,
    ModelMode.DFINEX_CLOUD: pb.DFINEX_CLOUD,
    ModelMode.RTDETR_SPLIT_CLOUD: pb.RTDETR_SPLIT_CLOUD,
}
_PROTO_TO_MODE = {value: key for key, value in _MODE_TO_PROTO.items()}


def frame_to_proto(frame: FrameIdentity):
    return pb.FrameIdentity(
        node_id=frame.node_id, camera_id=frame.camera_id, stream_id=frame.stream_id,
        stream_epoch=frame.stream_epoch, source_frame_id=frame.source_frame_id,
        capture_utc_ns=frame.capture_utc_ns, source_pts=frame.source_pts,
    )


def frame_from_proto(frame: Any) -> FrameIdentity:
    return FrameIdentity(
        node_id=frame.node_id, camera_id=frame.camera_id, stream_id=frame.stream_id,
        stream_epoch=frame.stream_epoch, source_frame_id=frame.source_frame_id,
        capture_utc_ns=frame.capture_utc_ns, source_pts=frame.source_pts,
    )


def reference_to_proto(reference: MediaFrameRef):
    return pb.MediaFrameRef(
        schema_version=reference.schema_version, frame=frame_to_proto(reference.frame),
        encoded_seq=reference.encoded_seq, media_pts_ns=reference.media_pts_ns,
        submitted_utc_ns=reference.submitted_utc_ns,
    )


def reference_from_proto(reference: Any) -> MediaFrameRef:
    return MediaFrameRef(
        schema_version=reference.schema_version, frame=frame_from_proto(reference.frame),
        encoded_seq=reference.encoded_seq, media_pts_ns=reference.media_pts_ns,
        submitted_utc_ns=reference.submitted_utc_ns,
    )


def result_to_proto(result: DetectionResult):
    message = pb.DetectionResult(
        schema_version=result.schema_version, frame=frame_to_proto(result.frame),
        model_mode=_MODE_TO_PROTO[result.model_mode], model_id=result.model_id,
        decision_id=result.decision_id, inference_ms=result.inference_ms,
        produced_utc_ns=result.produced_utc_ns, request_id=result.request_id or "",
        attributes_json=json.dumps(dict(result.attributes), separators=(",", ":"), allow_nan=False).encode(),
    )
    if result.preview_pts is not None:
        message.preview_pts = result.preview_pts
    if result.confidence_calibration_id is not None:
        message.confidence_calibration_id = result.confidence_calibration_id
    for box in result.boxes:
        item = message.boxes.add(
            left=box.left, top=box.top, right=box.right, bottom=box.bottom,
            confidence=box.confidence, class_id=box.class_id, class_name=box.class_name,
        )
        if box.track_id is not None:
            item.track_id = box.track_id
        if box.calibrated_confidence is not None:
            item.calibrated_confidence = box.calibrated_confidence
    return message


def result_from_proto(message: Any) -> DetectionResult:
    if message.model_mode not in _PROTO_TO_MODE:
        raise ValueError(f"unsupported protobuf model_mode: {message.model_mode}")
    boxes = tuple(Box(
        left=item.left, top=item.top, right=item.right, bottom=item.bottom,
        confidence=item.confidence, class_id=item.class_id, class_name=item.class_name,
        track_id=item.track_id if item.HasField("track_id") else None,
        calibrated_confidence=(
            item.calibrated_confidence if item.HasField("calibrated_confidence") else None
        ),
    ) for item in message.boxes)
    return DetectionResult(
        schema_version=message.schema_version, frame=frame_from_proto(message.frame),
        model_mode=_PROTO_TO_MODE[message.model_mode], model_id=message.model_id,
        decision_id=message.decision_id, boxes=boxes, inference_ms=message.inference_ms,
        produced_utc_ns=message.produced_utc_ns,
        request_id=message.request_id or None,
        preview_pts=message.preview_pts if message.HasField("preview_pts") else None,
        attributes=json.loads(message.attributes_json) if message.attributes_json else {},
        confidence_calibration_id=(
            message.confidence_calibration_id
            if message.HasField("confidence_calibration_id") else None
        ),
    )


def ticket_to_proto(ticket: CloudInferenceTicket):
    return pb.CloudInferenceTicket(
        schema_version=ticket.schema_version, frame=frame_to_proto(ticket.frame),
        model_mode=_MODE_TO_PROTO[ticket.model_mode], model_id=ticket.model_id,
        decision_id=ticket.decision_id, preview_pts=ticket.preview_pts,
        deadline_utc_ns=ticket.deadline_utc_ns, request_id=ticket.request_id or "", jpeg=ticket.jpeg,
        tensor=ticket.tensor,
    )


def ticket_from_proto(message: Any) -> CloudInferenceTicket:
    if message.model_mode not in _PROTO_TO_MODE:
        raise ValueError(f"unsupported protobuf model_mode: {message.model_mode}")
    return CloudInferenceTicket(
        schema_version=message.schema_version, frame=frame_from_proto(message.frame),
        model_mode=_PROTO_TO_MODE[message.model_mode], model_id=message.model_id,
        decision_id=message.decision_id, preview_pts=message.preview_pts,
        deadline_utc_ns=message.deadline_utc_ns, request_id=message.request_id or None, jpeg=message.jpeg,
        tensor=message.tensor,
    )
