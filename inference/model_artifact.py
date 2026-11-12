"""Explicit, checksummed export contract for opt-in detectors (never guess IO)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_IDS = {
    "rtdetrv2_small": "rtdetrv2_r18vd_sp2_120e_coco",
    "lwdetr_large": "lwdetr_large_coco",
    "dfine_x": "dfine_x_obj2coco",
    "rtmdet_small": "rtmdet_s_8xb32_300e_coco",
}
OUTPUT_NAMES = {
    "rtdetrv2_small": ("labels", "boxes", "scores"),
    "lwdetr_large": ("dets", "labels"),
    "dfine_x": ("labels", "boxes", "scores"),
    "rtmdet_small": ("boxes", "scores"),
}
CANDIDATE_PREFIXES = {"rtdetrv2_small": "rtdetrv2_s_candidate", "lwdetr_large": "lwdetr_large_candidate",
                      "dfine_x": "dfine_x_candidate"}


def candidate_model_id(kind: str, digest: str) -> str:
    if kind not in CANDIDATE_PREFIXES:
        raise ValueError("imported candidate supports RT-DETRv2-S, LW-DETR-L and D-FINE-X only")
    return f"{CANDIDATE_PREFIXES[kind]}_{digest[:12]}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_external_references(artifact: Path, expected: set[str]):
    """Read graph metadata without loading external weights before checking inventory."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Imported candidate admission requires onnx for graph/sidecar validation") from exc
    model = onnx.load(str(artifact), load_external_data=False)
    found = set()

    def walk(message):
        if isinstance(message, onnx.TensorProto) and message.data_location == onnx.TensorProto.EXTERNAL:
            metadata = {item.key: item.value for item in message.external_data}
            location = metadata.get("location")
            if location not in expected:
                raise ValueError("ONNX external reference is not in checksummed inventory")
            size = (artifact.parent / location).stat().st_size
            offset = int(metadata.get("offset", "0"))
            length = int(metadata.get("length", str(size - offset)))
            if offset < 0 or length <= 0 or offset + length > size:
                raise ValueError("ONNX external tensor range exceeds sidecar")
            found.add(location)
        for field, value in message.ListFields():
            if field.message_type is not None:
                repeated = field.is_repeated if hasattr(field, "is_repeated") else field.label == field.LABEL_REPEATED
                if repeated:
                    for item in value:
                        walk(item)
                else:
                    walk(value)

    walk(model)
    if found != expected:
        raise ValueError("ONNX external references differ from manifest inventory")
    onnx.checker.check_model(str(artifact))


def _check_bundle(artifact: Path, data: dict):
    """Hash every declared sidecar; never follow paths outside the model folder."""
    if data.get("artifact") != artifact.name or data.get("size_bytes") != artifact.stat().st_size:
        raise ValueError("candidate artifact filename/size mismatch")
    entries = data.get("external_data")
    if not isinstance(entries, list):
        raise ValueError("candidate external_data inventory required")
    seen = set()
    for entry in entries:
        name = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name or name in seen or "/" in name or "\\" in name or ":" in name or name in {".", ".."}:
            raise ValueError("external_data must contain unique adjacent filenames")
        sidecar = (artifact.parent / name).resolve(strict=True)
        if sidecar.parent != artifact.parent or not sidecar.is_file():
            raise ValueError("external_data escapes artifact directory")
        if entry.get("size_bytes") != sidecar.stat().st_size or entry.get("sha256") != sha256(sidecar):
            raise ValueError("external_data size/checksum mismatch")
        seen.add(name)
    # RT-DETRv2: the 10/2026 batch-1 export has a .onnx.data sidecar; the v4 dynamic-batch export is self-contained.
    if data["kind"] == "rtdetrv2_small" and seen not in ({artifact.name + ".data"}, set()):
        raise ValueError("imported RT-DETRv2 expects its matching .onnx.data sidecar or none")
    if data["kind"] in {"lwdetr_large", "dfine_x"} and seen:
        raise ValueError(f"imported {data['kind']} expects a self-contained ONNX with no external_data")
    _check_external_references(artifact, seen)


def load_artifact(path: str, kind: str, manifest_path: str | None = None, *, allow_imported_candidate: bool = False,
                  rtmdet_float_onnx: bool = False):
    if rtmdet_float_onnx and kind != "rtmdet_small":
        raise ValueError("Float ONNX opt-in applies only to RTMDet")
    artifact = Path(path).expanduser().resolve(strict=True)
    manifest = Path(manifest_path).expanduser() if manifest_path else Path(str(artifact) + ".json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    imported = data.get("schema_version") == 2
    if data.get("schema_version") not in {1, 2} or data.get("kind") != kind:
        raise ValueError("artifact manifest schema/kind mismatch")
    digest = sha256(artifact)
    expected_id = candidate_model_id(kind, digest) if imported else MODEL_IDS[kind]
    if data.get("model_id") != expected_id or data.get("sha256") != digest:
        raise ValueError("artifact model identity/checksum mismatch")
    if imported:
        if not allow_imported_candidate:
            raise ValueError("imported candidate requires explicit allow_imported_candidate opt-in")
        if kind not in CANDIDATE_PREFIXES or data.get("manifest_type") != "imported_candidate":
            raise ValueError("imported candidate supports RT-DETRv2-S, LW-DETR-L and D-FINE-X only")
        if data.get("admission", {}).get("production_ready") is not False:
            raise ValueError("imported candidate must not claim production readiness")
        if kind in {"rtdetrv2_small", "dfine_x"} and data.get("output_shapes") != {"labels": [1, 300], "boxes": [1, 300, 4], "scores": [1, 300]}:
            raise ValueError("imported candidate expects batch1 and 300 output queries")
        if kind == "lwdetr_large" and data.get("output_shapes") != {"dets": [1, 300, 4], "labels": [1, 300, 91]}:
            raise ValueError("imported LW-DETR expects batch1, 300 queries and 91 category logits")
        _check_bundle(artifact, data)
    if data.get("input_size") != 640:
        raise ValueError("these candidate exports require fixed input_size=640")
    layout = data.get("input_layout")
    if layout not in ({"nchw", "nhwc"} if kind == "rtmdet_small" else {"nchw"}):
        raise ValueError("unsupported artifact input layout")
    if set(data.get("output_shapes", {})) != set(OUTPUT_NAMES[kind]):
        raise ValueError("artifact output names mismatch")
    for shape in data["output_shapes"].values():
        if not isinstance(shape, list) or not shape or any(type(x) is not int or x <= 0 for x in shape):
            raise ValueError("artifact output shapes must be concrete positive integers (batch=1)")
    if not imported and (not data.get("source_revision") or not data.get("checkpoint_sha256")):
        raise ValueError("artifact requires source revision and checkpoint checksum provenance")
    if not imported and (data.get("label_map") != "coco80_contiguous" or data.get("license") != "Apache-2.0"):
        raise ValueError("artifact requires explicit supported class map/license")
    expected_suffix = ".dlc" if kind == "rtmdet_small" and not rtmdet_float_onnx else ".onnx"
    if artifact.suffix != expected_suffix:
        raise ValueError(f"{kind} requires {expected_suffix}")
    return artifact, data
