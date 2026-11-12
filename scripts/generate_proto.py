"""Generate protobuf/gRPC Python modules from the v1 schema."""

from __future__ import annotations

from pathlib import Path

from grpc_tools import protoc


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    proto = root / "contracts" / "proto" / "edge_video_platform.proto"
    code = protoc.main([
        "grpc_tools.protoc",
        f"-I{root}",
        f"--python_out={root}",
        f"--grpc_python_out={root}",
        str(proto),
    ])
    if code:
        raise SystemExit(code)
    print("Generated contracts/proto/edge_video_platform_pb2{,_grpc}.py")


if __name__ == "__main__":
    main()
