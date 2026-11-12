"""Shared gRPC message size — tier1_fp32 ~17.2 MB."""
GRPC_MAX_MESSAGE_BYTES = 24 * 1024 * 1024


def grpc_channel_options() -> list[tuple[str, int]]:
    return [
        ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_BYTES),
    ]
