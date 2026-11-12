"""Transport selection shared by Edge and Cloud launchers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import os
from typing import Any


class TransportKind(str, Enum):
    GRPC = "grpc"
    KAFKA = "kafka"

    @classmethod
    def configured(cls, value: str | None = None) -> "TransportKind":
        selected = (value or os.getenv("TRANSPORT_MODE", "grpc")).strip().lower()
        try:
            return cls(selected)
        except ValueError as exc:
            raise ValueError(f"TRANSPORT_MODE must be one of: {', '.join(x.value for x in cls)}") from exc


@dataclass(frozen=True, slots=True)
class TransportEnvelope:
    kind: str
    correlation_id: str
    stream_key: tuple[str, str, str, int]
    payload: Any

    def __post_init__(self) -> None:
        if not self.kind or not self.correlation_id:
            raise ValueError("kind and correlation_id are required")


class TransportAdapter(ABC):
    """One adapter is active per benchmark/service run."""

    kind: TransportKind

    @abstractmethod
    async def send(self, envelope: TransportEnvelope) -> None:
        """Publish an envelope, respecting transport backpressure."""

    @abstractmethod
    async def receive(self) -> TransportEnvelope:
        """Receive the next envelope for this adapter session."""
