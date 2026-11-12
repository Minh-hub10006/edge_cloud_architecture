"""Low-overhead telemetry for the isolated benchmark package."""

from .trace import FrameTrace, JsonlTraceWriter, STAGES

__all__ = ["FrameTrace", "JsonlTraceWriter", "STAGES"]
