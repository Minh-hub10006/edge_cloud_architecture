"""Cloud-side media synchronization primitives.

These modules are model-agnostic. They make a decoded media frame available
only after its Edge-authored MediaFrameRef has been matched.
"""

from .frame_cache import (
    CachedCloudFrame,
    CloudFrameCache,
    CloudFrameTap,
    DecodedMediaFrame,
    FrameLookup,
    FrameLookupStatus,
)
from .result_hub import ResultHub, ResultSubmit, SubmitStatus, result_to_json, result_to_payload
from .tracking import CloudByteTrack

__all__ = [
    "CachedCloudFrame",
    "CloudFrameCache",
    "CloudFrameTap",
    "DecodedMediaFrame",
    "FrameLookup",
    "FrameLookupStatus",
    "CloudByteTrack",
    "ResultHub",
    "ResultSubmit",
    "SubmitStatus",
    "result_to_json",
    "result_to_payload",
]
