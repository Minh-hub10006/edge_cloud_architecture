"""Mock Edge route executor: feedback for window k affects window k+1."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import time_ns

from contracts import DEFAULT_MODEL_REGISTRY, ModelMode, RouteState, WindowStateFeedback

from .capture import CapturedFrame

RoutePolicy = Callable[[WindowStateFeedback, RouteState], tuple[ModelMode, float]]


@dataclass(frozen=True, slots=True)
class ScheduledFrame:
    frame: CapturedFrame
    route: RouteState


class WindowScheduler:
    """Uses source-frame windows and a monotonic clock for mock FPS pacing."""

    def __init__(
        self, *, stream_key: tuple[str, str, str, int], window_size: int,
        initial_mode: ModelMode, target_fps: float,
        policy: RoutePolicy | None = None,
    ) -> None:
        if window_size <= 0 or target_fps <= 0:
            raise ValueError("window_size and target_fps must be positive")
        self.stream_key = stream_key
        self.window_size = window_size
        self.policy = policy or (lambda _feedback, current: (current.model_mode, current.target_fps))
        self.current_route = RouteState(
            stream_key=stream_key, window_id=0, decision_id=self._decision_id(0),
            model_mode=initial_mode, target_fps=target_fps,
            effective_from_frame_id=0, committed_utc_ns=time_ns(),
        )
        self._next_route: RouteState | None = None
        self._next_due_ns: int | None = None
        self._last_frame_id = -1
        self._last_capture_mono_ns = -1

    def _decision_id(self, window_id: int) -> str:
        return ":".join(map(str, (*self.stream_key, window_id)))

    def accept_feedback(self, feedback: WindowStateFeedback) -> RouteState:
        if feedback.stream_key != self.stream_key:
            raise ValueError("feedback belongs to another stream")
        if feedback.window_id != self.current_route.window_id:
            raise ValueError("feedback is stale or from a future window")
        if feedback.decision_id != self.current_route.decision_id:
            raise ValueError("feedback decision_id does not match the current route")
        if feedback.model_id != DEFAULT_MODEL_REGISTRY[self.current_route.model_mode].model_id:
            raise ValueError("feedback model_id does not match the current route")
        window_end = (feedback.window_id + 1) * self.window_size - 1
        if self._last_frame_id < window_end:
            raise ValueError("window is not complete")
        if self._next_route is not None:
            raise ValueError("next-window route has already been committed")
        if feedback.last_frame_id > window_end:
            raise ValueError("feedback spans a later window")
        mode, fps = self.policy(feedback, self.current_route)
        next_window = feedback.window_id + 1
        route = RouteState(
            stream_key=self.stream_key, window_id=next_window,
            decision_id=self._decision_id(next_window),
            model_mode=mode, target_fps=fps,
            effective_from_frame_id=next_window * self.window_size,
            committed_utc_ns=time_ns(),
        )
        self._next_route = route
        return route

    def schedule(self, frame: CapturedFrame) -> ScheduledFrame | None:
        if frame.identity.stream_key != self.stream_key:
            raise ValueError("frame belongs to another stream")
        frame_id = frame.identity.source_frame_id
        if frame_id <= self._last_frame_id:
            raise ValueError("source frames must arrive in strictly increasing order")
        if frame.capture_mono_ns < self._last_capture_mono_ns:
            raise ValueError("capture monotonic time moved backwards")
        self._last_frame_id = frame_id
        self._last_capture_mono_ns = frame.capture_mono_ns
        window_id = frame_id // self.window_size
        if window_id != self.current_route.window_id:
            if self._next_route is not None and self._next_route.window_id == window_id:
                self.current_route = self._next_route
                self._next_route = None
            else:
                self._next_route = None
                self.current_route = RouteState(
                    stream_key=self.stream_key, window_id=window_id,
                    decision_id=self._decision_id(window_id),
                    model_mode=self.current_route.model_mode,
                    target_fps=self.current_route.target_fps,
                    effective_from_frame_id=window_id * self.window_size,
                    committed_utc_ns=time_ns(),
                )
            self._next_due_ns = None
        if self._next_due_ns is not None and frame.capture_mono_ns < self._next_due_ns:
            return None
        self._next_due_ns = frame.capture_mono_ns + round(1_000_000_000 / self.current_route.target_fps)
        return ScheduledFrame(frame, self.current_route)
