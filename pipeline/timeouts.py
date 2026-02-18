from __future__ import annotations

import time

MAX_CLIP_SECONDS = 10 * 60


def planned_total_timeout_seconds(planned_clip_count: int) -> int:
    return planned_clip_count * MAX_CLIP_SECONDS


class TimeoutGuard:
    def __init__(self, per_clip_seconds: int, total_seconds: int) -> None:
        self.per_clip_seconds = per_clip_seconds
        self.total_seconds = total_seconds
        self.total_started = time.monotonic()

    def start_clip(self) -> float:
        self.check_total()
        return time.monotonic()

    def check_clip(self, clip_started: float, clip_index: int) -> float:
        elapsed = time.monotonic() - clip_started
        if elapsed > self.per_clip_seconds:
            raise TimeoutError(
                f"Clip {clip_index} exceeded per-clip timeout: {elapsed:.2f}s > {self.per_clip_seconds}s"
            )
        return elapsed

    def check_total(self) -> float:
        elapsed = time.monotonic() - self.total_started
        if elapsed > self.total_seconds:
            raise TimeoutError(f"Run exceeded total timeout: {elapsed:.2f}s > {self.total_seconds}s")
        return elapsed

    def remaining_total_seconds(self) -> float:
        return self.total_seconds - (time.monotonic() - self.total_started)

    def remaining_clip_seconds(self, clip_started: float) -> float:
        return self.per_clip_seconds - (time.monotonic() - clip_started)
