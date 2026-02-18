from __future__ import annotations

import hashlib
import time
from pathlib import Path

from pipeline.ffmpeg_utils import create_color_clip

from .base import ClipRequest, ClipResult, ModelAdapter


def _seed_to_hex(seed: int, prompt: str) -> str:
    digest = hashlib.sha256(f"{seed}:{prompt}".encode("utf-8")).hexdigest()
    return f"#{digest[:6]}"


class MockAdapter(ModelAdapter):
    """
    Generates deterministic dummy MP4 clips for dry-run/testing.
    """

    def _target_duration_seconds(self, request: ClipRequest) -> float:
        if request.frames and request.fps > 0:
            return max(0.2, float(request.frames) / float(request.fps))
        return max(0.2, float(request.duration_seconds))

    def _try_create_with_opencv(self, request: ClipRequest, started: float) -> bool:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return False

        output = request.output_video_path
        output.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output), fourcc, float(request.fps), (request.width, request.height))
        if not writer.isOpened():
            return False

        color_hex = _seed_to_hex(request.seed, request.prompt).lstrip("#")
        # OpenCV uses BGR order.
        color = tuple(int(color_hex[i : i + 2], 16) for i in (4, 2, 0))
        total_frames = max(1, request.frames)
        for frame_index in range(total_frames):
            if time.monotonic() - started > request.max_runtime_seconds:
                raise TimeoutError(f"Mock adapter exceeded clip timeout ({request.max_runtime_seconds}s)")
            frame = np.full((request.height, request.width, 3), color, dtype=np.uint8)
            label = f"mock clip={request.clip_index} frame={frame_index}"
            cv2.putText(frame, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            writer.write(frame)
        writer.release()
        return output.exists()

    def generate_clip(self, request: ClipRequest) -> ClipResult:
        start = time.monotonic()
        output_path = Path(request.output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        created = self._try_create_with_opencv(request, start)
        if not created:
            duration = self._target_duration_seconds(request)
            color = _seed_to_hex(request.seed, request.prompt)
            create_color_clip(
                ffmpeg_bin=self.config.ffmpeg_bin,
                output_path=output_path,
                duration_seconds=duration,
                fps=request.fps,
                width=request.width,
                height=request.height,
                color_hex=color,
                timeout_seconds=request.max_runtime_seconds,
            )

        runtime = time.monotonic() - start
        return ClipResult(
            output_video_path=output_path,
            model_id=self.model_id,
            model_version=self.model_version,
            runtime_seconds=runtime,
            extra_metadata={
                "backend": "opencv" if created else "ffmpeg",
                "dry_run": request.dry_run,
            },
        )
