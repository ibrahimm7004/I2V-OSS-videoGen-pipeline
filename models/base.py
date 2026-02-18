from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ClipRequest:
    clip_index: int
    prompt: str
    negative_prompt: str | None
    input_image: Path | None
    output_video_path: Path
    duration_seconds: float
    fps: int
    frames: int
    steps: int
    seed: int
    width: int
    height: int
    params: dict[str, Any]
    global_params: dict[str, Any]
    dry_run: bool = False
    max_runtime_seconds: int = 600


@dataclass(frozen=True)
class ClipResult:
    output_video_path: Path
    model_id: str
    model_version: str
    runtime_seconds: float
    extra_metadata: dict[str, Any] = field(default_factory=dict)


class ModelAdapter:
    def __init__(self, model_id: str, model_version: str, config: Any) -> None:
        self.model_id = model_id
        self.model_version = model_version
        self.config = config

    def generate_clip(self, request: ClipRequest) -> ClipResult:
        raise NotImplementedError

