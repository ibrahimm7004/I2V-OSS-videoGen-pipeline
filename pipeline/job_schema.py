from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ModelSpec(BaseModel):
    id: str
    version: str = "TODO_MODEL_VERSION"


class ShotSpec(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    duration_seconds: float = Field(default=2.0, gt=0)
    steps: int = Field(default=20, gt=0)
    fps: int = Field(default=8, gt=0)
    frames: int | None = Field(default=None, gt=0)
    seed: int | None = None
    width: int = Field(default=512, gt=0)
    height: int = Field(default=512, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_default_frames(self) -> "ShotSpec":
        if self.frames is None:
            self.frames = max(1, int(round(self.duration_seconds * self.fps)))
        return self


class JobSpec(BaseModel):
    job_name: str = "video-gen-job"
    run_id: str | None = None
    model: ModelSpec
    input_image: Path | None = None
    output_root: Path = Path("outputs")
    dry_run: bool = False
    global_params: dict[str, Any] = Field(default_factory=dict)
    constants: dict[str, Any] = Field(default_factory=dict)
    shots: list[ShotSpec]

    @model_validator(mode="after")
    def ensure_shots(self) -> "JobSpec":
        if not self.shots:
            raise ValueError("JobSpec requires at least one shot.")
        return self

    @property
    def planned_clip_count(self) -> int:
        return len(self.shots)

