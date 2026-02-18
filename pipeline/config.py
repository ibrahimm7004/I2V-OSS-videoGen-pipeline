from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path

load_dotenv()

MAX_GUARDRAIL_SECONDS = 10 * 60


class PipelineConfig(BaseModel):
    output_root: Path = Field(default_factory=lambda: Path(os.getenv("PIPELINE_OUTPUT_ROOT", "outputs")))
    ffmpeg_bin: str = Field(default_factory=get_ffmpeg_path)
    ffprobe_bin: str = Field(default_factory=get_ffprobe_path)
    clip_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("CLIP_TIMEOUT_SECONDS", str(MAX_GUARDRAIL_SECONDS)))
    )
    wall_timeout_seconds_per_clip: int = Field(
        default_factory=lambda: int(os.getenv("WALL_TIMEOUT_SECONDS_PER_CLIP", str(MAX_GUARDRAIL_SECONDS)))
    )
    hf_home: str | None = Field(default_factory=lambda: os.getenv("HF_HOME"))
    hf_hub_cache: str | None = Field(default_factory=lambda: os.getenv("HF_HUB_CACHE"))


def load_config(overrides: dict | None = None) -> PipelineConfig:
    config = PipelineConfig()
    if overrides:
        config = config.model_copy(update=overrides)
    return config
