from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path

load_dotenv()

MAX_GUARDRAIL_SECONDS = 10 * 60


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class PipelineConfig(BaseModel):
    output_root: Path = Field(default_factory=lambda: Path(os.getenv("PIPELINE_OUTPUT_ROOT", "outputs")))
    ffmpeg_bin: str = Field(default_factory=get_ffmpeg_path)
    ffprobe_bin: str = Field(default_factory=get_ffprobe_path)
    wan22_repo_id: str = Field(
        default_factory=lambda: os.getenv("WAN22_REPO_ID", "Wan-AI/Wan2.2-TI2V-5B-Diffusers")
    )
    wan22_export_quality: int = Field(default_factory=lambda: int(os.getenv("WAN22_EXPORT_QUALITY", "9")))
    clip_timeout_seconds: int = Field(
        default_factory=lambda: int(os.getenv("CLIP_TIMEOUT_SECONDS", str(MAX_GUARDRAIL_SECONDS)))
    )
    wall_timeout_seconds_per_clip: int = Field(
        default_factory=lambda: int(os.getenv("WALL_TIMEOUT_SECONDS_PER_CLIP", str(MAX_GUARDRAIL_SECONDS)))
    )
    post_clip_validation_enabled: bool = Field(default_factory=lambda: _env_bool("POST_CLIP_VALIDATION_ENABLED", True))
    post_clip_min_size_bytes: int = Field(default_factory=lambda: int(os.getenv("POST_CLIP_MIN_SIZE_BYTES", "4096")))
    post_clip_min_frame_diff: float | None = Field(
        default_factory=lambda: float(os.getenv("POST_CLIP_MIN_FRAME_DIFF"))
        if os.getenv("POST_CLIP_MIN_FRAME_DIFF") is not None
        else None
    )
    hf_home: str | None = Field(default_factory=lambda: os.getenv("HF_HOME"))
    hf_hub_cache: str | None = Field(default_factory=lambda: os.getenv("HF_HUB_CACHE"))

    def resolve_post_clip_min_frame_diff(self, *, model_id: str, wan_profile: str | None = None) -> float:
        if self.post_clip_min_frame_diff is not None:
            return max(0.0, float(self.post_clip_min_frame_diff))

        if model_id.strip().lower().startswith("wan"):
            profile = (wan_profile or "").strip().lower()
            if profile == "quality":
                return 0.003
            if profile == "smoke":
                return 0.0
        return 0.0


def load_config(overrides: dict | None = None) -> PipelineConfig:
    config = PipelineConfig()
    if overrides:
        config = config.model_copy(update=overrides)
    return config
