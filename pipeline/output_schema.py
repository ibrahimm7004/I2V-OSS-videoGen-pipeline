from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ClipParamsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: int
    fps: int
    frames: int
    height: int
    width: int
    cfg: float | int | None = None
    sampler: str | None = None


class ClipLogSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    clip_index: int
    model_id: str
    model_version: str
    git_commit: str
    started_at: str
    ended_at: str
    runtime_sec: float
    prompt_full_text: str
    seed: int | None
    params: ClipParamsSchema
    input_image_path: str | None
    input_image_sha256: str | None
    output_clip_path: str | None
    output_clip_sha256: str | None
    last_frame_path: str | None
    last_frame_sha256: str | None
    adapter_metadata: dict[str, Any] | None = None
    status: Literal["success", "failed", "skipped"]
    error: str | None


class EnvironmentSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python_version: str
    platform: str
    cuda_available: bool
    torch_version: str | None
    nvidia_smi: str | None


class HFCacheSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    HF_HOME: str | None
    HF_HUB_CACHE: str | None


class OutputsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    final_stitched_path: str | None
    bundle_path: str | None


class StitchSettingsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overlap_sec: float
    transition: str
    encoder_settings: dict[str, Any]


class ManifestClipSummarySchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    input_image_path: str | None
    input_image_sha256: str | None
    output_clip_path: str | None
    output_clip_sha256: str | None
    last_frame_path: str | None
    last_frame_sha256: str | None
    runtime_sec: float
    seed: int | None
    status: Literal["success", "failed", "skipped"]


class ManifestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    created_at: str
    job_path: str
    job_sha256: str
    model_id: str
    model_version: str
    git_commit: str
    environment: EnvironmentSchema
    hf_cache: HFCacheSchema
    planned_clips: int
    completed_clips: int
    total_runtime_sec: float
    outputs: OutputsSchema
    stitch_settings: StitchSettingsSchema
    clips: list[ManifestClipSummarySchema]
    status: Literal["completed", "failed", "aborted_timeout", "aborted_user"]
    error: str | None
