from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from pipeline.job_schema import JobSpec
from pipeline.utils import load_structured_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MODEL_IDS = {"wan22_ti2v_5b", "hunyuan_i2v", "cogvideox15_5b_i2v"}


class HFCacheSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    HF_HOME: str
    HF_HUB_CACHE: str


class SeedStrategy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str
    base_seed: int
    per_clip_offsets: list[int]


class RunSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    model_id: str
    output_dir: str
    seed_strategy: SeedStrategy
    hf_cache: HFCacheSpec


class VideoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: int = Field(gt=0)
    clip_duration_sec: float = Field(gt=0)
    num_clips: int = Field(gt=0)


class GenerationDefaults(BaseModel):
    model_config = ConfigDict(extra="allow")
    steps: int = Field(gt=0)
    guidance_scale: float | int
    sampler: str


class RifeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    boundary_sec: float | int


class StitchingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    mode: str
    crossfade_sec: float | int
    rife_boundary_smoothing: RifeSpec


class PromptsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    global_prompt: str
    negative_prompt: str
    continuity_rules: list[str]


class InputsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_images: list[str]


class ShotPack(BaseModel):
    model_config = ConfigDict(extra="allow")
    index: int
    prompt: str
    name: str | None = None


class JobPackSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int | str
    run: RunSpec
    hf_cache: dict[str, Any] | None = None
    video: VideoSpec
    generation_defaults: GenerationDefaults
    stitching: StitchingSpec
    prompts: PromptsSpec
    inputs: InputsSpec
    shots: list[ShotPack]


def is_job_pack_payload(raw: Mapping[str, Any]) -> bool:
    return (
        isinstance(raw, Mapping)
        and isinstance(raw.get("run"), Mapping)
        and "video" in raw
        and "generation_defaults" in raw
        and "prompts" in raw
        and "inputs" in raw
        and "shots" in raw
        and "model" not in raw
    )


def collect_job_pack_issues(pack: JobPackSpec) -> list[str]:
    errors: list[str] = []
    if pack.run.model_id not in ALLOWED_MODEL_IDS:
        errors.append(f"model_id '{pack.run.model_id}' not in {sorted(ALLOWED_MODEL_IDS)}")
    if pack.video.num_clips != len(pack.shots):
        errors.append(f"video.num_clips={pack.video.num_clips} but shots={len(pack.shots)}")
    if len(pack.run.seed_strategy.per_clip_offsets) != pack.video.num_clips:
        errors.append(
            "per_clip_offsets length="
            f"{len(pack.run.seed_strategy.per_clip_offsets)} != num_clips={pack.video.num_clips}"
        )
    if not pack.prompts.global_prompt.strip():
        errors.append("prompts.global_prompt is empty")
    if not pack.prompts.negative_prompt.strip():
        errors.append("prompts.negative_prompt is empty")
    if not pack.prompts.continuity_rules:
        errors.append("prompts.continuity_rules is empty")
    if not pack.stitching.enabled:
        errors.append("stitching.enabled must be true")
    if pack.stitching.mode != "concat":
        errors.append("stitching.mode must be 'concat'")
    if not isinstance(pack.stitching.crossfade_sec, (int, float)):
        errors.append("stitching.crossfade_sec must be numeric")
    return errors


def load_job_pack(job_path: Path) -> JobPackSpec:
    raw = load_structured_file(job_path)
    return JobPackSpec.model_validate(raw)


def _resolve_input_image_path(
    input_image_value: str | None,
    *,
    source_job_path: Path | None,
) -> str | None:
    if not input_image_value:
        return None

    candidate = Path(input_image_value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    base_candidates: list[Path] = [REPO_ROOT]
    if source_job_path is not None:
        base_candidates.append(source_job_path.resolve().parent)
    base_candidates.append(Path.cwd().resolve())

    for base in base_candidates:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return str(resolved)

    # Keep deterministic fallback to repo-root-based absolute path.
    return str((REPO_ROOT / candidate).resolve())


def convert_job_pack_to_runtime_job(
    pack: JobPackSpec,
    *,
    run_id: str,
    model_id_override: str | None = None,
    output_root_override: str | None = None,
    dry_run: bool = True,
    fast_mode: bool = False,
    source_job_path: Path | None = None,
) -> dict[str, Any]:
    fps = pack.video.fps if not fast_mode else max(1, min(4, pack.video.fps))
    duration = pack.video.clip_duration_sec if not fast_mode else min(pack.video.clip_duration_sec, 0.5)
    frames = max(1, int(round(duration * fps)))
    steps = int(pack.generation_defaults.steps) if not fast_mode else min(8, int(pack.generation_defaults.steps))
    width = int(pack.video.width) if not fast_mode else min(512, int(pack.video.width))
    height = int(pack.video.height) if not fast_mode else min(512, int(pack.video.height))

    shots: list[dict[str, Any]] = []
    for idx, shot in enumerate(pack.shots):
        seed_offset = pack.run.seed_strategy.per_clip_offsets[idx]
        shots.append(
            {
                "prompt": shot.prompt,
                "negative_prompt": pack.prompts.negative_prompt,
                "duration_seconds": duration,
                "steps": steps,
                "fps": fps,
                "frames": frames,
                "seed": int(pack.run.seed_strategy.base_seed + seed_offset),
                "width": width,
                "height": height,
                "params": {
                    "cfg": pack.generation_defaults.guidance_scale,
                    "sampler": pack.generation_defaults.sampler,
                },
            }
        )

    input_image = _resolve_input_image_path(
        pack.inputs.initial_images[0] if pack.inputs.initial_images else None,
        source_job_path=source_job_path,
    )

    runtime = {
        "job_name": run_id,
        "run_id": run_id,
        "model": {"id": model_id_override or pack.run.model_id, "version": "TODO_MODEL_VERSION"},
        "output_root": output_root_override or pack.run.output_dir,
        "dry_run": dry_run,
        "global_params": {
            "cfg": pack.generation_defaults.guidance_scale,
            "sampler": pack.generation_defaults.sampler,
        },
        "constants": {
            "global_prompt": pack.prompts.global_prompt,
            "continuity_rules": pack.prompts.continuity_rules,
        },
        "input_image": input_image,
        "shots": shots,
    }
    JobSpec.model_validate(runtime)
    return runtime


def load_runtime_job_payload(
    job_path: Path,
    *,
    run_id_override: str | None = None,
    output_root_override: str | None = None,
    model_id_override: str | None = None,
    dry_run_override: bool | None = None,
    fast_mode: bool = False,
) -> tuple[dict[str, Any], str, JobPackSpec | None]:
    raw = load_structured_file(job_path)
    if is_job_pack_payload(raw):
        pack = JobPackSpec.model_validate(raw)
        issues = collect_job_pack_issues(pack)
        if issues:
            message = "; ".join(issues)
            raise ValueError(message)
        run_id = run_id_override or pack.run.id
        runtime = convert_job_pack_to_runtime_job(
            pack,
            run_id=run_id,
            model_id_override=model_id_override,
            output_root_override=output_root_override,
            dry_run=False if dry_run_override is None else dry_run_override,
            fast_mode=fast_mode,
            source_job_path=job_path,
        )
        return runtime, "job_pack", pack

    runtime_payload = dict(raw)
    if run_id_override:
        runtime_payload["run_id"] = run_id_override
    if output_root_override:
        runtime_payload["output_root"] = output_root_override
    if dry_run_override is not None:
        runtime_payload["dry_run"] = dry_run_override
    if model_id_override:
        model_obj = runtime_payload.get("model")
        if not isinstance(model_obj, dict):
            raise ValueError("Runtime job is missing object field 'model' for model override.")
        model_obj = dict(model_obj)
        model_obj["id"] = model_id_override
        runtime_payload["model"] = model_obj
    JobSpec.model_validate(runtime_payload)
    return runtime_payload, "runtime", None
