from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field

from pipeline.job_schema import JobSpec
from pipeline.utils import load_structured_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_MODEL_IDS = {"wan22_ti2v_5b", "hunyuan_i2v", "cogvideox15_5b_i2v"}

DEFAULT_MOTION_SUBJECT = (
    "subtle natural body movement; breathing; small head turns; micro-expressions"
)
DEFAULT_MOTION_ENVIRONMENT = (
    "rain streaks and droplets; neon reflections shimmering on wet asphalt; steam drifting; light wind"
)
DEFAULT_MOTION_CAMERA = (
    "slow subtle dolly-in only; no rapid zoom; no sudden framing changes; subject centered"
)
DEFAULT_MOTION_NOTES = ""


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
    motion_block: bool | None = None


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
    steps: int | None = None
    guidance_scale: float | int | None = None
    motion_strength: float | int | None = None
    cinematic_constraints: bool | None = None
    motion_block: bool | None = None
    motion_subject: str | None = None
    motion_environment: str | None = None
    motion_camera: str | None = None
    motion_notes: str | None = None


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


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _compose_motion_block_text(
    *,
    motion_subject: str,
    motion_environment: str,
    motion_camera: str,
    motion_notes: str = "",
) -> str:
    lines = [
        "MOTION:",
        f"- Subject: {motion_subject}",
        f"- Environment: {motion_environment}",
        f"- Camera: {motion_camera}",
    ]
    if motion_notes.strip():
        lines.append(f"- Notes: {motion_notes.strip()}")
    return "\n".join(lines)


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
    generation_defaults = pack.generation_defaults.model_dump()
    motion_strength_default = generation_defaults.get("motion_strength")
    export_quality = generation_defaults.get("export_quality")
    chain_last_frame = generation_defaults.get("chain_last_frame")
    postprocess = generation_defaults.get("postprocess")
    cinematic_constraints = generation_defaults.get("cinematic_constraints")
    motion_block = generation_defaults.get("motion_block")
    wan_profile = (
        generation_defaults.get("wan_profile")
        or generation_defaults.get("run_profile")
        or generation_defaults.get("profile")
    )
    guidance_default = generation_defaults.get("guidance_scale", pack.generation_defaults.guidance_scale)
    steps_default = generation_defaults.get("steps", pack.generation_defaults.steps)
    if wan_profile == "quality":
        guidance_default = 5.0 if generation_defaults.get("guidance_scale") is None else guidance_default
        steps_default = 60 if generation_defaults.get("steps") is None else steps_default
        if motion_strength_default is None:
            motion_strength_default = 0.70
        if cinematic_constraints is None:
            cinematic_constraints = True
        if motion_block is None:
            motion_block = True
    elif wan_profile == "smoke":
        if cinematic_constraints is None:
            cinematic_constraints = False
        if motion_block is None:
            motion_block = False

    fps = pack.video.fps if not fast_mode else max(1, min(4, pack.video.fps))
    duration = pack.video.clip_duration_sec if not fast_mode else min(pack.video.clip_duration_sec, 0.5)
    frames = max(1, int(round(duration * fps)))
    steps = int(steps_default) if not fast_mode else min(8, int(steps_default))
    width = int(pack.video.width) if not fast_mode else min(512, int(pack.video.width))
    height = int(pack.video.height) if not fast_mode else min(512, int(pack.video.height))

    shots: list[dict[str, Any]] = []
    for idx, shot in enumerate(pack.shots):
        shot_data = shot.model_dump()
        shot_guidance_raw = shot_data.get("guidance_scale")
        shot_guidance = guidance_default if shot_guidance_raw is None else shot_guidance_raw
        shot_motion_raw = shot_data.get("motion_strength")
        shot_motion = motion_strength_default if shot_motion_raw is None else shot_motion_raw
        shot_steps_raw = shot_data.get("steps")
        shot_steps = int(steps if shot_steps_raw is None else shot_steps_raw)
        if fast_mode:
            shot_steps = min(8, shot_steps)
        shot_cinematic_constraints = shot_data.get("cinematic_constraints")
        shot_motion_block = shot_data.get("motion_block")
        motion_block_enabled = _coerce_bool(
            shot_motion_block if shot_motion_block is not None else motion_block,
            False,
        )
        shot_motion_subject = str(shot_data.get("motion_subject") or DEFAULT_MOTION_SUBJECT).strip()
        shot_motion_environment = str(shot_data.get("motion_environment") or DEFAULT_MOTION_ENVIRONMENT).strip()
        shot_motion_camera = str(shot_data.get("motion_camera") or DEFAULT_MOTION_CAMERA).strip()
        shot_motion_notes = str(shot_data.get("motion_notes") or DEFAULT_MOTION_NOTES).strip()
        motion_block_text = _compose_motion_block_text(
            motion_subject=shot_motion_subject,
            motion_environment=shot_motion_environment,
            motion_camera=shot_motion_camera,
            motion_notes=shot_motion_notes,
        )
        shot_params = shot_data.get("params")
        if not isinstance(shot_params, dict):
            shot_params = {}
        seed_offset = pack.run.seed_strategy.per_clip_offsets[idx]
        shots.append(
            {
                "prompt": shot.prompt,
                "negative_prompt": pack.prompts.negative_prompt,
                "duration_seconds": duration,
                "steps": shot_steps,
                "fps": fps,
                "frames": frames,
                "seed": int(pack.run.seed_strategy.base_seed + seed_offset),
                "width": width,
                "height": height,
                "params": {
                    "cfg": shot_guidance,
                    "guidance_scale": shot_guidance,
                    "sampler": pack.generation_defaults.sampler,
                    **({"motion_strength": shot_motion} if shot_motion is not None else {}),
                    **({"export_quality": export_quality} if export_quality is not None else {}),
                    **({"wan_profile": wan_profile} if wan_profile is not None else {}),
                    **(
                        {"cinematic_constraints": _coerce_bool(shot_cinematic_constraints, True)}
                        if shot_cinematic_constraints is not None
                        else {}
                    ),
                    "motion_block": bool(motion_block_enabled),
                    "motion_subject": shot_motion_subject,
                    "motion_environment": shot_motion_environment,
                    "motion_camera": shot_motion_camera,
                    "motion_notes": shot_motion_notes,
                    "motion_block_text": motion_block_text,
                    **shot_params,
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
            "cfg": guidance_default,
            "guidance_scale": guidance_default,
            "sampler": pack.generation_defaults.sampler,
            **({"motion_strength": motion_strength_default} if motion_strength_default is not None else {}),
            **({"export_quality": export_quality} if export_quality is not None else {}),
            **({"chain_last_frame": bool(chain_last_frame)} if chain_last_frame is not None else {}),
            **({"postprocess": postprocess} if isinstance(postprocess, dict) else {}),
            **({"wan_profile": wan_profile} if wan_profile is not None else {}),
            **({"motion_block": _coerce_bool(motion_block, False)} if motion_block is not None else {}),
        },
        "constants": {
            "global_prompt": pack.prompts.global_prompt,
            "continuity_rules": pack.prompts.continuity_rules,
            **(
                {"cinematic_constraints_enabled": _coerce_bool(cinematic_constraints, True)}
                if cinematic_constraints is not None
                else {}
            ),
            **(
                {"motion_block_enabled": _coerce_bool(motion_block, False)}
                if motion_block is not None
                else {}
            ),
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
