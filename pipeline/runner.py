from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from models import get_adapter
from models.base import ClipRequest

from .artifacts import build_manifest_base, hash_file, prepare_run_directories, write_clip_log, write_manifest
from .bundling import create_run_bundle
from .clip_validation import validate_clip_output
from .config import MAX_GUARDRAIL_SECONDS, load_config
from .frames import extract_last_frame
from .job_schema import JobSpec
from .output_schema import ClipLogSchema
from .postprocess import run_optional_wan_postprocess
from .prompting import compile_prompt
from .progress import ProgressTracker
from .stitching import stitch_clips_deterministic
from .timeouts import TimeoutGuard, planned_total_timeout_seconds
from .utils import ensure_dir, generate_run_id, load_structured_file, parse_overrides, set_by_dot_path, utc_now_iso


class UserAbortError(RuntimeError):
    pass


def _split_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config_overrides: dict[str, Any] = {}
    job_overrides: dict[str, Any] = {}
    for key, value in overrides.items():
        if key.startswith("config."):
            config_overrides[key[len("config.") :]] = value
        else:
            job_overrides[key] = value
    return config_overrides, job_overrides


def _read_control_action(status_dir: Path) -> str | None:
    if (status_dir / "STOP").exists():
        return "stop"

    action: str | None = None
    control_json = status_dir / "control.json"
    if control_json.exists():
        try:
            payload = json.loads(control_json.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw = payload.get("action")
                if isinstance(raw, str):
                    action = raw.strip().lower()
        except Exception:
            action = None

    if action == "stop":
        return "stop"
    if (status_dir / "PAUSE").exists():
        return "pause"
    if action == "pause":
        return "pause"
    return action


def _control_gate(
    *,
    status_dir: Path,
    progress: ProgressTracker,
    clip_index: int | None,
    percent: float,
    context: str,
) -> None:
    paused_logged = False
    while True:
        action = _read_control_action(status_dir)
        if action == "stop":
            raise UserAbortError(f"User requested stop ({context}).")
        if action == "pause":
            if not paused_logged:
                progress.update(clip_index, "paused", percent, f"Paused by user ({context}).")
                paused_logged = True
            time.sleep(2.0)
            continue
        if paused_logged:
            progress.update(clip_index, "resumed", percent, f"Resumed ({context}).")
        return


def _display_path(path: Path | None, base: Path) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        try:
            return resolved.relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()


def _clip_params(job: JobSpec, shot: Any) -> dict[str, Any]:
    cfg = shot.params.get("cfg")
    if cfg is None:
        cfg = shot.params.get("guidance_scale")
    if cfg is None:
        cfg = job.global_params.get("cfg", job.global_params.get("guidance_scale"))

    sampler = shot.params.get("sampler", job.global_params.get("sampler"))
    return {
        "steps": shot.steps,
        "fps": shot.fps,
        "frames": int(shot.frames or 1),
        "height": shot.height,
        "width": shot.width,
        "cfg": cfg,
        "sampler": sampler,
    }


def _expected_num_frames(request: ClipRequest, adapter_metadata: dict[str, Any] | None) -> int:
    meta = adapter_metadata or {}
    for key in ("requested_num_frames", "wan_num_frames", "target_num_frames"):
        value = meta.get(key)
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    if request.frames and int(request.frames) > 0:
        return int(request.frames)
    return max(1, int(round(float(request.duration_seconds) * float(request.fps))))


def _is_wan_model(model_id: str) -> bool:
    lowered = model_id.strip().lower()
    return lowered.startswith("wan")


def _effective_min_frame_diff(
    *,
    config: Any,
    model_id: str,
    request: ClipRequest,
) -> float:
    wan_profile = (
        request.params.get("wan_profile")
        or request.params.get("profile")
        or request.global_params.get("wan_profile")
        or request.global_params.get("profile")
    )
    if hasattr(config, "resolve_post_clip_min_frame_diff"):
        return float(
            config.resolve_post_clip_min_frame_diff(
                model_id=model_id,
                wan_profile=str(wan_profile) if wan_profile is not None else None,
            )
        )
    explicit = getattr(config, "post_clip_min_frame_diff", None)
    if explicit is not None:
        return max(0.0, float(explicit))
    return 0.0


def _wan_retry_reason(exc: Exception) -> str | None:
    text = str(exc).lower()
    if "static output" in text or ("frame_diff" in text and "threshold" in text):
        return "static_output"
    if "ffprobe nb_frames mismatch" in text or "frame count mismatch" in text:
        return "encoded_frame_mismatch"
    if "clip output too small" in text or "mp4 too small" in text:
        return "mp4_too_small"
    return None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _next_wan_retry_params(
    *,
    current_params: dict[str, Any],
    global_params: dict[str, Any],
    current_steps: int,
) -> tuple[dict[str, Any], int, dict[str, Any], dict[str, Any]]:
    params = dict(current_params)
    current_motion = _as_float(
        params.get("motion_strength", global_params.get("motion_strength", 0.70)),
        0.70,
    )
    current_guidance = _as_float(
        params.get(
            "cfg",
            params.get(
                "guidance_scale",
                global_params.get("cfg", global_params.get("guidance_scale", 5.0)),
            ),
        ),
        5.0,
    )

    next_motion = min(0.85, current_motion + 0.10)
    next_guidance = max(4.5, current_guidance - 0.3)
    next_steps = min(72, int(current_steps) + 6)

    params["motion_strength"] = round(next_motion, 3)
    params["cfg"] = round(next_guidance, 3)
    params["guidance_scale"] = round(next_guidance, 3)
    before = {
        "motion_strength": current_motion,
        "guidance_scale": current_guidance,
        "steps": int(current_steps),
    }
    after = {
        "motion_strength": round(next_motion, 3),
        "guidance_scale": round(next_guidance, 3),
        "steps": int(next_steps),
    }
    return params, next_steps, before, after


def _resolve_cinematic_constraints_enabled(
    *,
    wan_model: bool,
    job: JobSpec,
    shot: Any,
) -> bool:
    if not wan_model:
        return False
    shot_override = shot.params.get("cinematic_constraints")
    if shot_override is not None:
        text = str(shot_override).strip().lower()
        return text in {"1", "true", "yes", "on"}

    constant_override = job.constants.get("cinematic_constraints_enabled")
    if constant_override is not None:
        text = str(constant_override).strip().lower()
        return text in {"1", "true", "yes", "on"}

    wan_profile = str(
        shot.params.get("wan_profile")
        or job.global_params.get("wan_profile")
        or job.global_params.get("profile")
        or ""
    ).strip().lower()
    if wan_profile == "smoke":
        return False
    return True


def _resolve_motion_block_enabled(
    *,
    wan_model: bool,
    job: JobSpec,
    shot: Any,
) -> bool:
    if not wan_model:
        return False
    shot_override = shot.params.get("motion_block")
    if shot_override is not None:
        text = str(shot_override).strip().lower()
        return text in {"1", "true", "yes", "on"}

    global_override = job.global_params.get("motion_block")
    if global_override is not None:
        text = str(global_override).strip().lower()
        return text in {"1", "true", "yes", "on"}

    constant_override = job.constants.get("motion_block_enabled")
    if constant_override is not None:
        text = str(constant_override).strip().lower()
        return text in {"1", "true", "yes", "on"}

    wan_profile = str(
        shot.params.get("wan_profile")
        or job.global_params.get("wan_profile")
        or job.global_params.get("profile")
        or ""
    ).strip().lower()
    if wan_profile == "smoke":
        return False
    return True


def _resolve_motion_block_text(
    *,
    enabled: bool,
    shot: Any,
) -> str | None:
    if not enabled:
        return None
    if "MOTION:" in str(shot.prompt):
        return None
    provided = shot.params.get("motion_block_text")
    if isinstance(provided, str) and provided.strip():
        return provided.strip()

    subject = str(
        shot.params.get("motion_subject")
        or "subtle natural body movement; breathing; small head turns; micro-expressions"
    ).strip()
    environment = str(
        shot.params.get("motion_environment")
        or "rain streaks and droplets; neon reflections shimmering on wet asphalt; steam drifting; light wind"
    ).strip()
    camera = str(
        shot.params.get("motion_camera")
        or "slow subtle dolly-in only; no rapid zoom; no sudden framing changes; subject centered"
    ).strip()
    notes = str(shot.params.get("motion_notes") or "").strip()
    lines = [
        "MOTION:",
        f"- Subject: {subject}",
        f"- Environment: {environment}",
        f"- Camera: {camera}",
    ]
    if notes:
        lines.append(f"- Notes: {notes}")
    return "\n".join(lines)


def _empty_clip_log(
    *,
    run_id: str,
    clip_index: int,
    model_id: str,
    model_version: str,
    git_commit: str,
    started_at: str,
    prompt_full_text: str,
    seed: int | None,
    params: dict[str, Any],
    input_image_path: str | None,
    output_clip_path: str | None,
    last_frame_path: str | None,
    status: str,
    error: str | None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "clip_index": clip_index,
        "model_id": model_id,
        "model_version": model_version,
        "git_commit": git_commit,
        "started_at": started_at,
        "ended_at": started_at,
        "runtime_sec": 0.0,
        "prompt_full_text": prompt_full_text,
        "seed": seed,
        "params": params,
        "input_image_path": input_image_path,
        "input_image_sha256": None,
        "output_clip_path": output_clip_path,
        "output_clip_sha256": None,
        "last_frame_path": last_frame_path,
        "last_frame_sha256": None,
        "adapter_metadata": None,
        "status": status,
        "error": error,
    }


def _manifest_clip_summary(log_entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": log_entry["clip_index"],
        "input_image_path": log_entry["input_image_path"],
        "input_image_sha256": log_entry["input_image_sha256"],
        "output_clip_path": log_entry["output_clip_path"],
        "output_clip_sha256": log_entry["output_clip_sha256"],
        "last_frame_path": log_entry["last_frame_path"],
        "last_frame_sha256": log_entry["last_frame_sha256"],
        "runtime_sec": log_entry["runtime_sec"],
        "seed": log_entry["seed"],
        "status": log_entry["status"],
    }


def _write_skipped_logs(
    *,
    start_index: int,
    job: JobSpec,
    run_paths: Any,
    manifest: dict[str, Any],
    reason: str,
    wan_model: bool = False,
    chain_last_frame_enabled: bool = False,
) -> None:
    for clip_index in range(start_index, job.planned_clip_count):
        shot = job.shots[clip_index]
        started_at = utc_now_iso()
        prompt_full_text = compile_prompt(
            job.constants,
            shot,
            clip_index,
            chain_last_frame_enabled=chain_last_frame_enabled,
            cinematic_constraints_enabled=_resolve_cinematic_constraints_enabled(
                wan_model=wan_model,
                job=job,
                shot=shot,
            ),
            motion_block_text=_resolve_motion_block_text(
                enabled=_resolve_motion_block_enabled(
                    wan_model=wan_model,
                    job=job,
                    shot=shot,
                ),
                shot=shot,
            ),
        )
        seed = shot.seed if shot.seed is not None else clip_index
        params = _clip_params(job, shot)
        clip_output = run_paths.clips_dir / f"clip_{clip_index:03d}.mp4"
        last_frame_output = run_paths.frames_dir / f"last_frame_{clip_index:03d}.png"
        log_payload = _empty_clip_log(
            run_id=manifest["run_id"],
            clip_index=clip_index,
            model_id=manifest["model_id"],
            model_version=manifest["model_version"],
            git_commit=manifest["git_commit"],
            started_at=started_at,
            prompt_full_text=prompt_full_text,
            seed=seed,
            params=params,
            input_image_path=None,
            output_clip_path=_display_path(clip_output, run_paths.run_dir),
            last_frame_path=_display_path(last_frame_output, run_paths.run_dir),
            status="skipped",
            error=reason,
        )
        validated = ClipLogSchema.model_validate(log_payload).model_dump()
        clip_log_path = run_paths.logs_dir / f"log_{clip_index:03d}.json"
        write_clip_log(clip_log_path, validated)
        manifest["clips"].append(_manifest_clip_summary(validated))


def run_job(job_path: str | Path, overrides: dict[str, Any] | None = None) -> Path:
    job_path = Path(job_path).resolve()
    raw_job = load_structured_file(job_path)

    config_overrides: dict[str, Any] = {}
    if overrides:
        config_overrides, job_overrides = _split_overrides(overrides)
        for key, value in job_overrides.items():
            set_by_dot_path(raw_job, key, value)

    job = JobSpec.model_validate(raw_job)
    config = load_config(config_overrides)

    run_id = job.run_id or generate_run_id(job.job_name)
    output_root = job.output_root if job.output_root else config.output_root
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    run_paths = prepare_run_directories(output_root, run_id)
    progress = ProgressTracker(run_paths.run_dir)
    status_dir = progress.status_dir

    per_clip_timeout = MAX_GUARDRAIL_SECONDS
    total_timeout = planned_total_timeout_seconds(job.planned_clip_count)
    guard = TimeoutGuard(per_clip_timeout, total_timeout)

    started_at = utc_now_iso()
    run_started = time.monotonic()
    manifest = build_manifest_base(
        run_id=run_id,
        created_at=started_at,
        job_path=job_path,
        job_sha256=hash_file(job_path) or "",
        model_id=job.model.id,
        model_version=job.model.version,
        config=config,
    )
    manifest["planned_clips"] = job.planned_clip_count

    adapter = get_adapter(job.model.id, job.model.version, config)
    progress.update(None, "init", 0.0, f"Run initialized with adapter '{job.model.id}'.")
    wan_model = _is_wan_model(job.model.id)
    chain_last_frame_enabled = bool(job.global_params.get("chain_last_frame", False)) if wan_model else True

    initial_input_image: Path | None = None
    if job.input_image is not None:
        initial_input_image = job.input_image
        if not initial_input_image.is_absolute():
            initial_input_image = (job_path.parent / initial_input_image).resolve()

    clip_paths: list[Path] = []
    previous_last_frame: Path | None = None
    completed_clips = 0
    failure_exception: Exception | None = None
    aborted_timeout = False
    aborted_user = False
    abort_reason: str | None = None

    try:
        for clip_index, shot in enumerate(job.shots):
            progress_percent = (clip_index / max(1, job.planned_clip_count)) * 85.0
            try:
                _control_gate(
                    status_dir=status_dir,
                    progress=progress,
                    clip_index=clip_index,
                    percent=progress_percent,
                    context=f"before clip {clip_index}",
                )
                guard.check_total()
            except UserAbortError as exc:
                aborted_user = True
                abort_reason = str(exc)
                progress.update(clip_index, "aborted_user", 100.0, abort_reason)
                _write_skipped_logs(
                    start_index=clip_index,
                    job=job,
                    run_paths=run_paths,
                    manifest=manifest,
                    reason=abort_reason,
                    wan_model=wan_model,
                    chain_last_frame_enabled=wan_model and chain_last_frame_enabled,
                )
                break
            except TimeoutError as exc:
                aborted_timeout = True
                abort_reason = str(exc)
                progress.update(None, "aborted_timeout", 100.0, abort_reason)
                _write_skipped_logs(
                    start_index=clip_index,
                    job=job,
                    run_paths=run_paths,
                    manifest=manifest,
                    reason=abort_reason,
                    wan_model=wan_model,
                    chain_last_frame_enabled=wan_model and chain_last_frame_enabled,
                )
                break

            progress.update(
                clip_index,
                "generating_clip_start",
                progress_percent,
                f"generating clip {clip_index} start",
            )

            clip_started = guard.start_clip()
            clip_started_at = utc_now_iso()
            clip_output = run_paths.clips_dir / f"clip_{clip_index:03d}.mp4"
            clip_log_path = run_paths.logs_dir / f"log_{clip_index:03d}.json"
            if clip_index == 0:
                input_image = initial_input_image
            elif wan_model and not chain_last_frame_enabled:
                input_image = initial_input_image
            else:
                input_image = previous_last_frame
            seed = shot.seed if shot.seed is not None else clip_index
            prompt_full_text = compile_prompt(
                job.constants,
                shot,
                clip_index,
                chain_last_frame_enabled=wan_model and chain_last_frame_enabled,
                cinematic_constraints_enabled=_resolve_cinematic_constraints_enabled(
                    wan_model=wan_model,
                    job=job,
                    shot=shot,
                ),
                motion_block_text=_resolve_motion_block_text(
                    enabled=_resolve_motion_block_enabled(
                        wan_model=wan_model,
                        job=job,
                        shot=shot,
                    ),
                    shot=shot,
                ),
            )
            params = _clip_params(job, shot)
            last_frame_output = run_paths.frames_dir / f"last_frame_{clip_index:03d}.png"

            log_payload = _empty_clip_log(
                run_id=run_id,
                clip_index=clip_index,
                model_id=job.model.id,
                model_version=job.model.version,
                git_commit=manifest["git_commit"],
                started_at=clip_started_at,
                prompt_full_text=prompt_full_text,
                seed=seed,
                params=params,
                input_image_path=_display_path(input_image, run_paths.run_dir),
                output_clip_path=_display_path(clip_output, run_paths.run_dir),
                last_frame_path=_display_path(last_frame_output, run_paths.run_dir),
                status="failed",
                error=None,
            )
            retry_attempted = False
            retry_reason: str | None = None
            retry_params_before: dict[str, Any] | None = None
            retry_params_after: dict[str, Any] | None = None

            try:
                if wan_model and input_image is None:
                    raise RuntimeError(
                        "WAN clip requires an input image but none was available. "
                        "Set input_image for clip 0 and/or enable generation_defaults.chain_last_frame."
                    )
                attempt = 0
                max_attempts = 2 if wan_model else 1
                runtime_params = dict(shot.params)
                runtime_steps = int(shot.steps)
                request: ClipRequest | None = None
                result = None
                adapter_meta: dict[str, Any] = {}

                while True:
                    remaining_for_clip = int(
                        max(1, min(guard.remaining_clip_seconds(clip_started), guard.remaining_total_seconds()))
                    )
                    request = ClipRequest(
                        clip_index=clip_index,
                        prompt=prompt_full_text,
                        negative_prompt=shot.negative_prompt,
                        input_image=input_image,
                        output_video_path=clip_output,
                        duration_seconds=shot.duration_seconds,
                        fps=shot.fps,
                        frames=shot.frames or 1,
                        steps=runtime_steps,
                        seed=seed,
                        width=shot.width,
                        height=shot.height,
                        params=runtime_params,
                        global_params=job.global_params,
                        dry_run=job.dry_run,
                        max_runtime_seconds=remaining_for_clip,
                    )

                    try:
                        result = adapter.generate_clip(request)
                        guard.check_clip(clip_started, clip_index)
                        adapter_meta = dict(result.extra_metadata or {})

                        if wan_model:
                            postprocess_meta = run_optional_wan_postprocess(
                                video_path=result.output_video_path,
                                ffmpeg_bin=config.ffmpeg_bin,
                                fps=request.fps,
                                postprocess_config=job.global_params.get("postprocess"),
                                timeout_seconds=remaining_for_clip,
                            )
                            adapter_meta["postprocess"] = postprocess_meta

                        validation_enabled = bool(config.post_clip_validation_enabled)
                        if validation_enabled:
                            remaining_for_validation = int(
                                max(1, min(guard.remaining_clip_seconds(clip_started), guard.remaining_total_seconds()))
                            )
                            min_frame_diff_effective = _effective_min_frame_diff(
                                config=config,
                                model_id=job.model.id,
                                request=request,
                            )
                            min_size_bytes_effective = max(0, int(config.post_clip_min_size_bytes))
                            validation_report = validate_clip_output(
                                ffmpeg_bin=config.ffmpeg_bin,
                                ffprobe_bin=config.ffprobe_bin,
                                video_path=result.output_video_path,
                                requested_num_frames=_expected_num_frames(request, adapter_meta),
                                min_size_bytes=min_size_bytes_effective,
                                min_frame_diff=min_frame_diff_effective,
                                timeout_seconds=remaining_for_validation,
                            )
                            validation_report["min_frame_diff_effective"] = min_frame_diff_effective
                            validation_report["min_size_bytes_effective"] = min_size_bytes_effective
                            adapter_meta["post_clip_validation"] = validation_report
                        break
                    except Exception as exc:
                        reason = _wan_retry_reason(exc)
                        if not (wan_model and attempt + 1 < max_attempts and reason is not None):
                            raise
                        runtime_params, runtime_steps, retry_params_before, retry_params_after = _next_wan_retry_params(
                            current_params=runtime_params,
                            global_params=job.global_params,
                            current_steps=runtime_steps,
                        )
                        retry_attempted = True
                        retry_reason = reason
                        attempt += 1
                        progress.update(
                            clip_index,
                            "retrying_clip",
                            progress_percent,
                            f"retrying clip {clip_index} ({reason}) with higher motion/lower guidance",
                        )

                if request is None or result is None:
                    raise RuntimeError("Clip generation did not produce a request/result pair.")
                adapter_meta["retry_attempted"] = bool(retry_attempted)
                adapter_meta["retry_reason"] = retry_reason
                adapter_meta["params_before_retry"] = retry_params_before
                adapter_meta["params_after_retry"] = retry_params_after
                adapter_meta["retry_success"] = bool(retry_attempted)
                adapter_meta["adaptive_retry"] = {
                    "attempted": bool(retry_attempted),
                    "retry_count": int(attempt),
                    "reason": retry_reason,
                    "params_before_retry": retry_params_before,
                    "params_after_retry": retry_params_after,
                    "retry_success": bool(retry_attempted),
                }

                resolved_cfg = request.params.get(
                    "cfg",
                    request.params.get(
                        "guidance_scale",
                        request.global_params.get("cfg", request.global_params.get("guidance_scale")),
                    ),
                )
                resolved_motion = request.params.get("motion_strength", request.global_params.get("motion_strength"))
                adapter_meta["resolved_runtime_params"] = {
                    "steps": request.steps,
                    "fps": request.fps,
                    "frames": request.frames,
                    "guidance_scale": resolved_cfg,
                    "motion_strength": resolved_motion,
                    "motion_block": request.params.get("motion_block", request.global_params.get("motion_block")),
                    "motion_subject": request.params.get("motion_subject"),
                    "motion_environment": request.params.get("motion_environment"),
                    "motion_camera": request.params.get("motion_camera"),
                    "motion_notes": request.params.get("motion_notes"),
                    "motion_block_text": request.params.get("motion_block_text"),
                    "cinematic_constraints": request.params.get("cinematic_constraints"),
                    "export_quality": request.params.get(
                        "export_quality",
                        request.global_params.get("export_quality"),
                    ),
                }
                if resolved_cfg is not None:
                    log_payload["params"]["cfg"] = resolved_cfg
                _control_gate(
                    status_dir=status_dir,
                    progress=progress,
                    clip_index=clip_index,
                    percent=progress_percent,
                    context=f"after clip {clip_index} generation",
                )

                remaining_for_extract = int(
                    max(1, min(guard.remaining_clip_seconds(clip_started), guard.remaining_total_seconds()))
                )
                progress.update(
                    clip_index,
                    "extracting_last_frame",
                    progress_percent,
                    f"extracting last frame {clip_index}",
                )
                extract_last_frame(
                    result.output_video_path,
                    last_frame_output,
                    ffmpeg_bin=config.ffmpeg_bin,
                    timeout_seconds=remaining_for_extract,
                )
                if wan_model and chain_last_frame_enabled and last_frame_output.exists():
                    continuity_dir = ensure_dir(run_paths.run_dir / "debug" / "continuity")
                    continuity_frame_path = continuity_dir / f"last_frame_clip_{clip_index:03d}.png"
                    shutil.copy2(last_frame_output, continuity_frame_path)
                    adapter_meta["continuity_last_frame_artifact_path"] = _display_path(
                        continuity_frame_path,
                        run_paths.run_dir,
                    )
                clip_runtime_wall = guard.check_clip(clip_started, clip_index)

                log_payload["ended_at"] = utc_now_iso()
                log_payload["runtime_sec"] = clip_runtime_wall
                log_payload["input_image_sha256"] = hash_file(input_image) if input_image else None
                log_payload["output_clip_sha256"] = hash_file(result.output_video_path)
                log_payload["last_frame_sha256"] = hash_file(last_frame_output)
                adapter_meta["chain_last_frame_enabled"] = bool(wan_model and chain_last_frame_enabled)
                adapter_meta["init_image_used_path"] = _display_path(input_image, run_paths.run_dir)
                adapter_meta["previous_clip_last_frame_path"] = (
                    _display_path(previous_last_frame, run_paths.run_dir) if clip_index > 0 else None
                )
                prompt_passed_to_pipe = adapter_meta.get("prompt_passed_to_pipe")
                if isinstance(prompt_passed_to_pipe, str) and prompt_passed_to_pipe.strip():
                    log_payload["prompt_full_text"] = prompt_passed_to_pipe
                prompt_debug_text = log_payload["prompt_full_text"]
                if isinstance(prompt_debug_text, str) and prompt_debug_text.strip():
                    prompt_debug_dir = ensure_dir(run_paths.run_dir / "debug" / "prompts")
                    prompt_debug_path = prompt_debug_dir / f"prompt_{clip_index:03d}.txt"
                    prompt_debug_path.write_text(prompt_debug_text, encoding="utf-8")
                    adapter_meta["prompt_debug_artifact_path"] = _display_path(prompt_debug_path, run_paths.run_dir)
                    resolved_params = adapter_meta.get("resolved_runtime_params")
                    if isinstance(resolved_params, dict):
                        params_debug_path = prompt_debug_dir / f"prompt_{clip_index:03d}_params.json"
                        params_debug_path.write_text(
                            json.dumps(resolved_params, indent=2, sort_keys=True),
                            encoding="utf-8",
                        )
                        adapter_meta["prompt_params_artifact_path"] = _display_path(
                            params_debug_path,
                            run_paths.run_dir,
                        )

                log_payload["adapter_metadata"] = adapter_meta if adapter_meta else None
                log_payload["status"] = "success"
                log_payload["error"] = None
                if wan_model:
                    model_runtime = manifest.get("model_runtime")
                    if not isinstance(model_runtime, dict):
                        model_runtime = {}
                    repo_id_used = adapter_meta.get("repo_id_used") or adapter_meta.get("repo_id")
                    if repo_id_used:
                        model_runtime["repo_id_used"] = str(repo_id_used)
                    manifest["model_runtime"] = model_runtime
                if adapter_meta.get("wan_native_720p_applied"):
                    requested_res = adapter_meta.get("requested_resolution")
                    internal_res = adapter_meta.get("internal_resolution")
                    model_runtime = manifest.get("model_runtime")
                    if not isinstance(model_runtime, dict):
                        model_runtime = {}
                    model_runtime["wan_native_720p"] = {
                        "applied": True,
                        "requested_resolution": requested_res,
                        "internal_resolution": internal_res,
                        "guidance": "Wan2.2 TI2V 720P should use 1280x704.",
                        "source": adapter_meta.get("wan_native_720p_guidance_url"),
                    }
                    manifest["model_runtime"] = model_runtime

                validated_log = ClipLogSchema.model_validate(log_payload).model_dump()
                write_clip_log(clip_log_path, validated_log)
                manifest["clips"].append(_manifest_clip_summary(validated_log))

                previous_last_frame = last_frame_output
                clip_paths.append(result.output_video_path)
                completed_clips += 1
                progress_percent = ((clip_index + 1) / max(1, job.planned_clip_count)) * 85.0
                progress.update(
                    clip_index,
                    "generating_clip_done",
                    progress_percent,
                    f"generating clip {clip_index} done",
                )
            except UserAbortError as exc:
                log_payload["ended_at"] = utc_now_iso()
                log_payload["runtime_sec"] = max(0.0, time.monotonic() - clip_started)
                log_payload["input_image_sha256"] = hash_file(input_image) if input_image else None
                log_payload["output_clip_sha256"] = hash_file(clip_output)
                log_payload["last_frame_sha256"] = hash_file(last_frame_output)
                log_payload["status"] = "failed"
                log_payload["error"] = str(exc)
                log_payload["adapter_metadata"] = {
                    "retry_attempted": bool(retry_attempted),
                    "retry_reason": retry_reason,
                    "params_before_retry": retry_params_before,
                    "params_after_retry": retry_params_after,
                    "retry_success": False,
                }
                validated_log = ClipLogSchema.model_validate(log_payload).model_dump()
                write_clip_log(clip_log_path, validated_log)
                manifest["clips"].append(_manifest_clip_summary(validated_log))

                aborted_user = True
                abort_reason = str(exc)
                progress.update(clip_index, "aborted_user", 100.0, abort_reason)
                _write_skipped_logs(
                    start_index=clip_index + 1,
                    job=job,
                    run_paths=run_paths,
                    manifest=manifest,
                    reason=abort_reason,
                    wan_model=wan_model,
                    chain_last_frame_enabled=wan_model and chain_last_frame_enabled,
                )
                break
            except TimeoutError as exc:
                log_payload["ended_at"] = utc_now_iso()
                log_payload["runtime_sec"] = max(0.0, time.monotonic() - clip_started)
                log_payload["input_image_sha256"] = hash_file(input_image) if input_image else None
                log_payload["output_clip_sha256"] = hash_file(clip_output)
                log_payload["last_frame_sha256"] = hash_file(last_frame_output)
                log_payload["status"] = "failed"
                log_payload["error"] = str(exc)
                log_payload["adapter_metadata"] = {
                    "retry_attempted": bool(retry_attempted),
                    "retry_reason": retry_reason,
                    "params_before_retry": retry_params_before,
                    "params_after_retry": retry_params_after,
                    "retry_success": False,
                }
                validated_log = ClipLogSchema.model_validate(log_payload).model_dump()
                write_clip_log(clip_log_path, validated_log)
                manifest["clips"].append(_manifest_clip_summary(validated_log))

                aborted_timeout = True
                abort_reason = str(exc)
                progress.update(clip_index, "aborted_timeout", 100.0, abort_reason)
                _write_skipped_logs(
                    start_index=clip_index + 1,
                    job=job,
                    run_paths=run_paths,
                    manifest=manifest,
                    reason=abort_reason,
                    wan_model=wan_model,
                    chain_last_frame_enabled=wan_model and chain_last_frame_enabled,
                )
                break
            except Exception as exc:
                log_payload["ended_at"] = utc_now_iso()
                log_payload["runtime_sec"] = max(0.0, time.monotonic() - clip_started)
                log_payload["input_image_sha256"] = hash_file(input_image) if input_image else None
                log_payload["output_clip_sha256"] = hash_file(clip_output)
                log_payload["last_frame_sha256"] = hash_file(last_frame_output)
                log_payload["status"] = "failed"
                log_payload["error"] = str(exc)
                log_payload["adapter_metadata"] = {
                    "retry_attempted": bool(retry_attempted),
                    "retry_reason": retry_reason,
                    "params_before_retry": retry_params_before,
                    "params_after_retry": retry_params_after,
                    "retry_success": False,
                }
                validated_log = ClipLogSchema.model_validate(log_payload).model_dump()
                write_clip_log(clip_log_path, validated_log)
                manifest["clips"].append(_manifest_clip_summary(validated_log))
                failure_exception = exc
                progress.update(clip_index, "failed", 100.0, str(exc))
                break

        final_stitched_path: str | None = None
        bundle_path: str | None = None

        if not aborted_timeout and not aborted_user and failure_exception is None:
            try:
                _control_gate(
                    status_dir=status_dir,
                    progress=progress,
                    clip_index=None,
                    percent=90.0,
                    context="before stitching",
                )
                progress.update(None, "stitching_start", 90.0, "stitching start")
                guard.check_total()
                final_video_path = run_paths.run_dir / "final_stitched.mp4"
                stitch_timeout = int(max(1, guard.remaining_total_seconds()))
                stitch_clips_deterministic(
                    clip_paths,
                    final_video_path,
                    ffmpeg_bin=config.ffmpeg_bin,
                    timeout_seconds=stitch_timeout,
                )
                final_stitched_path = _display_path(final_video_path, run_paths.run_dir)
                progress.update(None, "stitching_done", 95.0, "stitching done")

                _control_gate(
                    status_dir=status_dir,
                    progress=progress,
                    clip_index=None,
                    percent=96.0,
                    context="before bundling",
                )
                archive_path = run_paths.run_dir / f"{run_paths.run_dir.name}_bundle.zip"
                bundle_path = _display_path(archive_path, run_paths.run_dir)
                progress.update(None, "bundling_start", 96.0, "bundling start")
            except UserAbortError as exc:
                aborted_user = True
                abort_reason = str(exc)

        manifest["completed_clips"] = completed_clips
        manifest["total_runtime_sec"] = max(0.0, time.monotonic() - run_started)
        manifest["outputs"] = {
            "final_stitched_path": final_stitched_path,
            "bundle_path": bundle_path,
        }
        if aborted_user:
            manifest["status"] = "aborted_user"
            manifest["error"] = abort_reason
            progress.update(None, "complete", 100.0, "Run aborted by user.")
        elif aborted_timeout:
            manifest["status"] = "aborted_timeout"
            manifest["error"] = abort_reason
            progress.update(None, "complete", 100.0, "Run aborted due to timeout.")
        elif failure_exception is not None:
            manifest["status"] = "failed"
            manifest["error"] = str(failure_exception)
            progress.update(None, "failed", 100.0, str(failure_exception))
        else:
            manifest["status"] = "completed"
            manifest["error"] = None

        write_manifest(run_paths.run_dir, manifest)
        if manifest["status"] == "completed":
            create_run_bundle(
                run_paths.run_dir,
                archive_path=run_paths.run_dir / f"{run_paths.run_dir.name}_bundle.zip",
                fmt="zip",
            )
            progress.update(None, "bundling_done", 99.0, "bundling done")
        if manifest["status"] == "completed":
            progress.update(
                None,
                "bundle_ready",
                100.0,
                "bundle ready",
                extra={"bundle_path": manifest["outputs"]["bundle_path"]},
            )

        if failure_exception is not None:
            raise failure_exception
        return run_paths.run_dir
    except Exception as exc:
        manifest["completed_clips"] = completed_clips
        manifest["total_runtime_sec"] = max(0.0, time.monotonic() - run_started)
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        write_manifest(run_paths.run_dir, manifest)
        progress.update(None, "failed", 100.0, str(exc))
        raise


def cli_main() -> None:
    parser = argparse.ArgumentParser(description="Run a video generation pipeline job.")
    parser.add_argument("job", type=Path, help="Job spec file (.yaml/.yml/.json).")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override field with KEY=VALUE. Use config.KEY for environment config fields.",
    )
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    run_dir = run_job(args.job, overrides=overrides)
    print(f"Run finished: {run_dir}")


if __name__ == "__main__":
    cli_main()
