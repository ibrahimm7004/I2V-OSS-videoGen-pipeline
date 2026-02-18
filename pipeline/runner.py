from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from models import get_adapter
from models.base import ClipRequest

from .artifacts import build_manifest_base, hash_file, prepare_run_directories, write_clip_log, write_manifest
from .bundling import create_run_bundle
from .config import MAX_GUARDRAIL_SECONDS, load_config
from .frames import extract_last_frame
from .job_schema import JobSpec
from .output_schema import ClipLogSchema
from .prompting import compile_prompt
from .progress import ProgressTracker
from .stitching import stitch_clips_deterministic
from .timeouts import TimeoutGuard, planned_total_timeout_seconds
from .utils import generate_run_id, load_structured_file, parse_overrides, set_by_dot_path, utc_now_iso


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
) -> None:
    for clip_index in range(start_index, job.planned_clip_count):
        shot = job.shots[clip_index]
        started_at = utc_now_iso()
        prompt_full_text = compile_prompt(job.constants, shot, clip_index)
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
            input_image = initial_input_image if clip_index == 0 else previous_last_frame
            seed = shot.seed if shot.seed is not None else clip_index
            prompt_full_text = compile_prompt(job.constants, shot, clip_index)
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

            try:
                remaining_for_clip = int(max(1, min(guard.remaining_clip_seconds(clip_started), guard.remaining_total_seconds())))
                request = ClipRequest(
                    clip_index=clip_index,
                    prompt=prompt_full_text,
                    negative_prompt=shot.negative_prompt,
                    input_image=input_image,
                    output_video_path=clip_output,
                    duration_seconds=shot.duration_seconds,
                    fps=shot.fps,
                    frames=shot.frames or 1,
                    steps=shot.steps,
                    seed=seed,
                    width=shot.width,
                    height=shot.height,
                    params=shot.params,
                    global_params=job.global_params,
                    dry_run=job.dry_run,
                    max_runtime_seconds=remaining_for_clip,
                )
                result = adapter.generate_clip(request)
                guard.check_clip(clip_started, clip_index)
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
                clip_runtime_wall = guard.check_clip(clip_started, clip_index)

                log_payload["ended_at"] = utc_now_iso()
                log_payload["runtime_sec"] = clip_runtime_wall
                log_payload["input_image_sha256"] = hash_file(input_image) if input_image else None
                log_payload["output_clip_sha256"] = hash_file(result.output_video_path)
                log_payload["last_frame_sha256"] = hash_file(last_frame_output)
                log_payload["status"] = "success"
                log_payload["error"] = None

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
