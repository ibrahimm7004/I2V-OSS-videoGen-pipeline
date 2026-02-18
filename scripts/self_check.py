from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.output_schema import ClipLogSchema, ManifestSchema
from pipeline.runner import run_job
try:
    from scripts.ffmpeg_check import run_ffmpeg_check
except ModuleNotFoundError:
    from ffmpeg_check import run_ffmpeg_check
try:
    from scripts.validate_run import validate_run_directory
except ModuleNotFoundError:
    from validate_run import validate_run_directory


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve_artifact(run_dir: Path, path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return run_dir / path


def _resolve_job_arg(job_arg: Path | None) -> Path:
    if job_arg is not None:
        return job_arg
    candidates = [Path("jobs/example_mock.yaml"), Path("jobs/idea03_cogvideox.yaml")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    fallback = sorted(Path("jobs").glob("*.yaml"))
    if fallback:
        return fallback[0]
    raise RuntimeError("No job YAML found under jobs/. Provide --job explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-pytest scaffold self-check with schema validation.")
    parser.add_argument("--job", type=Path, default=None)
    parser.add_argument("--with-ffmpeg", action="store_true", help="Also run ffmpeg/ffprobe validation.")
    parser.add_argument("--validate-run", action="store_true", help="Run full run validation checks at the end.")
    args = parser.parse_args()

    job_path = _resolve_job_arg(args.job)
    run_id = "self-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_job(
        job_path,
        overrides={
            "run_id": run_id,
            "model.id": "mock",
            "dry_run": True,
        },
    )

    manifest_path = run_dir / "manifest.json"
    _assert(manifest_path.exists(), f"Missing manifest: {manifest_path}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ManifestSchema.model_validate(manifest_data)
    _assert(manifest.status == "completed", f"Expected completed status, got: {manifest.status}")
    _assert(manifest.run_id == run_id, "Manifest run_id does not match self-check run_id.")
    _assert(manifest.completed_clips == manifest.planned_clips, "Completed clips does not match planned clips.")

    final_video = _resolve_artifact(run_dir, manifest.outputs.final_stitched_path)
    bundle_path = _resolve_artifact(run_dir, manifest.outputs.bundle_path)
    _assert(final_video is not None and final_video.exists(), "Missing final stitched video.")
    _assert(bundle_path is not None and bundle_path.exists(), "Missing run bundle archive.")

    for clip_index in range(manifest.planned_clips):
        clip_path = run_dir / "clips" / f"clip_{clip_index:03d}.mp4"
        frame_path = run_dir / "frames" / f"last_frame_{clip_index:03d}.png"
        log_path = run_dir / "logs" / f"log_{clip_index:03d}.json"
        _assert(clip_path.exists(), f"Missing clip file: {clip_path}")
        _assert(frame_path.exists(), f"Missing last-frame file: {frame_path}")
        _assert(log_path.exists(), f"Missing clip log: {log_path}")

        log_data = json.loads(log_path.read_text(encoding="utf-8"))
        log_entry = ClipLogSchema.model_validate(log_data)
        _assert(log_entry.status == "success", f"Clip {clip_index} did not succeed.")
        _assert(bool(log_entry.output_clip_sha256), f"Clip {clip_index} missing output hash.")
        _assert(bool(log_entry.last_frame_sha256), f"Clip {clip_index} missing frame hash.")

    if args.with_ffmpeg:
        run_ffmpeg_check(run_dir=run_dir, job=job_path)
    if args.validate_run:
        validate_run_directory(run_dir)

    print(f"Self-check passed. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
