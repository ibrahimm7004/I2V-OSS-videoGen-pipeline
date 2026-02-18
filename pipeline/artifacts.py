from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .env_probe import probe_environment
from .output_schema import ClipLogSchema, EnvironmentSchema, HFCacheSchema, ManifestSchema
from .utils import ensure_dir, get_git_commit, write_json


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    clips_dir: Path
    frames_dir: Path
    logs_dir: Path


def prepare_run_directories(output_root: Path, run_id: str) -> RunPaths:
    run_dir = ensure_dir(output_root / run_id)
    return RunPaths(
        run_dir=run_dir,
        clips_dir=ensure_dir(run_dir / "clips"),
        frames_dir=ensure_dir(run_dir / "frames"),
        logs_dir=ensure_dir(run_dir / "logs"),
    )


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _to_display_path(path: Path, base: Path | None = None) -> str:
    resolved = path.resolve()
    if base is not None:
        try:
            return resolved.relative_to(base.resolve()).as_posix()
        except ValueError:
            pass
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def environment_summary() -> EnvironmentSchema:
    return EnvironmentSchema.model_validate(probe_environment())


def hf_cache_summary(config: Any) -> HFCacheSchema:
    return HFCacheSchema(HF_HOME=config.hf_home, HF_HUB_CACHE=config.hf_hub_cache)


def build_manifest_base(
    *,
    run_id: str,
    created_at: str,
    job_path: Path,
    job_sha256: str,
    model_id: str,
    model_version: str,
    config: Any,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at": created_at,
        "job_path": _to_display_path(job_path, base=Path.cwd()),
        "job_sha256": job_sha256,
        "model_id": model_id,
        "model_version": model_version,
        "git_commit": get_git_commit(),
        "environment": environment_summary().model_dump(),
        "hf_cache": hf_cache_summary(config).model_dump(),
        "planned_clips": 0,
        "completed_clips": 0,
        "total_runtime_sec": 0.0,
        "outputs": {
            "final_stitched_path": None,
            "bundle_path": None,
        },
        "stitch_settings": {
            "overlap_sec": 0.0,
            "transition": "concat",
            "encoder_settings": {
                "video_codec": "TODO",
                "audio_codec": "TODO",
                "preset": "TODO",
            },
        },
        "clips": [],
        "status": "failed",
        "error": None,
    }


def write_clip_log(path: Path, payload: dict[str, Any]) -> None:
    validated = ClipLogSchema.model_validate(payload)
    write_json(path, validated.model_dump())


def write_manifest(run_dir: Path, payload: dict[str, Any]) -> Path:
    validated = ManifestSchema.model_validate(payload)
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, validated.model_dump())
    return manifest_path
