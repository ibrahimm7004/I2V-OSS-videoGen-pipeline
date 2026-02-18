from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ffmpeg_utils import get_ffprobe_path, run_subprocess
from pipeline.output_schema import ClipLogSchema, ManifestSchema


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve_run_path(run_dir: Path, path_value: str | None) -> Path | None:
    if path_value is None:
        return None
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return run_dir / candidate


def _ffprobe_video(ffprobe_path: str, video_path: Path, timeout_seconds: int = 20) -> dict[str, Any]:
    command = [
        ffprobe_path,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = run_subprocess(command, timeout_seconds=timeout_seconds)
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    _assert(bool(streams), f"No video stream found in {video_path}")
    duration_text = (data.get("format") or {}).get("duration")
    duration = float(duration_text or 0.0)
    _assert(duration > 0.0, f"Non-positive duration for {video_path}: {duration}")
    stream = streams[0]
    return {
        "codec_name": stream.get("codec_name"),
        "width": stream.get("width"),
        "height": stream.get("height"),
        "duration": duration,
    }


def _bundle_members(bundle_path: Path) -> set[str]:
    if bundle_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(bundle_path, "r") as zf:
            return set(zf.namelist())
    if bundle_path.name.lower().endswith(".tar.gz"):
        with tarfile.open(bundle_path, "r:gz") as tf:
            return set(tf.getnames())
    raise RuntimeError(f"Unsupported bundle format: {bundle_path}")


def validate_run_directory(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    status_json = run_dir / "status" / "status.json"
    progress_log = run_dir / "status" / "progress.log"

    _assert(manifest_path.exists(), f"Missing manifest: {manifest_path}")
    _assert(status_json.exists(), f"Missing status file: {status_json}")
    _assert(progress_log.exists(), f"Missing progress log: {progress_log}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = ManifestSchema.model_validate(manifest_data)
    _assert(manifest.status == "completed", f"Run is not completed: {manifest.status}")

    final_video = _resolve_run_path(run_dir, manifest.outputs.final_stitched_path)
    bundle_path = _resolve_run_path(run_dir, manifest.outputs.bundle_path)
    _assert(final_video is not None and final_video.exists(), "Missing final stitched video.")
    _assert(bundle_path is not None and bundle_path.exists(), "Missing bundle archive.")

    ffprobe_path = get_ffprobe_path()
    clip_count = manifest.planned_clips
    clip_probe_summaries: list[dict[str, Any]] = []

    for idx in range(clip_count):
        clip_path = run_dir / "clips" / f"clip_{idx:03d}.mp4"
        frame_path = run_dir / "frames" / f"last_frame_{idx:03d}.png"
        log_path = run_dir / "logs" / f"log_{idx:03d}.json"
        _assert(clip_path.exists(), f"Missing clip file: {clip_path}")
        _assert(frame_path.exists(), f"Missing frame file: {frame_path}")
        _assert(log_path.exists(), f"Missing clip log: {log_path}")

        log_data = json.loads(log_path.read_text(encoding="utf-8"))
        ClipLogSchema.model_validate(log_data)
        clip_probe_summaries.append(_ffprobe_video(ffprobe_path, clip_path))

    final_probe = _ffprobe_video(ffprobe_path, final_video)

    members = _bundle_members(bundle_path)
    required_members = {"manifest.json", "final_stitched.mp4"}
    required_members.update({f"clips/clip_{idx:03d}.mp4" for idx in range(clip_count)})
    required_members.update({f"frames/last_frame_{idx:03d}.png" for idx in range(clip_count)})
    required_members.update({f"logs/log_{idx:03d}.json" for idx in range(clip_count)})
    missing_members = sorted(item for item in required_members if item not in members)
    _assert(not missing_members, f"Bundle missing required members: {missing_members}")

    return {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "bundle": str(bundle_path),
        "clips_checked": clip_count,
        "final_video_probe": final_probe,
        "clip_probes": clip_probe_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate run artifacts, ffprobe readability, and schema integrity.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        summary = validate_run_directory(args.run_dir)
    except Exception as exc:
        print(f"Run validation failed: {exc}", file=sys.stderr)
        return 1

    print("Run validation passed")
    print(f"  run_dir: {summary['run_dir']}")
    print(f"  clips_checked: {summary['clips_checked']}")
    print(f"  bundle: {summary['bundle']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

