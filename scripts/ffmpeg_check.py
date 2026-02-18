from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ffmpeg_utils import assert_ffmpeg_working, run_subprocess
from pipeline.runner import run_job


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _run_ffprobe(ffprobe_path: str, video_path: Path, timeout_seconds: int = 20) -> dict[str, Any]:
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
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid ffprobe JSON for {video_path}: {exc}") from exc
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video streams reported by ffprobe: {video_path}")
    stream0 = streams[0]
    duration = (payload.get("format") or {}).get("duration")
    return {
        "codec_name": stream0.get("codec_name"),
        "width": stream0.get("width"),
        "height": stream0.get("height"),
        "avg_frame_rate": stream0.get("avg_frame_rate"),
        "duration": duration,
    }


def run_ffmpeg_check(run_dir: Path | None = None, job: Path | None = None) -> dict[str, Any]:
    info = assert_ffmpeg_working()

    resolved_run_dir: Path
    if run_dir is not None:
        resolved_run_dir = run_dir.resolve()
    else:
        job_path = (job or Path("jobs/example_mock.yaml")).resolve()
        run_id = "ffmpeg-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        resolved_run_dir = run_job(
            job_path,
            overrides={
                "run_id": run_id,
                "model.id": "mock",
                "dry_run": True,
            },
        )

    clip0 = resolved_run_dir / "clips" / "clip_000.mp4"
    stitched = resolved_run_dir / "final_stitched.mp4"
    _assert(clip0.exists(), f"Missing clip file: {clip0}")
    _assert(stitched.exists(), f"Missing stitched file: {stitched}")

    clip_info = _run_ffprobe(info["ffprobe_path"], clip0)
    stitched_info = _run_ffprobe(info["ffprobe_path"], stitched)

    print("FFmpeg check report")
    print(f"  ffmpeg:  {info['ffmpeg_path']}")
    print(f"  ffprobe: {info['ffprobe_path']}")
    print(f"  ffmpeg_version:  {info['ffmpeg_version']}")
    print(f"  ffprobe_version: {info['ffprobe_version']}")
    print(f"  run_dir: {resolved_run_dir}")
    print(
        "  clip_000.mp4:"
        f" codec={clip_info['codec_name']}, size={clip_info['width']}x{clip_info['height']},"
        f" duration={clip_info['duration']}, fps={clip_info['avg_frame_rate']}"
    )
    print(
        "  final_stitched.mp4:"
        f" codec={stitched_info['codec_name']}, size={stitched_info['width']}x{stitched_info['height']},"
        f" duration={stitched_info['duration']}, fps={stitched_info['avg_frame_rate']}"
    )

    return {
        "tooling": info,
        "run_dir": str(resolved_run_dir),
        "clip_000": clip_info,
        "final_stitched": stitched_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ffmpeg/ffprobe binaries and probe generated outputs.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Reuse an existing run dir instead of running a new mock job.")
    parser.add_argument("--job", type=Path, default=Path("jobs/example_mock.yaml"), help="Job file for mock run.")
    args = parser.parse_args()

    try:
        run_ffmpeg_check(run_dir=args.run_dir, job=args.job)
    except Exception as exc:
        print(f"FFmpeg check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

