from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path


def _run(command: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=timeout, check=False)


def _probe(video_path: Path) -> dict:
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,avg_frame_rate",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(video_path),
    ]
    result = _run(cmd, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    return json.loads(result.stdout or "{}")


def _extract_png(video_path: Path, png_path: Path) -> int:
    ffmpeg = get_ffmpeg_path()
    cmd = [ffmpeg, "-y", "-hide_banner", "-v", "error", "-i", str(video_path), "-frames:v", "1", str(png_path)]
    result = _run(cmd, timeout=20)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extract failed: {result.stderr.strip()}")
    if not png_path.exists():
        raise RuntimeError(f"ffmpeg did not create {png_path}")
    return int(png_path.stat().st_size)


def _run_smoke_if_needed(mp4_path: Path, force: bool) -> None:
    if mp4_path.exists() and not force:
        return
    cmd = [sys.executable, "scripts/smoke_adapter_cogvx15.py"]
    result = _run(cmd, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(
            "smoke_adapter_cogvx15 failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug Cog smoke output/frame statistics.")
    parser.add_argument("--mp4", type=Path, default=Path("outputs/_adapter_smoke/cogvx15/clip_000.mp4"))
    parser.add_argument("--run-smoke", action="store_true", help="Force running smoke before collecting stats.")
    args = parser.parse_args()

    mp4_path = args.mp4
    if not mp4_path.is_absolute():
        mp4_path = (REPO_ROOT / mp4_path).resolve()

    _run_smoke_if_needed(mp4_path, force=args.run_smoke)
    if not mp4_path.exists():
        raise SystemExit(f"MP4 not found: {mp4_path}")

    report_path = mp4_path.parent / "smoke_report.json"
    extra = {}
    if report_path.exists():
        try:
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            extra = report_data.get("extra_metadata") or {}
        except Exception:
            extra = {}

    probe = _probe(mp4_path)
    png_path = mp4_path.parent / "debug_first_frame.png"
    png_size = _extract_png(mp4_path, png_path)

    size_bytes = int(mp4_path.stat().st_size)
    streams = probe.get("streams") or []
    stream = streams[0] if streams else {}
    frame_count = stream.get("nb_frames")
    print(f"mp4: {mp4_path}")
    print(f"mp4_size_bytes: {size_bytes}")
    print(f"ffprobe_nb_frames: {frame_count}")
    print(f"ffprobe_duration: {(probe.get('format') or {}).get('duration')}")
    print(f"png_frame: {png_path}")
    print(f"png_size_bytes: {png_size}")
    print("extra_metadata:")
    print(json.dumps(extra, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
