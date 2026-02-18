from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _discover_binary(env_var: str, binary_name: str) -> str:
    env_value = os.getenv(env_var)
    if env_value:
        return str(Path(env_value).expanduser())

    repo_root = _repo_root()
    local_windows = repo_root / "ffmpeg" / "bin" / f"{binary_name}.exe"
    local_linux = repo_root / "ffmpeg" / "bin" / binary_name

    if sys.platform.startswith("win"):
        if local_windows.exists():
            return str(local_windows)
        if local_linux.exists():
            return str(local_linux)
    else:
        if local_linux.exists():
            return str(local_linux)
        if local_windows.exists():
            return str(local_windows)
    return binary_name


def _resolved_executable_path(candidate: str) -> str:
    path_candidate = Path(candidate)
    if path_candidate.exists():
        return str(path_candidate.resolve())
    which_path = shutil.which(candidate)
    if which_path:
        return str(Path(which_path).resolve())
    return candidate


def get_ffmpeg_path() -> str:
    return _discover_binary("FFMPEG_BIN", "ffmpeg")


def get_ffprobe_path() -> str:
    return _discover_binary("FFPROBE_BIN", "ffprobe")


def run_subprocess(command: list[str], timeout_seconds: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"Command timed out after {timeout_seconds}s: {' '.join(command)}") from exc
    if result.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(f"Command failed ({result.returncode}): {joined}\n{result.stderr.strip()}")
    return result


def run_ffmpeg(ffmpeg_bin: str, args: list[str], timeout_seconds: int | None = None) -> subprocess.CompletedProcess[str]:
    return run_subprocess([ffmpeg_bin, *args], timeout_seconds=timeout_seconds)


def _version_for(binary_path: str, timeout_seconds: int = 10) -> str:
    result = run_subprocess([binary_path, "-version"], timeout_seconds=timeout_seconds)
    first_line = (result.stdout or "").splitlines()
    if first_line:
        return first_line[0].strip()
    return "unknown"


def assert_ffmpeg_working(timeout_seconds: int = 10) -> dict[str, str]:
    ffmpeg_path = _resolved_executable_path(get_ffmpeg_path())
    ffprobe_path = _resolved_executable_path(get_ffprobe_path())
    ffmpeg_version = _version_for(ffmpeg_path, timeout_seconds=timeout_seconds)
    ffprobe_version = _version_for(ffprobe_path, timeout_seconds=timeout_seconds)
    return {
        "ffmpeg_path": ffmpeg_path,
        "ffprobe_path": ffprobe_path,
        "ffmpeg_version": ffmpeg_version,
        "ffprobe_version": ffprobe_version,
    }


def create_color_clip(
    ffmpeg_bin: str,
    output_path: Path,
    duration_seconds: float,
    fps: int,
    width: int,
    height: int,
    color_hex: str,
    timeout_seconds: int | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = f"color=c={color_hex}:s={width}x{height}:r={fps}:d={duration_seconds}"
    run_ffmpeg(
        ffmpeg_bin,
        [
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        timeout_seconds=timeout_seconds,
    )
    return output_path


def concat_videos_deterministic(
    ffmpeg_bin: str,
    clips: Iterable[Path],
    output_path: Path,
    timeout_seconds: int | None = None,
) -> Path:
    clip_list = [Path(item) for item in clips]
    if not clip_list:
        raise ValueError("No clips supplied for concatenation.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8", dir=str(output_path.parent)) as f:
        list_path = Path(f.name)
        for clip in clip_list:
            f.write(f"file '{clip.resolve().as_posix()}'\n")

    try:
        run_ffmpeg(
            ffmpeg_bin,
            [
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(output_path),
            ],
            timeout_seconds=timeout_seconds,
        )
    except RuntimeError:
        run_ffmpeg(
            ffmpeg_bin,
            [
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-vf",
                "format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                str(output_path),
            ],
            timeout_seconds=timeout_seconds,
        )
    finally:
        list_path.unlink(missing_ok=True)
    return output_path
