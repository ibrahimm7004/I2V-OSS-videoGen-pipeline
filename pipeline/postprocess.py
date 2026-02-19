from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .ffmpeg_utils import run_ffmpeg


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text not in {"0", "false", "no", "off", ""}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _run_upscaler(
    *,
    input_video: Path,
    output_video: Path,
    ffmpeg_bin: str,
    fps: int,
    upscaler_cfg: dict[str, Any],
    timeout_seconds: int,
) -> tuple[bool, str | None]:
    binary = _as_str(upscaler_cfg.get("binary"), "realesrgan-ncnn-vulkan")
    resolved = shutil.which(binary)
    if resolved is None:
        return False, f"upscaler binary '{binary}' not found on PATH; skipping upscale."

    extra_args_raw = upscaler_cfg.get("args", "")
    try:
        extra_args = shlex.split(str(extra_args_raw), posix=False)
    except Exception:
        extra_args = []

    with tempfile.TemporaryDirectory(prefix="wan-upscale-") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        in_dir = tmp_dir / "in"
        out_dir = tmp_dir / "out"
        in_dir.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        input_pattern = in_dir / "frame_%06d.png"
        output_pattern = out_dir / "frame_%06d.png"

        run_ffmpeg(
            ffmpeg_bin,
            [
                "-y",
                "-i",
                str(input_video),
                str(input_pattern),
            ],
            timeout_seconds=timeout_seconds,
        )

        frames = sorted(in_dir.glob("frame_*.png"))
        if not frames:
            return False, "no frames extracted for upscaler; skipping upscale."

        for frame in frames:
            out_frame = out_dir / frame.name
            command = [resolved, "-i", str(frame), "-o", str(out_frame), *extra_args]
            try:
                result = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    timeout=max(5, timeout_seconds),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False, f"upscaler timed out on frame {frame.name}; skipping upscale."
            if result.returncode != 0:
                return (
                    False,
                    f"upscaler failed on frame {frame.name}: {result.stderr.strip() or result.stdout.strip()}",
                )

        run_ffmpeg(
            ffmpeg_bin,
            [
                "-y",
                "-framerate",
                str(max(1, int(fps))),
                "-i",
                str(output_pattern),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output_video),
            ],
            timeout_seconds=timeout_seconds,
        )
    return True, None


def run_optional_wan_postprocess(
    *,
    video_path: Path,
    ffmpeg_bin: str,
    fps: int,
    postprocess_config: dict[str, Any] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "enabled": False,
        "applied": False,
        "upscaler_applied": False,
        "warnings": [],
    }
    config = postprocess_config if isinstance(postprocess_config, dict) else {}
    if not _as_bool(config.get("enable"), default=False):
        return metadata
    metadata["enabled"] = True

    input_for_encode = video_path
    upscaler_cfg = config.get("upscaler")
    if isinstance(upscaler_cfg, dict) and _as_bool(upscaler_cfg.get("enable"), default=False):
        upscaled_video = video_path.with_suffix(".upscaled.mp4")
        ok, warn = _run_upscaler(
            input_video=video_path,
            output_video=upscaled_video,
            ffmpeg_bin=ffmpeg_bin,
            fps=fps,
            upscaler_cfg=upscaler_cfg,
            timeout_seconds=timeout_seconds,
        )
        if ok and upscaled_video.exists():
            metadata["upscaler_applied"] = True
            input_for_encode = upscaled_video
        elif warn:
            metadata["warnings"].append(warn)

    crf = _as_int(config.get("ffmpeg_crf"), 17)
    preset = _as_str(config.get("preset"), "slow")
    tune = str(config.get("tune")).strip() if config.get("tune") is not None else ""
    bitrate = str(config.get("bitrate")).strip() if config.get("bitrate") is not None else ""
    sharpen = config.get("sharpen")

    filters: list[str] = []
    if _as_bool(sharpen, default=False) or (isinstance(sharpen, str) and sharpen.strip().lower() == "mild"):
        filters.append("unsharp=5:5:0.5:5:5:0.0")

    output_tmp = video_path.with_suffix(".post.mp4")
    args = [
        "-y",
        "-i",
        str(input_for_encode),
    ]
    if filters:
        args.extend(["-vf", ",".join(filters)])
    args.extend(["-c:v", "libx264", "-preset", preset, "-crf", str(crf)])
    if tune:
        args.extend(["-tune", tune])
    if bitrate:
        args.extend(["-b:v", bitrate])
    args.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_tmp)])

    try:
        run_ffmpeg(ffmpeg_bin, args, timeout_seconds=timeout_seconds)
        output_tmp.replace(video_path)
        metadata["applied"] = True
    except Exception as exc:
        metadata["warnings"].append(f"postprocess encode failed: {exc}")
    finally:
        output_tmp.unlink(missing_ok=True)
        if input_for_encode != video_path:
            input_for_encode.unlink(missing_ok=True)

    metadata["ffmpeg_crf_used"] = crf
    metadata["preset_used"] = preset
    metadata["tune_used"] = tune or None
    metadata["bitrate_used"] = bitrate or None
    metadata["sharpen_used"] = filters[0] if filters else None
    return metadata
