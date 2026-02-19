from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _run_ffprobe(ffprobe_bin: str, video_path: Path, timeout_seconds: int) -> dict[str, Any]:
    command = [
        ffprobe_bin,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames,width,height,avg_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=max(5, int(timeout_seconds)),
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"ffprobe executable not found: {ffprobe_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"ffprobe timed out while validating clip: {video_path}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed ({result.returncode}) for {video_path}: {result.stderr.strip()}"
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe returned invalid JSON for {video_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe payload is not an object for {video_path}")
    return payload


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            parsed = int(text)
            return parsed if parsed > 0 else None
    return None


def _extract_probe_info(payload: dict[str, Any]) -> tuple[int, int, int]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise RuntimeError("ffprobe returned no video stream information.")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise RuntimeError("ffprobe stream payload is malformed.")

    width = _coerce_positive_int(stream.get("width"))
    height = _coerce_positive_int(stream.get("height"))
    if width is None or height is None:
        raise RuntimeError("Could not read width/height from ffprobe output.")

    frame_count = _coerce_positive_int(stream.get("nb_read_frames"))
    if frame_count is None:
        frame_count = _coerce_positive_int(stream.get("nb_frames"))
    if frame_count is None:
        raise RuntimeError("Could not read nb_frames from ffprobe output.")

    return width, height, frame_count


def _extract_frame_rgb24(
    ffmpeg_bin: str,
    video_path: Path,
    frame_index: int,
    width: int,
    height: int,
    timeout_seconds: int,
) -> bytes:
    expected_size = int(width) * int(height) * 3
    with tempfile.NamedTemporaryFile(suffix=".rgb24", delete=False) as tmp:
        raw_path = Path(tmp.name)

    command = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-vframes",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        str(raw_path),
    ]
    try:
        try:
            result = subprocess.run(
                command,
                text=True,
                capture_output=True,
                timeout=max(5, int(timeout_seconds)),
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"ffmpeg executable not found: {ffmpeg_bin}") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"ffmpeg timed out while extracting frame {frame_index} from {video_path}"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg frame extraction failed ({result.returncode}) for frame {frame_index}: "
                f"{result.stderr.strip()}"
            )
        frame_bytes = raw_path.read_bytes()
    finally:
        raw_path.unlink(missing_ok=True)

    if len(frame_bytes) != expected_size:
        raise RuntimeError(
            f"Extracted frame byte size mismatch for frame {frame_index}: "
            f"expected {expected_size}, got {len(frame_bytes)}"
        )
    return frame_bytes


def _mean_abs_diff_normalized(first_frame: bytes, last_frame: bytes) -> float:
    if len(first_frame) != len(last_frame):
        raise RuntimeError("Frame-diff metric failed: frame sizes differ.")
    if not first_frame:
        raise RuntimeError("Frame-diff metric failed: empty frame payload.")

    total = 0
    for a, b in zip(first_frame, last_frame):
        total += abs(a - b)
    return total / (len(first_frame) * 255.0)


def validate_clip_output(
    *,
    ffmpeg_bin: str,
    ffprobe_bin: str,
    video_path: Path,
    requested_num_frames: int,
    min_size_bytes: int,
    min_frame_diff: float,
    timeout_seconds: int,
) -> dict[str, Any]:
    if requested_num_frames <= 0:
        raise RuntimeError(f"Invalid requested_num_frames={requested_num_frames} for clip validation.")
    if not video_path.exists():
        raise RuntimeError(f"Clip output missing: {video_path}")

    mp4_size = video_path.stat().st_size
    if mp4_size < int(min_size_bytes):
        raise RuntimeError(
            "clip output too small: "
            f"{mp4_size} bytes < threshold {int(min_size_bytes)} bytes"
        )

    probe_payload = _run_ffprobe(ffprobe_bin, video_path, timeout_seconds)
    width, height, actual_num_frames = _extract_probe_info(probe_payload)
    if actual_num_frames != int(requested_num_frames):
        raise RuntimeError(
            "ffprobe nb_frames mismatch: "
            f"requested {int(requested_num_frames)}, got {actual_num_frames}"
        )

    first_frame = _extract_frame_rgb24(
        ffmpeg_bin=ffmpeg_bin,
        video_path=video_path,
        frame_index=0,
        width=width,
        height=height,
        timeout_seconds=timeout_seconds,
    )
    last_frame = _extract_frame_rgb24(
        ffmpeg_bin=ffmpeg_bin,
        video_path=video_path,
        frame_index=max(0, actual_num_frames - 1),
        width=width,
        height=height,
        timeout_seconds=timeout_seconds,
    )
    frame_diff = _mean_abs_diff_normalized(first_frame, last_frame)
    if frame_diff < float(min_frame_diff):
        raise RuntimeError(
            "static output, likely over-conditioning or motion not applied: "
            f"frame_diff={frame_diff:.6f} threshold={float(min_frame_diff):.6f}"
        )

    return {
        "requested_num_frames": int(requested_num_frames),
        "actual_num_frames": int(actual_num_frames),
        "mp4_size_bytes": int(mp4_size),
        "first_last_frame_diff": float(frame_diff),
        "min_size_bytes_threshold": int(min_size_bytes),
        "min_frame_diff_threshold": float(min_frame_diff),
        "width": int(width),
        "height": int(height),
    }
