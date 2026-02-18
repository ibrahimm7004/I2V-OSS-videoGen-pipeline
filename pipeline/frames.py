from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import run_ffmpeg


def _extract_with_opencv(video_path: Path, output_path: Path) -> bool:
    try:
        import cv2  # type: ignore
    except Exception:
        return False

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False

    last_frame = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        last_frame = frame
    capture.release()
    if last_frame is None:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), last_frame))


def extract_last_frame(
    video_path: Path,
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: int | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if _extract_with_opencv(video_path, output_path):
        return output_path

    run_ffmpeg(
        ffmpeg_bin,
        [
            "-y",
            "-sseof",
            "-0.1",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(output_path),
        ],
        timeout_seconds=timeout_seconds,
    )
    return output_path
