from __future__ import annotations

from pathlib import Path

from .ffmpeg_utils import concat_videos_deterministic


def rife_boundary_smoothing_stub(clip_paths: list[Path]) -> list[Path]:
    """
    Placeholder for future RIFE interpolation at shot boundaries.

    TODO: Implement frame interpolation blending across clip boundaries.
    """
    return clip_paths


def stitch_clips_deterministic(
    clip_paths: list[Path],
    output_path: Path,
    ffmpeg_bin: str = "ffmpeg",
    timeout_seconds: int | None = None,
) -> Path:
    smoothed_paths = rife_boundary_smoothing_stub(clip_paths)
    try:
        return concat_videos_deterministic(ffmpeg_bin, smoothed_paths, output_path, timeout_seconds=timeout_seconds)
    except RuntimeError:
        return _concat_with_opencv(smoothed_paths, output_path)


def _concat_with_opencv(clip_paths: list[Path], output_path: Path) -> Path:
    try:
        import cv2  # type: ignore
    except Exception as exc:
        raise RuntimeError("Stitching requires ffmpeg or opencv-python-headless.") from exc

    if not clip_paths:
        raise ValueError("No clip paths provided for stitching.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_capture = cv2.VideoCapture(str(clip_paths[0]))
    if not first_capture.isOpened():
        raise RuntimeError(f"Unable to open clip for stitching: {clip_paths[0]}")

    fps = first_capture.get(cv2.CAP_PROP_FPS) or 8.0
    width = int(first_capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 512)
    height = int(first_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 512)
    first_capture.release()

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open output for stitching: {output_path}")

    try:
        for clip_path in clip_paths:
            capture = cv2.VideoCapture(str(clip_path))
            if not capture.isOpened():
                raise RuntimeError(f"Unable to open clip: {clip_path}")
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
            capture.release()
    finally:
        writer.release()
    return output_path
