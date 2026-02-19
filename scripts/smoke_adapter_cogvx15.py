from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import get_adapter
from models.base import ClipRequest
from pipeline.config import load_config
from pipeline.ffmpeg_utils import get_ffprobe_path


def _probe_video(video_path: Path) -> dict:
    ffprobe_bin = get_ffprobe_path()
    command = [
        ffprobe_bin,
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({result.returncode}): {result.stderr.strip()}")
    return json.loads(result.stdout or "{}")


def _frame_count_from_probe(payload: dict) -> int:
    streams = payload.get("streams") or []
    if not streams:
        return 0
    stream = streams[0] or {}
    nb_frames = stream.get("nb_frames")
    if isinstance(nb_frames, str):
        nb_frames = nb_frames.strip()
        if nb_frames.isdigit():
            return int(nb_frames)
    if isinstance(nb_frames, int):
        return int(nb_frames)

    # Fallback to avg_frame_rate * duration.
    avg_rate = str(stream.get("avg_frame_rate", "0/1"))
    duration = float((payload.get("format") or {}).get("duration") or 0.0)
    if "/" in avg_rate:
        num_str, den_str = avg_rate.split("/", 1)
        try:
            num = float(num_str)
            den = float(den_str)
            if den > 0 and duration > 0:
                return int(round((num / den) * duration))
        except Exception:
            return 0
    return 0


def _min_expected_bytes(args: argparse.Namespace) -> int:
    if (
        float(args.duration_sec) >= 2.0
        and int(args.width) >= 512
        and int(args.height) >= 320
        and int(args.fps) >= 6
    ):
        return 100_000
    return 10_000


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single CogVideoX 1.5 I2V adapter smoke clip.")
    parser.add_argument("--input-image", type=Path, default=Path("assets/idea03/ref_01.png"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/_adapter_smoke/cogvx15"))
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=345678)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--sampler", type=str, default="dpmpp_2m")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--timeout-sec", type=int, default=600)
    args = parser.parse_args()

    input_image = args.input_image
    if not input_image.is_absolute():
        input_image = (REPO_ROOT / input_image).resolve()
    if not input_image.exists():
        raise SystemExit(f"Input image not found: {input_image}")

    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_clip = out_dir / "clip_000.mp4"
    frames = max(1, int(round(float(args.duration_sec) * int(args.fps))))

    config = load_config()
    adapter = get_adapter("cogvideox15_5b_i2v", "COGVX15_SMOKE", config)
    request = ClipRequest(
        clip_index=0,
        prompt="A gigantic prehistoric snake moving through bright jungle riverbank, realistic documentary style.",
        negative_prompt="text, logo, watermark, blurry, low quality, deformed anatomy",
        input_image=input_image,
        output_video_path=output_clip,
        duration_seconds=float(args.duration_sec),
        fps=int(args.fps),
        frames=frames,
        steps=int(args.steps),
        seed=int(args.seed),
        width=int(args.width),
        height=int(args.height),
        params={"cfg": float(args.guidance), "sampler": str(args.sampler)},
        global_params={"cfg": float(args.guidance), "sampler": str(args.sampler)},
        dry_run=False,
        max_runtime_seconds=int(args.timeout_sec),
    )
    result = adapter.generate_clip(request)
    if not result.output_video_path.exists():
        raise SystemExit(f"Smoke output clip missing: {result.output_video_path}")

    size_bytes = result.output_video_path.stat().st_size
    min_bytes = _min_expected_bytes(args)
    probe = _probe_video(result.output_video_path)
    frame_count = _frame_count_from_probe(probe)

    if size_bytes < min_bytes or frame_count <= 0:
        extra = result.extra_metadata or {}
        raise SystemExit(
            "CogVideoX smoke output failed validation: "
            f"size={size_bytes}B min_expected={min_bytes} frame_count={frame_count}. "
            f"adapter_stats={json.dumps(extra, sort_keys=True)}. "
            "Inspect extra_metadata in smoke_report.json and try "
            "COGVX15_DTYPE=bf16 with COGVX15_ENABLE_CPU_OFFLOAD=true."
        )

    report = {
        "output_clip": str(result.output_video_path),
        "runtime_seconds": result.runtime_seconds,
        "extra_metadata": result.extra_metadata,
        "input_image": str(input_image),
        "frames": frames,
        "steps": int(args.steps),
        "fps": int(args.fps),
        "mp4_size_bytes": int(size_bytes),
        "ffprobe": probe,
        "frame_count": int(frame_count),
    }
    report_path = out_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"CogVideoX smoke clip: {result.output_video_path}")
    print(f"Smoke report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
