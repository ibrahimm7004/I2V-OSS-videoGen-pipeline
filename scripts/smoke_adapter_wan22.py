from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import get_adapter
from models.base import ClipRequest
from pipeline.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a single WAN 2.2 adapter smoke clip.")
    parser.add_argument("--input-image", type=Path, default=Path("assets/idea01/ref_01.png"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/_adapter_smoke/wan22"))
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--sampler", type=str, default="dpmpp_2m")
    parser.add_argument("--seed", type=int, default=123456)
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
    adapter = get_adapter("wan22_ti2v_5b", "WAN22_SMOKE", config)
    request = ClipRequest(
        clip_index=0,
        prompt="A cinematic rainy neon street shot, realistic motion, no text.",
        negative_prompt="text, watermark, logo, blurry, deformed anatomy",
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
    if result.output_video_path.stat().st_size <= 0:
        raise SystemExit(f"Smoke output clip is empty: {result.output_video_path}")

    report = {
        "output_clip": str(result.output_video_path),
        "runtime_seconds": result.runtime_seconds,
        "extra_metadata": result.extra_metadata,
        "input_image": str(input_image),
        "frames": frames,
        "steps": int(args.steps),
        "fps": int(args.fps),
    }
    report_path = out_dir / "smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"WAN smoke clip: {result.output_video_path}")
    print(f"Smoke report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
