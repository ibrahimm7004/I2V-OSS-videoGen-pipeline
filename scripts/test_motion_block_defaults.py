from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._job_loading import (
    DEFAULT_MOTION_CAMERA,
    DEFAULT_MOTION_ENVIRONMENT,
    DEFAULT_MOTION_SUBJECT,
    load_runtime_job_payload,
)


def _fixture_missing_motion_fields() -> dict:
    return {
        "version": 1,
        "run": {
            "id": "motion-defaults-fixture",
            "model_id": "wan22_ti2v_5b",
            "output_dir": "outputs",
            "seed_strategy": {"mode": "base_plus_offset", "base_seed": 123, "per_clip_offsets": [0]},
            "hf_cache": {"HF_HOME": "", "HF_HUB_CACHE": ""},
        },
        "video": {"width": 1280, "height": 704, "fps": 24, "clip_duration_sec": 2, "num_clips": 1},
        "generation_defaults": {
            "wan_profile": "quality",
            "steps": 60,
            "guidance_scale": 5.0,
            "sampler": "dpmpp_2m",
            "motion_block": True,
        },
        "stitching": {
            "enabled": True,
            "mode": "concat",
            "crossfade_sec": 0,
            "rife_boundary_smoothing": {"enabled": False, "boundary_sec": 0.0},
        },
        "prompts": {
            "global_prompt": "Global",
            "negative_prompt": "no text",
            "continuity_rules": ["keep identity"],
        },
        "inputs": {"initial_images": ["assets/wan/idea01/ref_01.png"]},
        "shots": [{"index": 0, "prompt": "Simple shot with missing motion overrides."}],
    }


def main() -> int:
    payload = _fixture_missing_motion_fields()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8", dir=str(REPO_ROOT)) as tmp:
        path = Path(tmp.name)
        yaml.safe_dump(payload, tmp, sort_keys=False)
    try:
        runtime, _, _ = load_runtime_job_payload(path, dry_run_override=True, fast_mode=False)
    finally:
        path.unlink(missing_ok=True)

    shot = runtime["shots"][0]
    params = shot.get("params") or {}

    if params.get("motion_subject") != DEFAULT_MOTION_SUBJECT:
        raise RuntimeError("Default motion_subject not inserted.")
    if params.get("motion_environment") != DEFAULT_MOTION_ENVIRONMENT:
        raise RuntimeError("Default motion_environment not inserted.")
    if params.get("motion_camera") != DEFAULT_MOTION_CAMERA:
        raise RuntimeError("Default motion_camera not inserted.")

    motion_text = str(params.get("motion_block_text") or "")
    for expected in ["MOTION:", "- Subject:", "- Environment:", "- Camera:"]:
        if expected not in motion_text:
            raise RuntimeError(f"Missing expected default motion block token: {expected}")

    print("Motion block defaults test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
