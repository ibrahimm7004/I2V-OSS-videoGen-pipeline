from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.job_schema import ShotSpec
from pipeline.prompting import compile_prompt
from scripts._job_loading import load_runtime_job_payload


def _fixture(motion_block_enabled: bool) -> dict:
    return {
        "version": 1,
        "run": {
            "id": "motion-block-fixture",
            "model_id": "wan22_ti2v_5b",
            "output_dir": "outputs",
            "seed_strategy": {"mode": "base_plus_offset", "base_seed": 10, "per_clip_offsets": [0]},
            "hf_cache": {"HF_HOME": "", "HF_HUB_CACHE": ""},
        },
        "video": {"width": 1280, "height": 704, "fps": 24, "clip_duration_sec": 2, "num_clips": 1},
        "generation_defaults": {
            "wan_profile": "quality",
            "steps": 60,
            "guidance_scale": 5.0,
            "sampler": "dpmpp_2m",
            "motion_block": motion_block_enabled,
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
        "shots": [
            {
                "index": 0,
                "prompt": "A subject standing in rain.",
            }
        ],
    }


def _render_prompt_for_fixture(payload: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8", dir=str(REPO_ROOT)) as tmp:
        path = Path(tmp.name)
        yaml.safe_dump(payload, tmp, sort_keys=False)
    try:
        runtime, _, _ = load_runtime_job_payload(path, dry_run_override=True, fast_mode=False)
    finally:
        path.unlink(missing_ok=True)

    shot = ShotSpec.model_validate(runtime["shots"][0])
    constants = runtime.get("constants") or {}
    motion_enabled = str(shot.params.get("motion_block", "")).strip().lower() in {"1", "true", "yes", "on"}
    motion_text = shot.params.get("motion_block_text") if motion_enabled else None
    prompt_text = compile_prompt(
        constants,
        shot,
        0,
        chain_last_frame_enabled=True,
        cinematic_constraints_enabled=True,
        motion_block_text=motion_text if isinstance(motion_text, str) else None,
    )
    return prompt_text


def main() -> int:
    prompt_enabled = _render_prompt_for_fixture(_fixture(True))
    for expected in ["motion_block:", "MOTION:", "- Subject:", "- Environment:", "- Camera:"]:
        if expected not in prompt_enabled:
            raise RuntimeError(f"Missing motion block content when enabled: {expected}")

    prompt_disabled = _render_prompt_for_fixture(_fixture(False))
    if any(line.strip() == "motion_block:" for line in prompt_disabled.splitlines()):
        raise RuntimeError("Motion block should be absent when disabled.")

    print("Motion block inclusion test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
