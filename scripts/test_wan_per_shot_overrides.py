from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.job_schema import JobSpec
from scripts._job_loading import load_runtime_job_payload


def _make_fixture() -> dict:
    return {
        "version": 1,
        "run": {
            "id": "wan-overrides-fixture",
            "model_id": "wan22_ti2v_5b",
            "output_dir": "outputs",
            "seed_strategy": {"mode": "base_plus_offset", "base_seed": 1000, "per_clip_offsets": [0, 11]},
            "hf_cache": {"HF_HOME": "", "HF_HUB_CACHE": ""},
        },
        "video": {"width": 1280, "height": 704, "fps": 24, "clip_duration_sec": 5, "num_clips": 2},
        "generation_defaults": {
            "wan_profile": "quality",
            "steps": 60,
            "guidance_scale": 5.0,
            "sampler": "dpmpp_2m",
            "motion_strength": 0.70,
            "export_quality": 9,
            "chain_last_frame": True,
        },
        "stitching": {
            "enabled": True,
            "mode": "concat",
            "crossfade_sec": 0,
            "rife_boundary_smoothing": {"enabled": False, "boundary_sec": 0.0},
        },
        "prompts": {
            "global_prompt": "Global prompt",
            "negative_prompt": "no text",
            "continuity_rules": ["keep identity"],
        },
        "inputs": {"initial_images": ["assets/wan/idea01/ref_01.png"]},
        "shots": [
            {
                "index": 0,
                "prompt": "shot zero",
                "motion_strength": 0.82,
                "guidance_scale": 4.8,
                "steps": 68,
                "cinematic_constraints": True,
            },
            {
                "index": 1,
                "prompt": "shot one",
                "motion_strength": 0.62,
                "guidance_scale": 5.4,
                "cinematic_constraints": False,
            },
        ],
    }


def main() -> int:
    fixture = _make_fixture()
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8", dir=str(REPO_ROOT)) as tmp:
        fixture_path = Path(tmp.name)
        yaml.safe_dump(fixture, tmp, sort_keys=False)

    try:
        runtime, source_kind, _ = load_runtime_job_payload(
            fixture_path,
            run_id_override="wan-overrides-runtime",
            output_root_override="outputs",
            dry_run_override=True,
            fast_mode=False,
        )
    finally:
        fixture_path.unlink(missing_ok=True)

    if source_kind != "job_pack":
        raise RuntimeError(f"Expected source_kind=job_pack, got {source_kind}")
    JobSpec.model_validate(runtime)

    constants = runtime.get("constants") or {}
    if constants.get("cinematic_constraints_enabled") is not True:
        raise RuntimeError("Quality profile should default cinematic_constraints_enabled=True.")

    shots = runtime.get("shots")
    if not isinstance(shots, list) or len(shots) != 2:
        raise RuntimeError("Expected 2 runtime shots.")

    shot0 = shots[0]
    shot1 = shots[1]
    if int(shot0.get("steps", -1)) != 68:
        raise RuntimeError(f"Shot 0 steps override failed: {shot0.get('steps')}")
    if float((shot0.get("params") or {}).get("motion_strength", -1)) != 0.82:
        raise RuntimeError("Shot 0 motion_strength override failed.")
    if float((shot0.get("params") or {}).get("guidance_scale", -1)) != 4.8:
        raise RuntimeError("Shot 0 guidance_scale override failed.")
    if (shot0.get("params") or {}).get("cinematic_constraints") is not True:
        raise RuntimeError("Shot 0 cinematic_constraints override failed.")

    if float((shot1.get("params") or {}).get("motion_strength", -1)) != 0.62:
        raise RuntimeError("Shot 1 motion_strength override failed.")
    if float((shot1.get("params") or {}).get("guidance_scale", -1)) != 5.4:
        raise RuntimeError("Shot 1 guidance_scale override failed.")
    if (shot1.get("params") or {}).get("cinematic_constraints") is not False:
        raise RuntimeError("Shot 1 cinematic_constraints override failed.")

    print("WAN per-shot override mapping test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
