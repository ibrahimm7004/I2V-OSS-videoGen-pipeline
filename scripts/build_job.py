from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

IDEA_SOURCES: dict[str, Path] = {
    "idea01": REPO_ROOT / "jobs" / "idea01_wan.yaml",
    "idea02": REPO_ROOT / "jobs" / "idea02_hunyuan.yaml",
    "idea03": REPO_ROOT / "jobs" / "idea03_cogvideox.yaml",
}

DEFAULT_MOTION_SUBJECT = (
    "subtle natural body movement; breathing; small head turns; micro-expressions"
)
DEFAULT_MOTION_ENVIRONMENT = (
    "rain streaks and droplets; neon reflections shimmering on wet asphalt; steam drifting; light wind"
)
DEFAULT_MOTION_CAMERA = (
    "slow subtle dolly-in only; no rapid zoom; no sudden framing changes; subject centered"
)
DEFAULT_MOTION_NOTES = ""

# Optional idea-specific motion enrichments keyed by shot index for stronger defaults.
MOTION_OVERRIDES: dict[str, dict[int, dict[str, str]]] = {
    "idea01": {
        0: {
            "motion_subject": "steady forward walk under umbrella; slight shoulder sway; eyes tracking forward",
            "motion_environment": "rain sheets, droplets on umbrella, drifting steam from vent, shimmering neon puddles",
        },
        2: {
            "motion_subject": "natural footstep impact through puddle with subtle leg/coat motion",
            "motion_camera": "gentle tilt-down then hold; no rapid zoom; maintain stable framing",
        },
        7: {
            "motion_subject": "smooth turn then walk-away with natural gait cadence",
            "motion_camera": "slow pull-back only; no sudden framing changes; centered composition",
        },
    },
    "idea02": {
        1: {
            "motion_subject": "fingers open case hinge smoothly; wrists stable with natural micro tremor",
        },
        4: {
            "motion_subject": "small exhale, relaxed shoulders, calm facial settling",
            "motion_camera": "slow subtle push-in only; no rapid zoom; subject centered",
        },
    },
    "idea03": {
        0: {
            "motion_subject": "massive serpentine head advance with heavy inertial sway",
            "motion_environment": "foliage displacement, sand compression, river ripple propagation, mist drift",
        },
        9: {
            "motion_subject": "broad undulating glide with consistent body wave rhythm",
            "motion_camera": "side-follow tracking only; no rapid zoom; maintain scale references",
        },
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at root in {path}")
    return data


def _default_offsets(num_clips: int) -> list[int]:
    base = [0, 11, 23, 37, 49, 61, 74, 88, 101, 117, 131, 149]
    if num_clips <= len(base):
        return base[:num_clips]
    values = list(base)
    while len(values) < num_clips:
        values.append(values[-1] + 17)
    return values


def compose_motion_block(
    *,
    motion_subject: str,
    motion_environment: str,
    motion_camera: str,
    motion_notes: str = "",
) -> str:
    lines = [
        "MOTION:",
        f"- Subject: {motion_subject}",
        f"- Environment: {motion_environment}",
        f"- Camera: {motion_camera}",
    ]
    if motion_notes.strip():
        lines.append(f"- Notes: {motion_notes.strip()}")
    return "\n".join(lines)


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_motion_fields(
    *,
    idea: str,
    shot_index: int,
    shot_data: dict[str, Any],
) -> dict[str, Any]:
    override = MOTION_OVERRIDES.get(idea, {}).get(shot_index, {})
    motion_subject = str(
        shot_data.get("motion_subject")
        or override.get("motion_subject")
        or DEFAULT_MOTION_SUBJECT
    ).strip()
    motion_environment = str(
        shot_data.get("motion_environment")
        or override.get("motion_environment")
        or DEFAULT_MOTION_ENVIRONMENT
    ).strip()
    motion_camera = str(
        shot_data.get("motion_camera")
        or override.get("motion_camera")
        or DEFAULT_MOTION_CAMERA
    ).strip()
    motion_notes = str(
        shot_data.get("motion_notes")
        or override.get("motion_notes")
        or DEFAULT_MOTION_NOTES
    ).strip()
    motion_block = _coerce_bool(shot_data.get("motion_block"), True)
    return {
        "motion_subject": motion_subject,
        "motion_environment": motion_environment,
        "motion_camera": motion_camera,
        "motion_notes": motion_notes,
        "motion_block": motion_block,
    }


def _build_wan_job(source: dict[str, Any], idea: str) -> dict[str, Any]:
    src_run = source.get("run") if isinstance(source.get("run"), dict) else {}
    src_video = source.get("video") if isinstance(source.get("video"), dict) else {}
    src_prompts = source.get("prompts") if isinstance(source.get("prompts"), dict) else {}
    src_shots = source.get("shots") if isinstance(source.get("shots"), list) else []

    num_clips = int(src_video.get("num_clips", len(src_shots) or 8))
    if num_clips <= 0:
        num_clips = len(src_shots) or 8

    base_seed = int(((src_run.get("seed_strategy") or {}).get("base_seed")) or 100001)
    per_clip_offsets = _default_offsets(num_clips)

    generation_defaults_src = (
        source.get("generation_defaults")
        if isinstance(source.get("generation_defaults"), dict)
        else {}
    )
    source_profile = str(
        generation_defaults_src.get("wan_profile")
        or generation_defaults_src.get("profile")
        or "quality"
    ).strip().lower()
    default_motion_block = False if source_profile == "smoke" else True

    shots: list[dict[str, Any]] = []
    for idx, shot in enumerate(src_shots[:num_clips]):
        if isinstance(shot, dict):
            shot_data = dict(shot)
            prompt = str(shot_data.get("prompt", "")).strip()
            name = shot.get("name")
        else:
            shot_data = {}
            prompt = str(shot).strip()
            name = None
        motion = _resolve_motion_fields(idea=idea, shot_index=idx, shot_data=shot_data)
        motion_block_text = compose_motion_block(
            motion_subject=motion["motion_subject"],
            motion_environment=motion["motion_environment"],
            motion_camera=motion["motion_camera"],
            motion_notes=motion["motion_notes"],
        )
        prompt_with_motion = prompt or f"Clip {idx + 1} prompt for {idea}."
        if motion["motion_block"]:
            prompt_with_motion = f"{prompt_with_motion}\n\n{motion_block_text}"
        shots.append(
            {
                "index": idx,
                **({"name": str(name)} if name else {}),
                "prompt": prompt_with_motion,
                "motion_block": motion["motion_block"],
                "motion_subject": motion["motion_subject"],
                "motion_environment": motion["motion_environment"],
                "motion_camera": motion["motion_camera"],
                "motion_notes": motion["motion_notes"],
            }
        )
    while len(shots) < num_clips:
        idx = len(shots)
        motion = _resolve_motion_fields(idea=idea, shot_index=idx, shot_data={})
        motion_block_text = compose_motion_block(
            motion_subject=motion["motion_subject"],
            motion_environment=motion["motion_environment"],
            motion_camera=motion["motion_camera"],
            motion_notes=motion["motion_notes"],
        )
        prompt_with_motion = f"Clip {idx + 1} prompt for {idea}."
        if motion["motion_block"]:
            prompt_with_motion = f"{prompt_with_motion}\n\n{motion_block_text}"
        shots.append(
            {
                "index": idx,
                "name": f"{idea}_clip_{idx:02d}",
                "prompt": prompt_with_motion,
                "motion_block": motion["motion_block"],
                "motion_subject": motion["motion_subject"],
                "motion_environment": motion["motion_environment"],
                "motion_camera": motion["motion_camera"],
                "motion_notes": motion["motion_notes"],
            }
        )

    continuity_rules = src_prompts.get("continuity_rules")
    if not isinstance(continuity_rules, list) or not continuity_rules:
        continuity_rules = [
            "Preserve main subject identity and wardrobe across clips.",
            "Maintain environment continuity and lighting consistency between clips.",
            "No readable text, logos, or watermarks.",
        ]

    return {
        "version": 1,
        "run": {
            "id": f"wan22_{idea}_v1",
            "model_id": "wan22_ti2v_5b",
            "output_dir": "outputs",
            "seed_strategy": {
                "mode": "base_plus_offset",
                "base_seed": base_seed,
                "per_clip_offsets": per_clip_offsets,
            },
            "hf_cache": {
                "HF_HOME": "",
                "HF_HUB_CACHE": "",
            },
        },
        "video": {
            "width": 1280,
            "height": 704,
            "fps": int(src_video.get("fps", 24) or 24),
            "clip_duration_sec": float(src_video.get("clip_duration_sec", 5) or 5),
            "num_clips": num_clips,
        },
        "generation_defaults": {
            "wan_profile": "quality",
            "steps": 60,
            "guidance_scale": 5.0,
            "sampler": "dpmpp_2m",
            "motion_strength": 0.70,
            "motion_block": default_motion_block,
            "export_quality": 9,
            "chain_last_frame": True,
            "other": {},
        },
        "stitching": {
            "enabled": True,
            "mode": "concat",
            "crossfade_sec": 0,
            "rife_boundary_smoothing": {"enabled": False, "boundary_sec": 0.0},
        },
        "prompts": {
            "global_prompt": str(src_prompts.get("global_prompt", "")).strip(),
            "negative_prompt": str(src_prompts.get("negative_prompt", "")).strip(),
            "continuity_rules": continuity_rules,
        },
        "inputs": {
            "initial_images": [
                f"assets/wan/{idea}/ref_01.png",
                f"assets/wan/{idea}/ref_02.png",
            ]
        },
        "shots": shots,
    }


def _sync_assets(idea: str) -> None:
    source_dir = REPO_ROOT / "assets" / idea
    target_dir = REPO_ROOT / "assets" / "wan" / idea
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in ["ref_01.png", "ref_02.png"]:
        src = source_dir / file_name
        if not src.exists():
            continue
        dst = target_dir / file_name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WAN job-pack YAMLs from existing idea specs.")
    parser.add_argument(
        "--ideas",
        nargs="+",
        choices=sorted(IDEA_SOURCES.keys()),
        default=sorted(IDEA_SOURCES.keys()),
        help="Idea set to build (default: all).",
    )
    parser.add_argument(
        "--sync-assets",
        action="store_true",
        help="Copy ref_01/ref_02 assets from assets/<idea>/ to assets/wan/<idea>/.",
    )
    args = parser.parse_args()

    jobs_wan_dir = REPO_ROOT / "jobs" / "wan"
    jobs_wan_dir.mkdir(parents=True, exist_ok=True)
    if args.sync_assets:
        (REPO_ROOT / "assets" / "wan").mkdir(parents=True, exist_ok=True)

    for idea in args.ideas:
        source_path = IDEA_SOURCES[idea]
        if not source_path.exists():
            print(f"Missing source file for {idea}: {source_path}", file=sys.stderr)
            return 1
        source_data = _load_yaml(source_path)
        built = _build_wan_job(source_data, idea)
        output_path = jobs_wan_dir / f"{idea}.yaml"
        output_path.write_text(yaml.safe_dump(built, sort_keys=False), encoding="utf-8")
        print(f"Wrote {output_path.relative_to(REPO_ROOT).as_posix()}")
        if args.sync_assets:
            _sync_assets(idea)
            print(f"Synced assets/wan/{idea}/ref_01.png and ref_02.png (if present)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
