from __future__ import annotations

import json
from typing import Any

from .job_schema import ShotSpec

CINEMATIC_CONSTRAINTS = """[CINEMATIC_CONSTRAINTS]
- slow subtle dolly-in only
- no rapid zoom
- no sudden framing changes
- subject centered
- face identity consistent
- crisp focus on subject
- fine fabric detail
- no smear
- 35mm lens
- filmic contrast"""


def _to_bool(value: Any, default: bool) -> bool:
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


def compile_prompt(
    constants: dict[str, Any],
    shot: ShotSpec,
    clip_index: int,
    *,
    chain_last_frame_enabled: bool = False,
    cinematic_constraints_enabled: bool | None = None,
    motion_block_text: str | None = None,
) -> str:
    """
    Compile a deterministic prompt payload from global constants + a shot card.
    """
    global_prompt = str(constants.get("global_prompt", "")).strip()
    continuity_rules = constants.get("continuity_rules")
    if not isinstance(continuity_rules, list):
        continuity_rules = []
    rules = [str(item).strip() for item in continuity_rules if str(item).strip()]

    lines: list[str] = ["[GLOBAL_PROMPT]"]
    lines.append(global_prompt or "(none)")
    lines.append("")
    lines.append("[CONTINUITY_RULES]")
    if rules:
        for idx, rule in enumerate(rules, start=1):
            lines.append(f"{idx}. {rule}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("[SHOT_CARD]")
    lines.append(f"clip_index: {clip_index}")
    lines.append(f"shot_prompt: {shot.prompt}")
    if isinstance(motion_block_text, str) and motion_block_text.strip():
        lines.append("motion_block:")
        lines.extend(motion_block_text.strip().splitlines())
    lines.append(f"negative_prompt: {shot.negative_prompt or ''}")
    lines.append(f"duration_seconds: {shot.duration_seconds}")
    lines.append(f"steps: {shot.steps}")
    lines.append(f"fps: {shot.fps}")
    lines.append(f"frames: {shot.frames}")
    lines.append(f"size: {shot.width}x{shot.height}")
    lines.append(f"shot_params: {json.dumps(shot.params, sort_keys=True)}")

    lines.append("")
    lines.append("[MOTION_AND_CONTINUITY]")
    lines.append("- include explicit subject motion, environment motion, and camera motion in this shot")
    if chain_last_frame_enabled:
        lines.append(
            "- match previous frame; preserve character identity; preserve clothing and color palette"
        )
    lines.append("- no face melting; no deforming; no extra limbs; no text")

    effective_constraints = cinematic_constraints_enabled
    if effective_constraints is None:
        effective_constraints = _to_bool(constants.get("cinematic_constraints_enabled"), True)
    if effective_constraints:
        lines.append("")
        lines.extend(CINEMATIC_CONSTRAINTS.splitlines())
    return "\n".join(lines)
