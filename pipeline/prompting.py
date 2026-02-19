from __future__ import annotations

import json
from typing import Any

from .job_schema import ShotSpec


def compile_prompt(constants: dict[str, Any], shot: ShotSpec, clip_index: int) -> str:
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
    lines.append(f"negative_prompt: {shot.negative_prompt or ''}")
    lines.append(f"duration_seconds: {shot.duration_seconds}")
    lines.append(f"steps: {shot.steps}")
    lines.append(f"fps: {shot.fps}")
    lines.append(f"frames: {shot.frames}")
    lines.append(f"size: {shot.width}x{shot.height}")
    lines.append(f"shot_params: {json.dumps(shot.params, sort_keys=True)}")
    return "\n".join(lines)
