from __future__ import annotations

import json
from typing import Any

from .job_schema import ShotSpec


def compile_prompt(constants: dict[str, Any], shot: ShotSpec, clip_index: int) -> str:
    """
    Compile a deterministic prompt payload from global constants + a shot card.
    """
    lines: list[str] = ["[GLOBAL_CONSTANTS]"]
    if constants:
        for key in sorted(constants.keys()):
            lines.append(f"{key}: {json.dumps(constants[key], sort_keys=True)}")
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("[SHOT_CARD]")
    lines.append(f"clip_index: {clip_index}")
    lines.append(f"prompt: {shot.prompt}")
    lines.append(f"negative_prompt: {shot.negative_prompt or ''}")
    lines.append(f"duration_seconds: {shot.duration_seconds}")
    lines.append(f"steps: {shot.steps}")
    lines.append(f"fps: {shot.fps}")
    lines.append(f"frames: {shot.frames}")
    lines.append(f"size: {shot.width}x{shot.height}")
    lines.append(f"shot_params: {json.dumps(shot.params, sort_keys=True)}")
    return "\n".join(lines)

