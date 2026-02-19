from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.job_schema import ShotSpec
from pipeline.prompting import CINEMATIC_CONSTRAINTS, compile_prompt


def main() -> int:
    shot = ShotSpec(
        prompt="A calm portrait shot.",
        negative_prompt="",
        duration_seconds=2.0,
        steps=20,
        fps=8,
        width=512,
        height=512,
        params={},
    )
    constants = {
        "global_prompt": "Base prompt",
        "continuity_rules": ["Keep identity."],
        "cinematic_constraints_enabled": True,
    }

    prompt_enabled = compile_prompt(constants, shot, 0, chain_last_frame_enabled=True)
    for phrase in CINEMATIC_CONSTRAINTS.splitlines()[1:]:
        if phrase not in prompt_enabled:
            raise RuntimeError(f"Missing cinematic constraints phrase when enabled: {phrase}")

    prompt_disabled = compile_prompt(
        constants,
        shot,
        0,
        chain_last_frame_enabled=True,
        cinematic_constraints_enabled=False,
    )
    if "[CINEMATIC_CONSTRAINTS]" in prompt_disabled:
        raise RuntimeError("Cinematic constraints block should be absent when disabled.")

    print("Prompt cinematic constraints inclusion test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
