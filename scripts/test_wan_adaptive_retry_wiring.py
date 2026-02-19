from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.runner import _next_wan_retry_params, _wan_retry_reason


def main() -> int:
    if _wan_retry_reason(RuntimeError("static output, likely over-conditioning")) != "static_output":
        raise RuntimeError("static retry reason mapping failed")
    if _wan_retry_reason(RuntimeError("ffprobe nb_frames mismatch: requested 121, got 97")) != "encoded_frame_mismatch":
        raise RuntimeError("frame mismatch retry reason mapping failed")
    if _wan_retry_reason(RuntimeError("clip output too small: 100 bytes")) != "mp4_too_small":
        raise RuntimeError("small mp4 retry reason mapping failed")
    if _wan_retry_reason(RuntimeError("random adapter error")) is not None:
        raise RuntimeError("unexpected retry reason for unrelated error")

    params, next_steps, before, after = _next_wan_retry_params(
        current_params={"motion_strength": 0.70, "cfg": 5.0, "guidance_scale": 5.0},
        global_params={},
        current_steps=60,
    )
    if float(params["motion_strength"]) != 0.8:
        raise RuntimeError(f"Expected motion_strength=0.8 after retry, got {params['motion_strength']}")
    if float(params["guidance_scale"]) != 4.7:
        raise RuntimeError(f"Expected guidance_scale=4.7 after retry, got {params['guidance_scale']}")
    if int(next_steps) != 66:
        raise RuntimeError(f"Expected next_steps=66, got {next_steps}")
    if before.get("steps") != 60 or after.get("steps") != 66:
        raise RuntimeError("Retry before/after step metadata mismatch.")

    capped_params, capped_steps, _, _ = _next_wan_retry_params(
        current_params={"motion_strength": 0.85, "cfg": 4.5},
        global_params={},
        current_steps=72,
    )
    if float(capped_params["motion_strength"]) != 0.85:
        raise RuntimeError("Motion cap (0.85) was not respected.")
    if float(capped_params["cfg"]) != 4.5:
        raise RuntimeError("Guidance floor (4.5) was not respected.")
    if int(capped_steps) != 72:
        raise RuntimeError("Steps cap (72) was not respected.")

    print("WAN adaptive retry wiring test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
