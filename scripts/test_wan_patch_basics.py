from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.base import ClipRequest
from models.wan22 import _choose_wan_frame_count, _resolve_export_quality
from pipeline.config import load_config


@contextmanager
def _temporary_env(name: str, value: str | None):
    previous = os.getenv(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _dummy_request(*, params: dict | None = None, global_params: dict | None = None) -> ClipRequest:
    return ClipRequest(
        clip_index=0,
        prompt="test",
        negative_prompt=None,
        input_image=None,
        output_video_path=Path("outputs/_tmp/test.mp4"),
        duration_seconds=5.0,
        fps=24,
        frames=120,
        steps=20,
        seed=123,
        width=1280,
        height=704,
        params=params or {},
        global_params=global_params or {},
        dry_run=True,
        max_runtime_seconds=60,
    )


def main() -> None:
    cases = {
        120: 121,
        119: 121,
        121: 121,
        1: 1,
        2: 5,
    }
    for target, expected in cases.items():
        actual = _choose_wan_frame_count(target)
        _assert(actual == expected, f"4n+1 quantization mismatch: target={target} expected={expected} actual={actual}")

    with _temporary_env("WAN22_EXPORT_QUALITY", "4"):
        cfg = load_config()
        req_yaml = _dummy_request(global_params={"export_quality": 9})
        req_env = _dummy_request()
        req_shot = _dummy_request(params={"export_quality": 7}, global_params={"export_quality": 9})

        _assert(
            _resolve_export_quality(req_yaml, cfg.wan22_export_quality) == 9,
            "Expected YAML/global export_quality to override env default.",
        )
        _assert(
            _resolve_export_quality(req_env, cfg.wan22_export_quality) == 4,
            "Expected env WAN22_EXPORT_QUALITY to apply when YAML value is absent.",
        )
        _assert(
            _resolve_export_quality(req_shot, cfg.wan22_export_quality) == 7,
            "Expected shot params export_quality to override global export_quality.",
        )

    print("WAN patch basic checks passed.")


if __name__ == "__main__":
    main()
