from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cogvideox15_i2v import _sanitize_frames_to_uint8


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def test_nan_inf_float01() -> None:
    frames = np.random.rand(4, 64, 64, 3).astype(np.float32)
    frames[0, 0, 0, 0] = np.nan
    frames[1, 0, 0, 1] = np.inf
    out, stats = _sanitize_frames_to_uint8(frames)
    arr = np.stack([np.asarray(frame) for frame in out], axis=0)
    _assert(arr.dtype == np.uint8, "Expected uint8 output for float01 sanitize")
    _assert(np.isfinite(arr).all(), "Sanitized frames contain non-finite values")
    _assert(stats["nan_fraction"] > 0.0, "Expected non-zero nan_fraction")
    _assert(stats["value_range_assumption"] in {"0..1", "minmax", "0..255", "-1..1"}, "Unexpected range assumption")


def test_minus1_to1() -> None:
    frames = np.linspace(-1.0, 1.0, num=4 * 32 * 32 * 3, dtype=np.float32).reshape(4, 32, 32, 3)
    out, stats = _sanitize_frames_to_uint8(frames)
    arr = np.stack([np.asarray(frame) for frame in out], axis=0)
    _assert(arr.min() >= 0 and arr.max() <= 255, "Expected uint8 range after -1..1 mapping")
    _assert(stats["value_range_assumption"] in {"-1..1", "0..1"}, "Expected range mapping for -1..1 data")


def test_uint8_tensor_layout() -> None:
    # F,C,H,W shape
    frames = np.random.randint(0, 255, size=(3, 3, 24, 24), dtype=np.uint8)
    out, stats = _sanitize_frames_to_uint8(frames)
    arr = np.stack([np.asarray(frame) for frame in out], axis=0)
    _assert(arr.shape == (3, 24, 24, 3), "Expected F,H,W,C output shape")
    _assert(stats["num_frames_sanitized"] == 3, "Expected frame count to be preserved")


def main() -> int:
    test_nan_inf_float01()
    test_minus1_to1()
    test_uint8_tensor_layout()
    print("Cog sanitize tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
