from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.runner import run_job


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _looks_like_windows_drive(path_text: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", path_text))


def _check_portable_path(value: str | None, field_name: str) -> None:
    if value is None:
        return
    path_obj = Path(value)
    _assert(not path_obj.is_absolute(), f"{field_name} must not be absolute: {value}")
    _assert(not _looks_like_windows_drive(value), f"{field_name} must not contain drive letter: {value}")

    normalized = value.replace("\\", "/")
    _assert(not normalized.startswith("/"), f"{field_name} normalized path is absolute: {value}")
    _assert(normalized == normalized.strip(), f"{field_name} has surrounding whitespace: {value}")
    # Tolerant check: allow backslashes if they normalize cleanly.
    _assert("//" not in normalized, f"{field_name} has malformed separators: {value}")


def run_path_check(run_dir: Path | None = None, job: Path | None = None) -> Path:
    if run_dir is None:
        run_id = "path-check-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = run_job(
            (job or Path("jobs/example_mock.yaml")).resolve(),
            overrides={
                "run_id": run_id,
                "model.id": "mock",
                "dry_run": True,
            },
        )
    run_dir = run_dir.resolve()

    manifest_path = run_dir / "manifest.json"
    _assert(manifest_path.exists(), f"Manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    outputs = manifest.get("outputs", {})
    _check_portable_path(outputs.get("final_stitched_path"), "manifest.outputs.final_stitched_path")
    _check_portable_path(outputs.get("bundle_path"), "manifest.outputs.bundle_path")

    clips = manifest.get("clips", [])
    _assert(bool(clips), "Manifest clips array is empty.")
    first_clip = clips[0]
    _check_portable_path(first_clip.get("output_clip_path"), "manifest.clips[0].output_clip_path")
    _check_portable_path(first_clip.get("last_frame_path"), "manifest.clips[0].last_frame_path")
    _check_portable_path(first_clip.get("input_image_path"), "manifest.clips[0].input_image_path")

    log_path = run_dir / "logs" / "log_000.json"
    _assert(log_path.exists(), f"Log file not found: {log_path}")
    log_data = json.loads(log_path.read_text(encoding="utf-8"))
    _check_portable_path(log_data.get("output_clip_path"), "logs.log_000.output_clip_path")
    _check_portable_path(log_data.get("last_frame_path"), "logs.log_000.last_frame_path")
    _check_portable_path(log_data.get("input_image_path"), "logs.log_000.input_image_path")

    print(f"Path portability check passed. Run dir: {run_dir}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate relative-friendly path storage in run artifacts.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Existing run directory to validate.")
    parser.add_argument("--job", type=Path, default=Path("jobs/example_mock.yaml"))
    args = parser.parse_args()

    try:
        run_path_check(run_dir=args.run_dir, job=args.job)
    except Exception as exc:
        print(f"Path check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

