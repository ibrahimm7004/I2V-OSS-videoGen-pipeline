from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.job_schema import JobSpec
from scripts.run_wan import build_runtime_job_for_idea, write_runtime_job_file


def main() -> int:
    out_root = Path("outputs").resolve()
    run_id = "test-wan-overrides"
    run_dir = out_root / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    runtime_payload, _ = build_runtime_job_for_idea(
        idea="idea01",
        out_root=out_root,
        run_id=run_id,
        dry_run=True,
        num_clips=2,
        clip_duration_sec=2.0,
        seed_base=1000,
        seed_offsets=[0, 7],
    )
    JobSpec.model_validate(runtime_payload)

    shots = runtime_payload.get("shots")
    if not isinstance(shots, list) or len(shots) != 2:
        raise RuntimeError(f"Expected 2 runtime shots, got {len(shots) if isinstance(shots, list) else 'invalid'}")

    for idx, shot in enumerate(shots):
        if not isinstance(shot, dict):
            raise RuntimeError(f"Shot {idx} is not a mapping.")
        fps = int(shot.get("fps", 0))
        duration = float(shot.get("duration_seconds", 0))
        frames = int(shot.get("frames", 0))
        expected_frames = max(1, int(round(duration * fps)))
        if duration != 2.0:
            raise RuntimeError(f"Shot {idx} duration override failed: {duration}")
        if frames != expected_frames:
            raise RuntimeError(f"Shot {idx} frames mismatch: expected {expected_frames}, got {frames}")

    if int(shots[0].get("seed", -1)) != 1000 or int(shots[1].get("seed", -1)) != 1007:
        raise RuntimeError(f"Seed override failed: {[shot.get('seed') for shot in shots]}")

    runtime_job_path = write_runtime_job_file(
        runtime_payload=runtime_payload,
        out_root=out_root,
        run_id=run_id,
        idea="idea01",
    )
    if not runtime_job_path.exists():
        raise RuntimeError(f"Runtime job file not written: {runtime_job_path}")
    expected_parent = out_root / run_id / "runtime_jobs"
    if runtime_job_path.parent.resolve() != expected_parent.resolve():
        raise RuntimeError(
            f"Runtime job path should be under outputs/<run_id>/runtime_jobs, got {runtime_job_path.parent}"
        )

    persisted = yaml.safe_load(runtime_job_path.read_text(encoding="utf-8")) or {}
    JobSpec.model_validate(persisted)
    if len(persisted.get("shots", [])) != 2:
        raise RuntimeError("Persisted runtime job shot count mismatch.")

    print("WAN override runtime job test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
