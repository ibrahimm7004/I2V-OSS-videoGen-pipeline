from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.job_schema import JobSpec

try:
    from scripts._job_loading import load_runtime_job_payload
except ModuleNotFoundError:
    from _job_loading import load_runtime_job_payload


IDEAS = ("idea01", "idea02", "idea03")


def _default_run_id(idea: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"wan-{idea}-{stamp}"


def _parse_seed_offsets(raw_offsets: Sequence[int] | None) -> list[int] | None:
    if raw_offsets is None:
        return None
    return [int(item) for item in raw_offsets]


def build_runtime_job_for_idea(
    *,
    idea: str,
    out_root: Path,
    run_id: str,
    dry_run: bool,
    num_clips: int | None = None,
    clip_duration_sec: float | None = None,
    seed_base: int | None = None,
    seed_offsets: Sequence[int] | None = None,
) -> tuple[dict, Path]:
    if idea not in IDEAS:
        raise ValueError(f"Unsupported idea: {idea}")

    job_path = (REPO_ROOT / "jobs" / "wan" / f"{idea}.yaml").resolve()
    if not job_path.exists():
        raise FileNotFoundError(f"WAN job file not found: {job_path}")

    runtime_payload, _, _ = load_runtime_job_payload(
        job_path,
        run_id_override=run_id,
        output_root_override=str(out_root),
        model_id_override=None,
        dry_run_override=dry_run,
        fast_mode=bool(dry_run),
    )
    if not isinstance(runtime_payload, dict):
        raise ValueError("Runtime payload is not a mapping.")

    shots = list(runtime_payload.get("shots") or [])
    if num_clips is not None:
        if num_clips <= 0:
            raise ValueError("--num-clips must be > 0")
        shots = shots[:num_clips]

    if clip_duration_sec is not None:
        if clip_duration_sec <= 0:
            raise ValueError("--clip-duration-sec must be > 0")
        for shot in shots:
            if not isinstance(shot, dict):
                continue
            fps = int(shot.get("fps", 1) or 1)
            shot["duration_seconds"] = float(clip_duration_sec)
            shot["frames"] = max(1, int(round(float(clip_duration_sec) * float(fps))))

    parsed_offsets = _parse_seed_offsets(seed_offsets)
    if seed_base is not None:
        if parsed_offsets is not None and len(parsed_offsets) != len(shots):
            raise ValueError(
                f"--seed-offsets length ({len(parsed_offsets)}) must match effective clip count ({len(shots)})."
            )
        for idx, shot in enumerate(shots):
            offset = parsed_offsets[idx] if parsed_offsets is not None else idx
            if isinstance(shot, dict):
                shot["seed"] = int(seed_base) + int(offset)

    runtime_payload["shots"] = shots
    runtime_payload["run_id"] = run_id
    runtime_payload["job_name"] = run_id
    runtime_payload["output_root"] = str(out_root)
    JobSpec.model_validate(runtime_payload)
    return runtime_payload, job_path


def write_runtime_job_file(*, runtime_payload: dict, out_root: Path, run_id: str, idea: str) -> Path:
    runtime_dir = out_root / run_id / "runtime_jobs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / f"{idea}_runtime.yaml"
    runtime_path.write_text(yaml.safe_dump(runtime_payload, sort_keys=False), encoding="utf-8")
    return runtime_path


def _run_via_run_all(*, runtime_job_path: Path, out_root: Path, dry_run: bool) -> int:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_all.py"),
        "--jobs",
        str(runtime_job_path),
        "--out",
        str(out_root),
        "--stop-on-fail",
    ]
    if dry_run:
        command.append("--dry-run")

    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one WAN idea with optional runtime overrides.")
    parser.add_argument("--idea", required=True, choices=IDEAS, help="WAN idea preset to run.")
    parser.add_argument("--num-clips", type=int, default=None, help="Override clip count at runtime.")
    parser.add_argument(
        "--clip-duration-sec",
        type=float,
        default=None,
        help="Override clip duration seconds at runtime.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output root.")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run via run_all (forces mock adapter).")
    parser.add_argument("--seed-base", type=int, default=None, help="Optional seed base override.")
    parser.add_argument(
        "--seed-offsets",
        type=int,
        nargs="+",
        default=None,
        help="Optional per-clip seed offsets (space-separated ints).",
    )
    parser.add_argument("--run-id", type=str, default=None, help="Optional explicit run id.")
    args = parser.parse_args()

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or _default_run_id(args.idea)

    runtime_payload, source_job = build_runtime_job_for_idea(
        idea=args.idea,
        out_root=out_root,
        run_id=run_id,
        dry_run=bool(args.dry_run),
        num_clips=args.num_clips,
        clip_duration_sec=args.clip_duration_sec,
        seed_base=args.seed_base,
        seed_offsets=args.seed_offsets,
    )
    runtime_job_path = write_runtime_job_file(
        runtime_payload=runtime_payload,
        out_root=out_root,
        run_id=run_id,
        idea=args.idea,
    )

    print(f"Source WAN job: {source_job}")
    print(f"Runtime job: {runtime_job_path}")
    return _run_via_run_all(runtime_job_path=runtime_job_path, out_root=out_root, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
