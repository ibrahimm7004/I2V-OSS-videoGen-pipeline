from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.runner import run_job
from pipeline.utils import safe_slug, write_json

try:
    from scripts._job_loading import load_runtime_job_payload
except ModuleNotFoundError:
    from _job_loading import load_runtime_job_payload


def _default_jobs() -> list[Path]:
    candidates = [
        Path("jobs/idea01_wan.yaml"),
        Path("jobs/idea02_hunyuan.yaml"),
        Path("jobs/idea03_cogvideox.yaml"),
    ]
    if all(path.exists() for path in candidates):
        return candidates
    fallback = Path("jobs/examples/example_mock.yaml")
    return [fallback, fallback, fallback]


def _resolve_bundle(run_dir: Path, manifest_data: dict[str, Any]) -> Path | None:
    bundle_value = (manifest_data.get("outputs") or {}).get("bundle_path")
    if bundle_value is None:
        return None
    path = Path(bundle_value)
    if path.is_absolute():
        return path
    return run_dir / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple job specs sequentially and produce a rollup manifest.")
    parser.add_argument("--jobs", nargs="+", default=None, help="Job specs to run sequentially.")
    parser.add_argument("--out", type=Path, default=Path("outputs"), help="Output root directory.")
    parser.add_argument(
        "--model-id-override",
        type=str,
        default=None,
        help="Override model id for every job (useful for mock runs).",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force dry_run for all jobs. If omitted, job defaults are used.",
    )
    parser.add_argument(
        "--stop-on-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop sequence when a job fails (default: true).",
    )
    args = parser.parse_args()

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    model_id_override = args.model_id_override
    if args.dry_run is True and model_id_override is None:
        model_id_override = "mock"
        print("dry-run enabled; forcing model_id=mock")

    jobs = [Path(item) for item in (args.jobs if args.jobs is not None else _default_jobs())]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    orchestrator_dir = out_root / f"run_all_{timestamp}"
    orchestrator_dir.mkdir(parents=True, exist_ok=True)
    run_all_manifest_path = orchestrator_dir / "run_all_manifest.json"
    generated_jobs_dir = orchestrator_dir / "runtime_jobs"
    generated_jobs_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    failures = 0

    for idx, job_path in enumerate(jobs):
        resolved_job = job_path.resolve()
        if not resolved_job.exists():
            message = f"Job file not found: {resolved_job}"
            records.append(
                {
                    "index": idx,
                    "job_path": str(resolved_job),
                    "run_id": None,
                    "status": "failed",
                    "error": message,
                    "bundle_path": None,
                }
            )
            failures += 1
            if args.stop_on_fail:
                break
            continue

        run_id_fallback = f"{safe_slug(resolved_job.stem)}-{timestamp}-{idx:02d}"
        try:
            runtime_job, source_kind, _ = load_runtime_job_payload(
                resolved_job,
                output_root_override=str(out_root),
                model_id_override=model_id_override,
                dry_run_override=args.dry_run,
                fast_mode=bool(args.dry_run),
            )
        except Exception as exc:
            failures += 1
            records.append(
                {
                    "index": idx,
                    "job_path": str(resolved_job),
                    "run_id": run_id_fallback,
                    "status": "failed",
                    "error": str(exc),
                    "run_dir": None,
                    "bundle_path": None,
                }
            )
            if args.stop_on_fail:
                break
            continue

        run_id = runtime_job.get("run_id") or run_id_fallback
        runtime_job["run_id"] = run_id
        runtime_job["job_name"] = runtime_job.get("job_name") or run_id
        runtime_job["output_root"] = str(out_root)

        runtime_job_path = generated_jobs_dir / f"{idx:02d}_{safe_slug(resolved_job.stem)}.yaml"
        runtime_job_path.write_text(yaml.safe_dump(runtime_job, sort_keys=False), encoding="utf-8")

        try:
            run_dir = run_job(runtime_job_path)
            manifest_path = run_dir / "manifest.json"
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            status = manifest_data.get("status", "failed")
            bundle_path = _resolve_bundle(run_dir, manifest_data)
            bundle_exists = bundle_path is not None and bundle_path.exists()

            error_value = manifest_data.get("error")
            if status != "completed" or not bundle_exists:
                failures += 1
                if error_value is None and not bundle_exists:
                    error_value = "Bundle missing after run completion."

            records.append(
                {
                    "index": idx,
                    "job_path": str(resolved_job),
                    "run_id": manifest_data.get("run_id", run_id),
                    "status": status,
                    "error": error_value,
                    "source_kind": source_kind,
                    "runtime_job_path": str(runtime_job_path),
                    "run_dir": str(run_dir),
                    "bundle_path": str(bundle_path) if bundle_path else None,
                }
            )

            if (status != "completed" or not bundle_exists) and args.stop_on_fail:
                break
        except Exception as exc:
            failures += 1
            records.append(
                {
                    "index": idx,
                    "job_path": str(resolved_job),
                    "run_id": run_id,
                    "status": "failed",
                    "error": str(exc),
                    "source_kind": source_kind,
                    "runtime_job_path": str(runtime_job_path),
                    "run_dir": None,
                    "bundle_path": None,
                }
            )
            if args.stop_on_fail:
                break

    payload = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "output_root": str(out_root),
        "jobs_requested": [str(path.resolve()) for path in jobs],
        "stop_on_fail": bool(args.stop_on_fail),
        "total_jobs": len(records),
        "failed_jobs": failures,
        "status": "completed" if failures == 0 else "failed",
        "runs": records,
    }
    write_json(run_all_manifest_path, payload)
    print(f"run_all manifest: {run_all_manifest_path}")
    print(f"runs completed: {len(records)} failed: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
