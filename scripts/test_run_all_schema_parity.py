from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, timeout: int = 1200) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _default_jobs() -> list[str]:
    return [
        "jobs/idea01_wan.yaml",
        "jobs/idea02_hunyuan.yaml",
        "jobs/idea03_cogvideox.yaml",
    ]


def _extract_manifest_path(output: str) -> Path:
    prefix = "run_all manifest:"
    for line in output.splitlines():
        if line.startswith(prefix):
            return Path(line[len(prefix) :].strip())
    raise RuntimeError("run_all did not print manifest path.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression check: verify_jobpacks and run_all share job-pack schema loading.")
    parser.add_argument("--jobs", nargs="+", default=_default_jobs(), help="Job-pack YAMLs.")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("outputs") / f"_schema_parity_{timestamp}"

    verify_cmd = [sys.executable, "scripts/verify_jobpacks.py", "--jobs", *args.jobs]
    verify_result = _run(verify_cmd)
    if verify_result.returncode != 0:
        print("verify_jobpacks failed")
        print(verify_result.stdout)
        print(verify_result.stderr)
        return 1

    run_all_cmd = [
        sys.executable,
        "scripts/run_all.py",
        "--jobs",
        *args.jobs,
        "--out",
        str(out_dir),
        "--stop-on-fail",
        "--dry-run",
    ]
    run_all_result = _run(run_all_cmd)
    combined_output = f"{run_all_result.stdout}\n{run_all_result.stderr}"
    if "JobSpec.model" in combined_output or "model\n  Field required" in combined_output:
        print("run_all still hit JobSpec.model schema mismatch")
        print(combined_output)
        return 1
    if run_all_result.returncode != 0:
        print("run_all failed")
        print(combined_output)
        return 1
    if "dry-run enabled; forcing model_id=mock" not in run_all_result.stdout:
        print("run_all dry-run did not announce automatic mock model override")
        print(combined_output)
        return 1
    if "adapter is a scaffold stub" in combined_output:
        print("run_all dry-run still attempted non-mock adapter")
        print(combined_output)
        return 1

    try:
        manifest_path = _extract_manifest_path(run_all_result.stdout)
    except Exception as exc:
        print(f"failed to parse run_all manifest path: {exc}")
        print(combined_output)
        return 1
    if not manifest_path.exists():
        print(f"run_all manifest not found: {manifest_path}")
        return 1

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest_data.get("failed_jobs", -1)) != 0:
        print("run_all dry-run reported failed jobs")
        print(json.dumps(manifest_data, indent=2))
        return 1
    runs = manifest_data.get("runs") or []
    if len(runs) != len(args.jobs):
        print(f"run_all dry-run expected {len(args.jobs)} runs, got {len(runs)}")
        return 1
    for item in runs:
        if item.get("status") != "completed":
            print("run_all dry-run has non-completed run record")
            print(json.dumps(item, indent=2))
            return 1
        runtime_job_path = item.get("runtime_job_path")
        if not isinstance(runtime_job_path, str):
            print("missing runtime_job_path in run record")
            print(json.dumps(item, indent=2))
            return 1
        runtime_path = Path(runtime_job_path)
        if not runtime_path.exists():
            print(f"runtime job file missing: {runtime_path}")
            return 1
        runtime_payload = yaml.safe_load(runtime_path.read_text(encoding="utf-8")) or {}
        model_id = ((runtime_payload.get("model") or {}).get("id")) if isinstance(runtime_payload, dict) else None
        if model_id != "mock":
            print(f"runtime job model.id is not mock: {runtime_path} -> {model_id}")
            return 1
        input_image = runtime_payload.get("input_image") if isinstance(runtime_payload, dict) else None
        if not isinstance(input_image, str):
            print(f"runtime job input_image missing/non-string: {runtime_path} -> {input_image!r}")
            return 1
        input_image_path = Path(input_image)
        if not input_image_path.is_absolute():
            print(f"runtime job input_image is not absolute: {runtime_path} -> {input_image}")
            return 1
        if not input_image_path.exists():
            print(f"runtime job input_image path does not exist: {runtime_path} -> {input_image}")
            return 1

    print("Schema parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
