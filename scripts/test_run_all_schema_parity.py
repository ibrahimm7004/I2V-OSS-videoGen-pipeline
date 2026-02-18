from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression check: verify_jobpacks and run_all share job-pack schema loading.")
    parser.add_argument("--jobs", nargs="+", default=_default_jobs(), help="Job-pack YAMLs.")
    args = parser.parse_args()

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
        "outputs",
        "--stop-on-fail",
        "--model-id-override",
        "mock",
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

    print("Schema parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
