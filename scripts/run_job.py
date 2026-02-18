from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.runner import run_job
from pipeline.utils import parse_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a pipeline job spec.")
    parser.add_argument("job_positional", nargs="?", type=Path, help="Path to job YAML/JSON.")
    parser.add_argument("--job", type=Path, default=None, help="Path to job YAML/JSON.")
    parser.add_argument("--out", type=Path, default=None, help="Override output root directory.")
    parser.add_argument("--run-id", type=str, default=None, help="Explicit run_id override.")
    parser.add_argument("--set", action="append", default=[], help="Override with KEY=VALUE.")
    args = parser.parse_args()

    job_path = args.job or args.job_positional
    if job_path is None:
        parser.error("Provide a job via positional path or --job.")

    overrides = parse_overrides(args.set)
    if args.out is not None:
        overrides["output_root"] = str(args.out)
    if args.run_id is not None:
        overrides["run_id"] = args.run_id

    run_dir = run_job(job_path, overrides=overrides)
    print(f"Run completed: {run_dir}")


if __name__ == "__main__":
    main()
